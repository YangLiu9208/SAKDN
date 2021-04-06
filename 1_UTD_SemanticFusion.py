import os
import sys
import time
import argparse
import random
import logging
import torch
import torchvision
from torchvision import models
import torch.optim as optim
import torchvision.transforms as transforms
import dataset
import model.backbone as backbone
import numpy as np
import metric.loss as loss
import metric.pairsampler as pair
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader

from metric.utils import recall, count_parameters_in_MB, accuracy, AverageMeter
from metric.batchsampler import NPairs, BalancedBatchSampler
from model.embedding import LinearEmbedding
#from kd_losses import *
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--dataset', type=str, default='UTD', choices=['UTD', 'MMAct'])
parser.add_argument('--stu_train_path', type=str, default=r"E:/Multi-modal Action Recognition/UTD-MHAD/Inertial_g_GASF_subject_specific_train/")
parser.add_argument('--stu_test_path', type=str, default=r"E:/Multi-modal Action Recognition/UTD-MHAD/Inertial_g_GASF_subject_specific_test/")
parser.add_argument('--tea_train_path', type=str, default=r"E:/Multi-modal Action Recognition/UTD-MHAD/Inertial_a_GASF_subject_specific_train/")
parser.add_argument('--tea_test_path', type=str, default=r"E:/Multi-modal Action Recognition/UTD-MHAD/Inertial_a_GASF_subject_specific_test/")
parser.add_argument('--modality', type=str, default='a', choices=['a', 'g'])
parser.add_argument('--output_dir', type=str, default='output/')
parser.add_argument('--mode',
                        choices=["train", "eval"],
                        default="train")

parser.add_argument('--load',
                        default=None)

parser.add_argument('--base',
                        choices=dict(googlenet=backbone.GoogleNet,
                                    inception_v1bn=backbone.InceptionV1BN,
                                    resnet18=backbone.ResNet18,
                                    resnet50=backbone.ResNet50,
                                    vggnet16=backbone.VggNet16,
                                    Sevggnet16=backbone.SeVggNet16,
                                    SeFusionVGG16=backbone.SeFusionVGG16,
                                    SemanticFusionVGG16=backbone.SemanticFusionVGG16,
                                    ),
                        default=backbone.VggNet16,
                        action=LookupChoices)

parser.add_argument('--sample',
                        choices=dict(random=pair.RandomNegative,
                                    hard=pair.HardNegative,
                                    all=pair.AllPairs,
                                    semihard=pair.SemiHardNegative,
                                    distance=pair.DistanceWeighted),
                        default=pair.AllPairs,
                        action=LookupChoices)

parser.add_argument('--loss',
                        choices=dict(l1_triplet=loss.L1Triplet,
                                    l2_triplet=loss.L2Triplet,
                                    contrastive=loss.ContrastiveLoss),
                        default=loss.L2Triplet,
                        action=LookupChoices)
parser.add_argument('--num_classes', default=27, type=int)
parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--l2normalize', choices=['true', 'false'], default='true')

parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--lr_decay_epochs', type=int, default=[25, 30, 35], nargs='+')
parser.add_argument('--lr_decay_gamma', default=0.5, type=float)

parser.add_argument('--batch', default=64, type=int)
parser.add_argument('--num_image_per_class', default=5, type=int)

parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--iter_per_epoch', type=int, default=100)
parser.add_argument('--recall', default=[1], type=int, nargs='+')

parser.add_argument('--seed', default=random.randint(1, 1000), type=int)
parser.add_argument('--data', default='data')
parser.add_argument('--save_dir', default=None)
parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')

opts = parser.parse_args()
opts.dataset='UTD_L1_subject_specific'
#opts.load=r"E:/Multi-modal Action Recognition/My codes/Relational Knowledge Distillation/output/UTD_a_g_SemanticFusionVGG16_margin0.2_epochs100_batch16_lr0.0001/tea_best_acc.pth"
opts.num_classes=27
opts.mode='train'
opts.modality='a_g'
opts.base=backbone.SemanticFusionVGG16
opts.sample=pair.DistanceWeighted
opts.loss=loss.L2Triplet
opts.lr=0.0002
opts.margin=0.2
opts.batch=16
opts.epochs=100
opts.lr_decay_epochs=[] 
opts.lr_decay_gamma=0.5
#opts.embedding_size=256
opts.print_freq=1
opts.output_dir='output/'
opts.save_dir= opts.output_dir+'_'.join(map(str, [opts.dataset, opts.modality, 'SemanticFusionVGG16', 
            'margin'+str(opts.margin), 'epochs'+str(opts.epochs),'batch'+str(opts.batch), 'lr'+str(opts.lr)]))
if not os.path.exists(opts.save_dir):
    os.makedirs(opts.save_dir)
log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(opts.save_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def loadtraindata(data_path):
    path = data_path                                         # 路径
    trainset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((224, 224)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    #transforms.RandomHorizontalFlip(),
                                                    #transforms.CenterCrop(64),
                                                    transforms.ToTensor(),
                                                    #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                                    #std  = [ 0.229, 0.224, 0.225 ]),
                                                ])
                                                )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opts.batch,
                                              shuffle=True, num_workers=2)                                        
    return trainloader

def loadtestdata(data_path):
    path = data_path
    testset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((224, 224)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    #transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                                    #std  = [ 0.229, 0.224, 0.225 ]),
                                                    ])
                                                )
    testloader = torch.utils.data.DataLoader(testset, batch_size=opts.batch,
                                             shuffle=False, num_workers=2)
    return testloader

def main():

    for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
        set_random_seed(opts.seed)
    logging.info("args = %s", opts)
    tea_train_loader = loadtraindata(opts.tea_train_path)
    tea_test_loader = loadtestdata(opts.tea_test_path)
    stu_train_loader = loadtraindata(opts.stu_train_path)
    stu_test_loader = loadtestdata(opts.stu_test_path)
    UTD_Glove=np.load('data/UTD_Glove.npy')
    UTD_Glove=torch.from_numpy(UTD_Glove)
    UTD_Glove=UTD_Glove.float().cuda()
    #UTD_Glove=F.normalize(UTD_Glove, p=2, dim=1)
    torch.cuda.empty_cache()
    logging.info('----------- Network Initialization --------------')
    model = opts.base(n_classes=opts.num_classes).cuda()
    logging.info('Teacher: %s', model)
    logging.info('Teacher param size = %fMB', count_parameters_in_MB(model))
    logging.info('-----------------------------------------------')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
    if opts.load is not None:
        model.load_state_dict(torch.load(opts.load))
        print("Loaded Model from %s" % opts.load)
    #base_model = opts.base(pretrained=True)
    #if isinstance(base_model, backbone.InceptionV1BN) or isinstance(base_model, backbone.GoogleNet):
    #    normalize = transforms.Compose([
    #        transforms.Lambda(lambda x: x[[2, 1, 0], ...] * 255.0),
    #        transforms.Normalize(mean=[104, 117, 128], std=[1, 1, 1]),
    #    ])
    #else:
    #    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    logging.info("Number of images in Teacher Training Set: %d" % len(tea_train_loader.dataset))
    logging.info("Number of images in Teacher Testing set: %d" % len(tea_test_loader.dataset))
    logging.info("Number of images in Student Training Set: %d" % len(stu_train_loader.dataset))
    logging.info("Number of images in Student Testing set: %d" % len(stu_test_loader.dataset))

    #model = LinearEmbedding(base_model,
    #                        output_size=base_model.output_size,
    #                        embedding_size=opts.embedding_size,
    #                        normalize=opts.l2normalize == 'true').cuda()

    if opts.load is not None:
        model.load_state_dict(torch.load(opts.load))
        logging.info("Loaded Model from %s" % opts.load)

    #criterion = opts.loss(sampler=opts.sample(), margin=opts.margin)
    criterion_cls=torch.nn.CrossEntropyLoss().cuda()
    criterion_semantic=torch.nn.L1Loss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-5)
    #optimizer = torch.optim.SGD(model.parameters(),
    #                            opts.lr,
    #                            momentum=0.9,
    #                            weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma)

    def train(net, t_loader, s_loader,ep):
        K = opts.recall
        batch_time = AverageMeter()
        data_time = AverageMeter()
        #triplet_loss = AverageMeter()
        cls_loss = AverageMeter()
        semantic_loss = AverageMeter()
        #top1_recall = AverageMeter()
        #top1_prec = AverageMeter()
        s_train_acc=0.
        t_train_acc=0.
        net.train()
        loss_all = []
        #train_iter = tqdm(loader, ncols=80)
        end = time.time()
        i=1
        for (t_images, t_labels),(s_images, s_labels) in zip(t_loader,s_loader):
            data_time.update(time.time() - end)
            t_images, t_labels = t_images.cuda(), t_labels.cuda()
            s_images, s_labels = s_images.cuda(), s_labels.cuda()
            t_semantic=UTD_Glove[t_labels]
            s_semantic=UTD_Glove[s_labels]
            s_out1,t_out1,s_out2,t_out2,s_out3,t_out3,s_out4,t_out4,s_out5,t_out5,s_out6,t_out6,s_out7,t_out7,s_out8,t_out8 = net(s_images,t_images,True)
            
            s_pred=torch.max(s_out8,1)[1]
            t_pred=torch.max(t_out8,1)[1]

            s_train_correct=(s_pred==s_labels).sum()
            t_train_correct=(t_pred==t_labels).sum()

            s_train_acc+=s_train_correct.item()
            t_train_acc+=t_train_correct.item()

            #loss_triplet = criterion(embedding, labels)
            s_loss_cls=criterion_cls(s_out8, s_labels)
            t_loss_cls=criterion_cls(t_out8, t_labels)
            s_semantic_loss=criterion_semantic(s_out7, s_semantic)
            t_semantic_loss=criterion_semantic(t_out7, t_semantic)
            loss=(s_loss_cls+t_loss_cls)/2.0+(s_semantic_loss+t_semantic_loss)/2.0 #+loss_triplet
            loss_all.append(loss.item())
            
            #rec = recall(embedding, labels, K=K)
            #prec = accuracy(embedding, labels, topk=(1,))
            #triplet_loss.update(loss_triplet.item(), images.size(0))
            cls_loss.update((s_loss_cls.item()+t_loss_cls.item()), s_images.size(0)+t_images.size(0))
            semantic_loss.update((s_semantic_loss.item()+t_semantic_loss.item()), s_images.size(0)+t_images.size(0))
            #top1_recall.update(rec[0], images.size(0))
            #top1_prec.update(prec[0]/100, images.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()
            if i % opts.print_freq == 0:
                log_str=('Epoch[{0}]:[{1:03}/{2:03}] '
                        'Batch:{batch_time.val:.4f} '
                        'Data:{data_time.val:.4f}  '
                        'Cls Loss:{loss_cls.val:.4f}({loss_cls.avg:.4f})  '
                        'Semantic Loss:{loss_semantic.val:.4f}({loss_semantic.avg:.4f})  '
                        #'Triplet:{loss_triplet.val:.4f}({loss_triplet.avg:.4f})  '
                        #'recall@1:{top1_recall.val:.2f}({top1_recall.avg:.2f})  '
                        #'pre@1:{top1_prec.val:.2f}({top1_prec.avg:.2f})  '.
                        .format(ep, i, len(s_loader), batch_time=batch_time, data_time=data_time,
                        loss_cls=cls_loss, loss_semantic=semantic_loss))
                logging.info(log_str)
            i=i+1
            #train_iter.set_description("[Train][Epoch %d] Loss: %.5f" % (ep, loss.item()))
        logging.info('[Epoch %d] Loss: %.5f  Student Acc: %.5f Teacher Acc: %.5f' % (ep, torch.Tensor(loss_all).mean(), 100*s_train_acc/(len(s_loader.dataset)), 100*t_train_acc/(len(t_loader.dataset))))


    def eval(net, t_loader, s_loader,ep):
        #K = opts.recall
        net.eval()
        #test_iter = tqdm(loader, ncols=80)
        s_embeddings_all, t_embeddings_all,s_labels_all,t_labels_all = [], [], [], []
        s_correct = 0
        t_correct = 0
        st_correct = 0
        #test_iter.set_description("[Eval][Epoch %d]" % ep)
        with torch.no_grad():
            for (t_images, t_labels),(s_images, s_labels) in zip(t_loader,s_loader):
                t_images, t_labels = t_images.cuda(), t_labels.cuda()
                s_images, s_labels = s_images.cuda(), s_labels.cuda()
                t_semantic=UTD_Glove[t_labels]
                s_semantic=UTD_Glove[s_labels]
                s_embedding, t_embedding = net(s_images,t_images)
                s_pred=torch.max(s_embedding,1)[1]
                t_pred=torch.max(t_embedding,1)[1]
                st_pred=torch.max((s_embedding+t_embedding),1)[1]
                s_num_correct=(s_pred==s_labels).sum()
                t_num_correct=(t_pred==t_labels).sum()
                st_num_correct=(st_pred==t_labels).sum()
                s_correct+=s_num_correct.item()
                t_correct+=t_num_correct.item()
                st_correct+=st_num_correct.item()
                s_embeddings_all.append(s_embedding.data)
                t_embeddings_all.append(t_embedding.data)
                s_labels_all.append(s_labels.data)  
                t_labels_all.append(t_labels.data)
            s_embeddings_all = torch.cat(s_embeddings_all).cpu()
            t_embeddings_all = torch.cat(t_embeddings_all).cpu()
            s_labels_all = torch.cat(s_labels_all).cpu()
            t_labels_all = torch.cat(t_labels_all).cpu()
            #rec = recall(embeddings_all, labels_all, K=K)
            #s_prec = accuracy(s_embeddings_all, s_labels_all, topk=(1,))
            s_acc = s_correct/(len(s_loader.dataset))
            t_acc = t_correct/(len(t_loader.dataset))
            st_acc= st_correct/(len(t_loader.dataset))
            logging.info('[Epoch %d] student acc: [%.4f] teacher acc: [%.4f] combined acc: [%.4f]' % (ep, s_acc*100, t_acc*100, st_acc*100))    
        return s_acc, t_acc ,st_acc


    if opts.mode == "eval":
        eval(model, tea_test_loader,stu_test_loader, 0)
    else:
        stu_val_acc, tea_val_acc , st_val_acc= eval(model, tea_test_loader,stu_test_loader, 0)
        stu_best_acc =stu_val_acc
        tea_best_acc =tea_val_acc
        st_best_acc =st_val_acc
        for epoch in range(1, opts.epochs+1):
            train(model, tea_train_loader,stu_train_loader, epoch)
            stu_val_acc, tea_val_acc ,st_val_acc= eval(model, tea_test_loader,stu_test_loader,epoch)
            if stu_best_acc < stu_val_acc:
                stu_best_acc = stu_val_acc
                if opts.save_dir is not None:
                    if not os.path.isdir(opts.save_dir):
                        os.mkdir(opts.save_dir)
                    torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "stu_best_acc.pth"))
            if tea_best_acc < tea_val_acc:
                tea_best_acc = tea_val_acc
                if opts.save_dir is not None:
                    if not os.path.isdir(opts.save_dir):
                        os.mkdir(opts.save_dir)
                    torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "tea_best_acc.pth"))
            if st_best_acc < st_val_acc:
                st_best_acc = st_val_acc
                if opts.save_dir is not None:
                    if not os.path.isdir(opts.save_dir):
                        os.mkdir(opts.save_dir)
                    torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "st_best_acc.pth"))
            #F_measure=(2*best_prec/100*best_rec)/(best_prec/100+best_rec)
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "last.pth"))
                with open("%s/result.txt"%opts.save_dir, 'w') as f:
                    #f.write("Best recall@1: %.4f\n" % (best_rec * 100))
                    #f.write("Best prec@1: %.4f\n" % (best_prec))
                    f.write("Best student acc: %.4f\n" % (stu_best_acc*100))
                    f.write("Best teacher acc: %.4f\n" % (tea_best_acc*100))
                    f.write("Best combined acc: %.4f\n" % (st_best_acc*100))
                    #f.write("Final recall@1: %.4f\n" % (val_recall * 100))
                    #f.write("Final Prec@1: %.4f\n" % (val_prec))
                    f.write("Final student acc: %.4f\n" % (stu_val_acc*100))
                    f.write("Final teacher acc: %.4f\n" % (tea_val_acc*100))
                    f.write("Final combined acc: %.4f\n" % (st_val_acc*100))
                    #f.write("F-measure: %.4f\n" % (F_measure*100))

            #logging.info("Best Recall@1: %.4f" % (best_rec*100))
            #logging.info("Best Prec@1: %.4f" % best_prec)
            logging.info("Best Student Acc: %.4f" % (stu_best_acc*100))
            logging.info("Best Teacher Acc: %.4f\n" % (tea_best_acc*100))
            logging.info("Best combined Acc: %.4f\n" % (st_best_acc*100))
            #logging.info("F-measure: %.4f" % (F_measure*100))




if __name__ == '__main__':
    main()