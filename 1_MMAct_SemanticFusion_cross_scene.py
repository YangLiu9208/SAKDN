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
from metric.batchsampler import NPairs
from model.embedding import LinearEmbedding
from itertools import cycle
#from kd_losses import *
import warnings
warnings.filterwarnings("ignore")
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--dataset', type=str, default='UTD', choices=['UTD', 'MMAct'])
parser.add_argument('--acc_phone_train_path', type=str, default=r"E:/Multi-modal Action Recognition/MMAct_sensors/acc_phone_cross_scene_train/")
parser.add_argument('--acc_phone_test_path', type=str, default=r"E:/Multi-modal Action Recognition/MMAct_sensors/acc_phone_cross_scene_test/")
parser.add_argument('--acc_watch_train_path', type=str, default=r"E:/Multi-modal Action Recognition/MMAct_sensors/acc_watch_cross_scene_train/")
parser.add_argument('--acc_watch_test_path', type=str, default=r"E:/Multi-modal Action Recognition/MMAct_sensors/acc_watch_cross_scene_test/")
parser.add_argument('--gyro_train_path', type=str, default=r"E:/Multi-modal Action Recognition/MMAct_sensors/gyro_cross_scene_train/")
parser.add_argument('--gyro_test_path', type=str, default=r"E:/Multi-modal Action Recognition/MMAct_sensors/gyro_cross_scene_test/")
parser.add_argument('--orientation_train_path', type=str, default=r"E:/Multi-modal Action Recognition/MMAct_sensors/orientation_cross_scene_train/")
parser.add_argument('--orientation_test_path', type=str, default=r"E:/Multi-modal Action Recognition/MMAct_sensors/orientation_cross_scene_test/")
parser.add_argument('--modality', type=str, default='a', choices=['a_p','a_w','g','o'])
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
                                    SemanticFusionVGG16_MMAct=backbone.SemanticFusionVGG16_MMAct,
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
opts.dataset='MMAct_cross_scene'
#opts.load=r"E:/Multi-modal Action Recognition/My codes/Relational Knowledge Distillation/output/UTD_a_g_SemanticFusionVGG16_margin0.2_epochs100_batch16_lr0.0001/tea_best_acc.pth"
opts.num_classes=36
opts.mode='train'
opts.modality='a_g_o'
opts.base=backbone.SemanticFusionVGG16_MMAct
opts.sample=pair.DistanceWeighted
opts.loss=loss.L2Triplet
opts.lr=0.0001
opts.margin=0.2
opts.batch=16
opts.epochs=70
opts.lr_decay_epochs=[50] 
opts.lr_decay_gamma=0.5
#opts.embedding_size=256
opts.print_freq=1
opts.output_dir='output/'
opts.save_dir= opts.output_dir+'_'.join(map(str, [opts.dataset, opts.modality, 'SemanticFusionVGG16_MMAct', 
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
                                              shuffle=True, num_workers=2, drop_last=True)
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
                                             shuffle=False, num_workers=2,drop_last=True)
    return testloader

def main():

    for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
        set_random_seed(opts.seed)
    logging.info("args = %s", opts)

    acc_phone_train_loader = loadtraindata(opts.acc_phone_train_path)
    acc_phone_test_loader = loadtestdata(opts.acc_phone_test_path)
    acc_watch_train_loader = loadtraindata(opts.acc_watch_train_path)
    acc_watch_test_loader = loadtestdata(opts.acc_watch_test_path)
    gyro_train_loader = loadtraindata(opts.gyro_train_path)
    gyro_test_loader = loadtestdata(opts.gyro_test_path)
    ori_train_loader = loadtraindata(opts.orientation_train_path)
    ori_test_loader = loadtestdata(opts.orientation_test_path)

    MMAct_Glove=np.load('data/MMAct_Glove.npy')
    MMAct_Glove=torch.from_numpy(MMAct_Glove)
    MMAct_Glove=MMAct_Glove.float().cuda()
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

    logging.info("Number of images in Acc Phone Training Set: %d" % len(acc_phone_train_loader.dataset))
    logging.info("Number of images in Acc Phone Testing set: %d" % len(acc_phone_test_loader.dataset))
    logging.info("Number of images in Acc Watch Training Set: %d" % len(acc_watch_train_loader.dataset))
    logging.info("Number of images in Acc Watch Testing set: %d" % len(acc_watch_test_loader.dataset))
    logging.info("Number of images in Gyro Training Set: %d" % len(gyro_train_loader.dataset))
    logging.info("Number of images in Gyro Testing set: %d" % len(gyro_test_loader.dataset))
    logging.info("Number of images in Ori Training Set: %d" % len(ori_train_loader.dataset))
    logging.info("Number of images in Ori Testing set: %d" % len(ori_test_loader.dataset))


    if opts.load is not None:
        model.load_state_dict(torch.load(opts.load))
        logging.info("Loaded Model from %s" % opts.load)

    #criterion = opts.loss(sampler=opts.sample(), margin=opts.margin)
    criterion_cls=torch.nn.CrossEntropyLoss().cuda()
    criterion_semantic=torch.nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-5)
    #optimizer = torch.optim.SGD(model.parameters(),
    #                            opts.lr,
    #                            momentum=0.9,
    #                            weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma)

    def train(net, ap_loader,aw_loader,gyro_loader,ori_loader,ep):
        K = opts.recall
        batch_time = AverageMeter()
        data_time = AverageMeter()
        cls_loss = AverageMeter()
        semantic_loss = AverageMeter()
        ap_train_acc=0.
        aw_train_acc=0.
        gyro_train_acc=0.
        ori_train_acc=0.
        net.train()
        loss_all = []
        end = time.time()
        i=1
        torch.cuda.empty_cache() 
        for (ap_images, ap_labels),(aw_images, aw_labels),(gyro_images, gyro_labels),(ori_images, ori_labels) in zip(ap_loader,cycle(aw_loader),gyro_loader,ori_loader):

            data_time.update(time.time() - end)

            ap_images, ap_labels = ap_images.cuda(), ap_labels.cuda()
            aw_images, aw_labels = aw_images.cuda(), aw_labels.cuda()
            gyro_images, gyro_labels = gyro_images.cuda(), gyro_labels.cuda()
            ori_images, ori_labels = ori_images.cuda(), ori_labels.cuda()

            ap_semantic=MMAct_Glove[ap_labels]
            aw_semantic=MMAct_Glove[aw_labels]
            gyro_semantic=MMAct_Glove[gyro_labels]
            ori_semantic=MMAct_Glove[ori_labels]

            ap_out1, aw_out1, gy_out1, ori_out1,\
            ap_out2, aw_out2, gy_out2, ori_out2,\
            ap_out3, aw_out3, gy_out3, ori_out3,\
            ap_out4, aw_out4, gy_out4, ori_out4,\
            ap_out5, aw_out5, gy_out5, ori_out5,\
            ap_out6, aw_out6, gy_out6,ori_out6,\
            ap_out7, aw_out7, gy_out7, ori_out7,\
            ap_out8, aw_out8, gy_out8, ori_out8= \
            net(ap_images,aw_images, gyro_images,ori_images,True)
            
            ap_pred=torch.max(ap_out8,1)[1]
            aw_pred=torch.max(aw_out8,1)[1]
            gyro_pred=torch.max(gy_out8,1)[1]
            ori_pred=torch.max(ori_out8,1)[1]

            ap_num_correct=(ap_pred==ap_labels).sum()
            aw_num_correct=(aw_pred==aw_labels).sum()
            gyro_num_correct=(gyro_pred==gyro_labels).sum()
            ori_num_correct=(ori_pred==ori_labels).sum()

            ap_train_acc+=ap_num_correct.item()
            aw_train_acc+=aw_num_correct.item()
            gyro_train_acc+=gyro_num_correct.item()
            ori_train_acc+=ori_num_correct.item()

            #loss_triplet = criterion(embedding, labels)
            ap_loss_cls=criterion_cls(ap_out8, ap_labels)
            aw_loss_cls=criterion_cls(aw_out8, aw_labels)
            gyro_loss_cls=criterion_cls(gy_out8, gyro_labels)
            ori_loss_cls=criterion_cls(ori_out8, ori_labels)

            ap_semantic_loss=criterion_semantic(ap_out7, ap_semantic)
            aw_semantic_loss=criterion_semantic(aw_out7, aw_semantic)
            gyro_semantic_loss=criterion_semantic(gy_out7, gyro_semantic)
            ori_semantic_loss=criterion_semantic(ori_out7, ori_semantic)

            loss=(ap_loss_cls+aw_loss_cls+gyro_loss_cls+ori_loss_cls+ap_semantic_loss+aw_semantic_loss+gyro_semantic_loss+ori_semantic_loss)/8.0 #+loss_triplet
            loss_all.append(loss.item())
            
            #rec = recall(embedding, labels, K=K)
            #prec = accuracy(embedding, labels, topk=(1,))
            #triplet_loss.update(loss_triplet.item(), images.size(0))
            cls_loss.update((ap_loss_cls.item()+aw_loss_cls.item()+gyro_loss_cls.item()+ori_loss_cls.item())/4.0,\
                 ap_images.size(0)+gyro_images.size(0)+ori_images.size(0))
            semantic_loss.update((ap_semantic_loss.item()+aw_semantic_loss.item()+gyro_semantic_loss.item()+ori_semantic_loss.item())/4.0, \
                 ap_images.size(0)+gyro_images.size(0)+ori_images.size(0))
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
                        .format(ep, i, len(ap_loader), batch_time=batch_time, data_time=data_time,
                        loss_cls=cls_loss, loss_semantic=semantic_loss))
                logging.info(log_str)
            i=i+1
            #train_iter.set_description("[Train][Epoch %d] Loss: %.5f" % (ep, loss.item()))
        logging.info('[Epoch %d] Loss: %.5f Ap Acc: %.5f Aw Acc: %.5f Gyro Acc: %.5f Ori Acc: %.5f' % \
             (ep, torch.Tensor(loss_all).mean(), 100*ap_train_acc/(len(ap_loader.dataset)), 100*aw_train_acc/(len(aw_loader.dataset)),\
                  100*gyro_train_acc/(len(gyro_loader.dataset)),  100*ori_train_acc/(len(ori_loader.dataset))))


    def eval(net, ap_loader,aw_loader,gyro_loader,ori_loader, ep):
        #K = opts.recall
        net.eval()
        #test_iter = tqdm(loader, ncols=80)
        ap_embeddings_all, aw_embeddings_all,gyro_embeddings_all,ori_embeddings_all,\
        ap_labels_all,aw_labels_all,gyro_labels_all,ori_labels_all = [], [], [], [], [], [], [], []
        ap_correct = 0
        aw_correct = 0
        gyro_correct = 0
        ori_correct = 0
        all_correct = 0
        #test_iter.set_description("[Eval][Epoch %d]" % ep)
        with torch.no_grad():
            for (ap_images, ap_labels), (aw_images, aw_labels),(gyro_images, gyro_labels),(ori_images, ori_labels) in zip(ap_loader,cycle(aw_loader),gyro_loader,ori_loader):

                ap_images, ap_labels = ap_images.cuda(), ap_labels.cuda()
                aw_images, aw_labels = aw_images.cuda(), aw_labels.cuda()
                gyro_images, gyro_labels = gyro_images.cuda(), gyro_labels.cuda()
                ori_images, ori_labels = ori_images.cuda(), ori_labels.cuda()

                ap_semantic=MMAct_Glove[ap_labels]
                aw_semantic=MMAct_Glove[aw_labels]
                gyro_semantic=MMAct_Glove[gyro_labels]
                ori_semantic=MMAct_Glove[ori_labels]

                ap_embedding,aw_embedding,gyro_embedding,ori_embedding = net(ap_images,aw_images,gyro_images,ori_images)

                ap_pred=torch.max(ap_embedding,1)[1]
                aw_pred=torch.max(aw_embedding,1)[1]
                gyro_pred=torch.max(gyro_embedding,1)[1]
                ori_pred=torch.max(ori_embedding,1)[1]
                all_pred=torch.max((ap_embedding+aw_embedding+gyro_embedding+ori_embedding),1)[1]

                ap_num_correct=(ap_pred==ap_labels).sum()
                aw_num_correct=(aw_pred==ap_labels).sum()
                gyro_num_correct=(gyro_pred==gyro_labels).sum()
                ori_num_correct=(ori_pred==ori_labels).sum()
                all_num_correct=(all_pred==ap_labels).sum()

                ap_correct+=ap_num_correct.item()
                aw_correct+=aw_num_correct.item()
                gyro_correct+=gyro_num_correct.item()
                ori_correct+=ori_num_correct.item()
                all_correct+=all_num_correct.item()

                ap_embeddings_all.append(ap_embedding.data)
                aw_embeddings_all.append(aw_embedding.data)
                gyro_embeddings_all.append(gyro_embedding.data)
                ori_embeddings_all.append(ori_embedding.data)

                ap_labels_all.append(ap_labels.data)  
                aw_labels_all.append(aw_labels.data)
                gyro_labels_all.append(gyro_labels.data)  
                ori_labels_all.append(ori_labels.data)

            ap_embeddings_all = torch.cat(ap_embeddings_all).cpu()
            aw_embeddings_all = torch.cat(aw_embeddings_all).cpu()
            gyro_embeddings_all = torch.cat(gyro_embeddings_all).cpu()
            ori_embeddings_all = torch.cat(ori_embeddings_all).cpu()

            ap_labels_all = torch.cat(ap_labels_all).cpu()
            aw_labels_all = torch.cat(aw_labels_all).cpu()
            gyro_labels_all = torch.cat(gyro_labels_all).cpu()
            ori_labels_all = torch.cat(ori_labels_all).cpu()

            #rec = recall(embeddings_all, labels_all, K=K)
            #s_prec = accuracy(s_embeddings_all, s_labels_all, topk=(1,))

            ap_acc = ap_correct/(len(ap_loader.dataset))
            aw_acc = aw_correct/(len(aw_loader.dataset))
            gyro_acc = gyro_correct/(len(gyro_loader.dataset))
            ori_acc = ori_correct/(len(ori_loader.dataset))
            all_acc = all_correct/(len(ap_loader.dataset))

            logging.info('[Epoch %d] Ap acc: [%.4f] Aw acc: [%.4f] Gyro acc: [%.4f] Ori acc: [%.4f] Combined acc: [%.4f]' \
                % (ep, ap_acc*100, aw_acc*100, gyro_acc*100, ori_acc*100, all_acc*100))    
        return ap_acc, aw_acc, gyro_acc, ori_acc, all_acc


    if opts.mode == "eval":
        eval(model, acc_phone_test_loader,acc_watch_test_loader, gyro_test_loader,ori_test_loader, 0)
    else:
        ap_val_acc, aw_val_acc,gyro_val_acc, ori_val_acc, all_val_acc = eval(model, acc_phone_test_loader,acc_watch_test_loader,gyro_test_loader,ori_test_loader, 0)
        ap_best_acc =ap_val_acc
        aw_best_acc =aw_val_acc
        gyro_best_acc =gyro_val_acc
        ori_best_acc =ori_val_acc
        all_best_acc =all_val_acc
        for epoch in range(1, opts.epochs+1):
            train(model, acc_phone_train_loader, acc_watch_train_loader, gyro_train_loader,ori_train_loader, epoch)
            ap_val_acc, aw_val_acc, gyro_val_acc, ori_val_acc, all_val_acc = eval(model, acc_phone_test_loader,acc_watch_test_loader,gyro_test_loader,ori_test_loader, epoch)
            if ap_best_acc < ap_val_acc:
                ap_best_acc = ap_val_acc
                if opts.save_dir is not None:
                    if not os.path.isdir(opts.save_dir):
                        os.mkdir(opts.save_dir)
                    torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "ap_best_acc.pth"))
            if aw_best_acc < aw_val_acc:
                aw_best_acc = aw_val_acc
                if opts.save_dir is not None:
                    if not os.path.isdir(opts.save_dir):
                        os.mkdir(opts.save_dir)
                    torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "aw_best_acc.pth"))
            if gyro_best_acc < gyro_val_acc:
                gyro_best_acc = gyro_val_acc
                if opts.save_dir is not None:
                    if not os.path.isdir(opts.save_dir):
                        os.mkdir(opts.save_dir)
                    torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "gyro_best_acc.pth"))
            if ori_best_acc < ori_val_acc:
                ori_best_acc = ori_val_acc
                if opts.save_dir is not None:
                    if not os.path.isdir(opts.save_dir):
                        os.mkdir(opts.save_dir)
                    torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "ori_best_acc.pth"))
            if all_best_acc < all_val_acc:
                all_best_acc = all_val_acc
                if opts.save_dir is not None:
                    if not os.path.isdir(opts.save_dir):
                        os.mkdir(opts.save_dir)
                    torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "all_best_acc.pth"))
            #F_measure=(2*best_prec/100*best_rec)/(best_prec/100+best_rec)
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "last.pth"))
                with open("%s/result.txt"%opts.save_dir, 'w') as f:
                    #f.write("Best recall@1: %.4f\n" % (best_rec * 100))
                    #f.write("Best prec@1: %.4f\n" % (best_prec))
                    f.write("Best Acc Phone acc: %.4f\n" % (ap_best_acc*100))
                    f.write("Best Acc Watch acc: %.4f\n" % (aw_best_acc*100))
                    f.write("Best Gyro acc: %.4f\n" % (gyro_best_acc*100))
                    f.write("Best Ori acc: %.4f\n" % (ori_best_acc*100))
                    f.write("Best Combined acc: %.4f\n" % (all_best_acc*100))
                    #f.write("Final recall@1: %.4f\n" % (val_recall * 100))
                    #f.write("Final Prec@1: %.4f\n" % (val_prec))
                    f.write("Final Acc Phone acc: %.4f\n" % (ap_val_acc*100))
                    f.write("Final Acc Watch acc: %.4f\n" % (aw_val_acc*100))
                    f.write("Final Gyro acc: %.4f\n" % (gyro_val_acc*100))
                    f.write("Final Ori acc: %.4f\n" % (ori_val_acc*100))
                    f.write("Final Combined acc: %.4f\n" % (all_val_acc*100))
                    #f.write("F-measure: %.4f\n" % (F_measure*100))

            #logging.info("Best Recall@1: %.4f" % (best_rec*100))
            #logging.info("Best Prec@1: %.4f" % best_prec)
            logging.info("Best Acc Phone acc: %.4f" % (ap_best_acc*100))
            logging.info("Best Acc Watch acc: %.4f" % (aw_best_acc*100))
            logging.info("Best Gyro acc: %.4f" % (gyro_best_acc*100))
            logging.info("Best Ori acc: %.4f\n" % (ori_best_acc*100))
            logging.info("Best Combined acc: %.4f\n" % (all_best_acc*100))
            #logging.info("F-measure: %.4f" % (F_measure*100))




if __name__ == '__main__':
    main()