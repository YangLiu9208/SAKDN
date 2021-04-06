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
parser.add_argument('--acc_one_train_path', type=str, default=r"D:/Berkeley-MHAD/Accelerometer/Shimmer01_GAF_train/")
parser.add_argument('--acc_one_test_path', type=str, default=r"D:/Berkeley-MHAD/Accelerometer/Shimmer01_GAF_test/")
parser.add_argument('--acc_two_train_path', type=str, default=r"D:/Berkeley-MHAD/Accelerometer/Shimmer02_GAF_train/")
parser.add_argument('--acc_two_test_path', type=str, default=r"D:/Berkeley-MHAD/Accelerometer/Shimmer02_GAF_test/")
parser.add_argument('--acc_three_train_path', type=str, default=r"D:/Berkeley-MHAD/Accelerometer/Shimmer03_GAF_train/")
parser.add_argument('--acc_three_test_path', type=str, default=r"D:/Berkeley-MHAD/Accelerometer/Shimmer03_GAF_test/")
parser.add_argument('--acc_four_train_path', type=str, default=r"D:/Berkeley-MHAD/Accelerometer/Shimmer04_GAF_train/")
parser.add_argument('--acc_four_test_path', type=str, default=r"D:/Berkeley-MHAD/Accelerometer/Shimmer04_GAF_test/")
parser.add_argument('--acc_five_train_path', type=str, default=r"D:/Berkeley-MHAD/Accelerometer/Shimmer05_GAF_train/")
parser.add_argument('--acc_five_test_path', type=str, default=r"D:/Berkeley-MHAD/Accelerometer/Shimmer05_GAF_test/")
parser.add_argument('--acc_six_train_path', type=str, default=r"D:/Berkeley-MHAD/Accelerometer/Shimmer06_GAF_train/")
parser.add_argument('--acc_six_test_path', type=str, default=r"D:/Berkeley-MHAD/Accelerometer/Shimmer06_GAF_test/")
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
                                    SemanticFusionVGG16_Berkeley=backbone.SemanticFusionVGG16_Berkeley,
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
opts.dataset='Berkeley_oneGPU'
#opts.load=r"E:/Multi-modal Action Recognition/My codes/Relational Knowledge Distillation/output/UTD_a_g_SemanticFusionVGG16_margin0.2_epochs100_batch16_lr0.0001/tea_best_acc.pth"
opts.num_classes=11
opts.mode='train'
opts.modality='6a'
opts.base=backbone.SemanticFusionVGG16_Berkeley
opts.sample=pair.DistanceWeighted
opts.loss=loss.L2Triplet
opts.lr=0.0001
opts.margin=0.2
opts.batch=4
opts.epochs=100
opts.lr_decay_epochs=[50] 
opts.lr_decay_gamma=0.5
#opts.embedding_size=256
opts.print_freq=1
opts.output_dir='output/'
opts.save_dir= opts.output_dir+'_'.join(map(str, [opts.dataset, opts.modality, 'SemanticFusionVGG16_Berkeley', 
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

    a1_train_loader = loadtraindata(opts.acc_one_train_path)
    a1_test_loader = loadtestdata(opts.acc_one_test_path)
    a2_train_loader = loadtraindata(opts.acc_two_train_path)
    a2_test_loader = loadtestdata(opts.acc_two_test_path)
    a3_train_loader = loadtraindata(opts.acc_three_train_path)
    a3_test_loader = loadtestdata(opts.acc_three_test_path)
    a4_train_loader = loadtraindata(opts.acc_four_train_path)
    a4_test_loader = loadtestdata(opts.acc_four_test_path)
    a5_train_loader = loadtraindata(opts.acc_five_train_path)
    a5_test_loader = loadtestdata(opts.acc_five_test_path)
    a6_train_loader = loadtraindata(opts.acc_six_train_path)
    a6_test_loader = loadtestdata(opts.acc_six_test_path)

    Berkeley_Glove=np.load('data/Berkeley_Glove.npy')
    Berkeley_Glove=torch.from_numpy(Berkeley_Glove)
    Berkeley_Glove=Berkeley_Glove.float().cuda()
    #UTD_Glove=F.normalize(UTD_Glove, p=2, dim=1)
    torch.cuda.empty_cache()
    logging.info('----------- Network Initialization --------------')
    model = opts.base(n_classes=opts.num_classes).cuda()
    logging.info('Teacher: %s', model)
    logging.info('Teacher param size = %fMB', count_parameters_in_MB(model))
    logging.info('-----------------------------------------------')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    if opts.load is not None:
        model.load_state_dict(torch.load(opts.load))
        print("Loaded Model from %s" % opts.load)

    logging.info("Number of images in Acc one Training Set: %d" % len(a1_train_loader.dataset))
    logging.info("Number of images in Acc one Testing set: %d" % len(a1_test_loader.dataset))
    logging.info("Number of images in Acc two Training Set: %d" % len(a2_train_loader.dataset))
    logging.info("Number of images in Acc two Testing set: %d" % len(a2_test_loader.dataset))
    logging.info("Number of images in Acc three Training Set: %d" % len(a3_train_loader.dataset))
    logging.info("Number of images in Acc three Testing set: %d" % len(a3_test_loader.dataset))
    logging.info("Number of images in Acc four Training Set: %d" % len(a4_train_loader.dataset))
    logging.info("Number of images in Acc four Testing set: %d" % len(a4_test_loader.dataset))
    logging.info("Number of images in Acc five Training Set: %d" % len(a5_train_loader.dataset))
    logging.info("Number of images in Acc five Testing set: %d" % len(a5_test_loader.dataset))
    logging.info("Number of images in Acc six Training Set: %d" % len(a6_train_loader.dataset))
    logging.info("Number of images in Acc six Testing set: %d" % len(a6_test_loader.dataset))


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

    def train(net, a1_loader,a2_loader,a3_loader,a4_loader,a5_loader,a6_loader,ep):
        K = opts.recall
        batch_time = AverageMeter()
        data_time = AverageMeter()
        cls_loss = AverageMeter()
        semantic_loss = AverageMeter()
        a1_train_acc=0.
        a2_train_acc=0.
        a3_train_acc=0.
        a4_train_acc=0.
        a5_train_acc=0.
        a6_train_acc=0.
        net.train()
        loss_all = []
        end = time.time()
        i=1
        torch.cuda.empty_cache() 
        for (a1_images, a1_labels),(a2_images, a2_labels),(a3_images, a3_labels),(a4_images, a4_labels),\
            (a5_images, a5_labels),(a6_images, a6_labels) in zip(a1_loader,a2_loader,a3_loader,a4_loader,a5_loader,a6_loader):

            data_time.update(time.time() - end)

            a1_images, a1_labels = a1_images.cuda(), a1_labels.cuda()
            a2_images, a2_labels = a2_images.cuda(), a2_labels.cuda()
            a3_images, a3_labels = a3_images.cuda(), a3_labels.cuda()
            a4_images, a4_labels = a4_images.cuda(), a4_labels.cuda()
            a5_images, a5_labels = a5_images.cuda(), a5_labels.cuda()
            a6_images, a6_labels = a6_images.cuda(), a6_labels.cuda()

            a1_semantic=Berkeley_Glove[a1_labels]
            a2_semantic=Berkeley_Glove[a2_labels]
            a3_semantic=Berkeley_Glove[a3_labels]
            a4_semantic=Berkeley_Glove[a4_labels]
            a5_semantic=Berkeley_Glove[a5_labels]
            a6_semantic=Berkeley_Glove[a6_labels]

            a1_out1, a2_out1, a3_out1, a4_out1,a5_out1, a6_out1,\
            a1_out2, a2_out2, a3_out2, a4_out2, a5_out2, a6_out2,\
            a1_out3, a2_out3, a3_out3, a4_out3, a5_out3, a6_out3,\
            a1_out4, a2_out4, a3_out4, a4_out4, a5_out4, a6_out4,\
            a1_out5, a2_out5, a3_out5, a4_out5, a5_out5, a6_out5,\
            a1_out6, a2_out6, a3_out6, a4_out6, a5_out6, a6_out6,\
            a1_out7, a2_out7, a3_out7, a4_out7, a5_out7, a6_out7,\
            a1_out8, a2_out8, a3_out8, a4_out8, a5_out8, a6_out8\
            =net(a1_images, a2_images, a3_images,a4_images, a5_images, a6_images, True)
            
            a1_pred=torch.max(a1_out8,1)[1]
            a2_pred=torch.max(a2_out8,1)[1]
            a3_pred=torch.max(a3_out8,1)[1]
            a4_pred=torch.max(a4_out8,1)[1]
            a5_pred=torch.max(a5_out8,1)[1]
            a6_pred=torch.max(a6_out8,1)[1]

            a1_num_correct=(a1_pred==a1_labels).sum()
            a2_num_correct=(a2_pred==a2_labels).sum()
            a3_num_correct=(a3_pred==a3_labels).sum()
            a4_num_correct=(a4_pred==a4_labels).sum()
            a5_num_correct=(a5_pred==a5_labels).sum()
            a6_num_correct=(a6_pred==a6_labels).sum()

            a1_train_acc+=a1_num_correct.item()
            a2_train_acc+=a2_num_correct.item()
            a3_train_acc+=a3_num_correct.item()
            a4_train_acc+=a4_num_correct.item()
            a5_train_acc+=a5_num_correct.item()
            a6_train_acc+=a6_num_correct.item()

            #loss_triplet = criterion(embedding, labels)
            a1_loss_cls=criterion_cls(a1_out8, a1_labels)
            a2_loss_cls=criterion_cls(a2_out8, a2_labels)
            a3_loss_cls=criterion_cls(a3_out8, a3_labels)
            a4_loss_cls=criterion_cls(a4_out8, a4_labels)
            a5_loss_cls=criterion_cls(a5_out8, a5_labels)
            a6_loss_cls=criterion_cls(a6_out8, a6_labels)

            a1_semantic_loss=criterion_semantic(a1_out7, a1_semantic)
            a2_semantic_loss=criterion_semantic(a2_out7, a2_semantic)
            a3_semantic_loss=criterion_semantic(a3_out7, a3_semantic)
            a4_semantic_loss=criterion_semantic(a4_out7, a4_semantic)
            a5_semantic_loss=criterion_semantic(a5_out7, a5_semantic)
            a6_semantic_loss=criterion_semantic(a6_out7, a6_semantic)

            loss=(a1_loss_cls+a2_loss_cls+a3_loss_cls+a4_loss_cls+a5_loss_cls+a6_loss_cls+\
                a1_semantic_loss+a2_semantic_loss+a3_semantic_loss+a4_semantic_loss+a5_semantic_loss+a6_semantic_loss)/12.0 #+loss_triplet
            loss_all.append(loss.item())
            
            #rec = recall(embedding, labels, K=K)
            #prec = accuracy(embedding, labels, topk=(1,))
            #triplet_loss.update(loss_triplet.item(), images.size(0))
            cls_loss.update((a1_loss_cls.item()+a2_loss_cls.item()+a3_loss_cls.item()+a4_loss_cls.item()+a5_loss_cls.item()+a6_loss_cls.item())/6.0,\
                 a1_images.size(0)+a2_images.size(0)+a3_images.size(0)+a4_images.size(0)+a5_images.size(0)+a6_images.size(0))
            semantic_loss.update((a1_semantic_loss.item()+a2_semantic_loss.item()+a3_semantic_loss.item()+a4_semantic_loss.item()+a5_semantic_loss.item()+a6_semantic_loss.item())/6.0, \
                 a1_images.size(0)+a2_images.size(0)+a3_images.size(0)+a4_images.size(0)+a5_images.size(0)+a6_images.size(0))
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
                        .format(ep, i, len(a1_loader), batch_time=batch_time, data_time=data_time,
                        loss_cls=cls_loss, loss_semantic=semantic_loss))
                logging.info(log_str)
            i=i+1
            #train_iter.set_description("[Train][Epoch %d] Loss: %.5f" % (ep, loss.item()))
        logging.info('[Epoch %d] Loss: %.5f A1 Acc: %.5f A2 Acc: %.5f A3 Acc: %.5f A4 Acc: %.5f A5 Acc: %.5f A6 Acc: %.5f' % \
             (ep, torch.Tensor(loss_all).mean(), 100*a1_train_acc/(len(a1_loader.dataset)), 100*a2_train_acc/(len(a2_loader.dataset)),\
                  100*a3_train_acc/(len(a3_loader.dataset)),  100*a4_train_acc/(len(a4_loader.dataset)),100*a5_train_acc/(len(a5_loader.dataset)),  100*a6_train_acc/(len(a6_loader.dataset))))


    def eval(net, a1_loader,a2_loader,a3_loader,a4_loader, a5_loader,a6_loader,ep):
        #K = opts.recall
        net.eval()
        #test_iter = tqdm(loader, ncols=80)
        a1_embeddings_all, a2_embeddings_all,a3_embeddings_all,a4_embeddings_all,a5_embeddings_all,a6_embeddings_all,\
        a1_labels_all,a2_labels_all,a3_labels_all,a4_labels_all,a5_labels_all,a6_labels_all = [], [], [], [], [], [], [], [], [], [], [], []
        a1_correct = 0
        a2_correct = 0
        a3_correct = 0
        a4_correct = 0
        a5_correct = 0
        a6_correct = 0
        all_correct = 0
        #test_iter.set_description("[Eval][Epoch %d]" % ep)
        with torch.no_grad():
            for (a1_images, a1_labels),(a2_images, a2_labels),(a3_images, a3_labels),(a4_images, a4_labels),\
            (a5_images, a5_labels),(a6_images, a6_labels) in zip(a1_loader,a2_loader,a3_loader,a4_loader,a5_loader,a6_loader):

                a1_images, a1_labels = a1_images.cuda(), a1_labels.cuda()
                a2_images, a2_labels = a2_images.cuda(), a2_labels.cuda()
                a3_images, a3_labels = a3_images.cuda(), a3_labels.cuda()
                a4_images, a4_labels = a4_images.cuda(), a4_labels.cuda()
                a5_images, a5_labels = a5_images.cuda(), a5_labels.cuda()
                a6_images, a6_labels = a6_images.cuda(), a6_labels.cuda()

                a1_semantic=Berkeley_Glove[a1_labels]
                a2_semantic=Berkeley_Glove[a2_labels]
                a3_semantic=Berkeley_Glove[a3_labels]
                a4_semantic=Berkeley_Glove[a4_labels]
                a5_semantic=Berkeley_Glove[a5_labels]
                a6_semantic=Berkeley_Glove[a6_labels]

                a1_embedding,a2_embedding,a3_embedding,a4_embedding,a5_embedding,a6_embedding = net(a1_images, a2_images, a3_images,a4_images, a5_images, a6_images)

                a1_pred=torch.max(a1_embedding,1)[1]
                a2_pred=torch.max(a2_embedding,1)[1]
                a3_pred=torch.max(a3_embedding,1)[1]
                a4_pred=torch.max(a4_embedding,1)[1]
                a5_pred=torch.max(a5_embedding,1)[1]
                a6_pred=torch.max(a6_embedding,1)[1]
                all_pred=torch.max((a1_embedding+a2_embedding+a2_embedding+a3_embedding+a4_embedding+a5_embedding+a6_embedding),1)[1]

                a1_num_correct=(a1_pred==a1_labels).sum()
                a2_num_correct=(a2_pred==a2_labels).sum()
                a3_num_correct=(a3_pred==a3_labels).sum()
                a4_num_correct=(a4_pred==a4_labels).sum()
                a5_num_correct=(a5_pred==a5_labels).sum()
                a6_num_correct=(a6_pred==a6_labels).sum()
                all_num_correct=(all_pred==a1_labels).sum()

                a1_correct+=a1_num_correct.item()
                a2_correct+=a2_num_correct.item()
                a3_correct+=a3_num_correct.item()
                a4_correct+=a4_num_correct.item()
                a5_correct+=a5_num_correct.item()
                a6_correct+=a6_num_correct.item()
                all_correct+=all_num_correct.item()

                a1_embeddings_all.append(a1_embedding.data)
                a2_embeddings_all.append(a2_embedding.data)
                a3_embeddings_all.append(a3_embedding.data)
                a4_embeddings_all.append(a4_embedding.data)
                a5_embeddings_all.append(a5_embedding.data)
                a6_embeddings_all.append(a6_embedding.data)

                a1_labels_all.append(a1_labels.data)  
                a2_labels_all.append(a2_labels.data)
                a3_labels_all.append(a3_labels.data)  
                a4_labels_all.append(a4_labels.data)
                a5_labels_all.append(a5_labels.data)  
                a6_labels_all.append(a6_labels.data)

            a1_embeddings_all = torch.cat(a1_embeddings_all).cpu()
            a2_embeddings_all = torch.cat(a2_embeddings_all).cpu()
            a3_embeddings_all = torch.cat(a3_embeddings_all).cpu()
            a4_embeddings_all = torch.cat(a4_embeddings_all).cpu()
            a5_embeddings_all = torch.cat(a5_embeddings_all).cpu()
            a6_embeddings_all = torch.cat(a6_embeddings_all).cpu()

            a1_labels_all = torch.cat(a1_labels_all).cpu()
            a2_labels_all = torch.cat(a2_labels_all).cpu()
            a3_labels_all = torch.cat(a3_labels_all).cpu()
            a4_labels_all = torch.cat(a4_labels_all).cpu()
            a5_labels_all = torch.cat(a5_labels_all).cpu()
            a6_labels_all = torch.cat(a6_labels_all).cpu()

            #rec = recall(embeddings_all, labels_all, K=K)
            #s_prec = accuracy(s_embeddings_all, s_labels_all, topk=(1,))

            a1_acc = a1_correct/(len(a1_loader.dataset))
            a2_acc = a2_correct/(len(a2_loader.dataset))
            a3_acc = a3_correct/(len(a3_loader.dataset))
            a4_acc = a4_correct/(len(a4_loader.dataset))
            a5_acc = a5_correct/(len(a5_loader.dataset))
            a6_acc = a6_correct/(len(a6_loader.dataset))
            all_acc = all_correct/(len(a1_loader.dataset))

            logging.info('[Epoch %d] A1 acc: [%.4f] A2 acc: [%.4f] A3 acc: [%.4f] A4 acc: [%.4f] A5 acc: [%.4f] A6 acc: [%.4f] Combined acc: [%.4f]' \
                % (ep, a1_acc*100, a2_acc*100, a3_acc*100, a4_acc*100, a5_acc*100, a6_acc*100, all_acc*100))    
        return a1_acc, a2_acc, a3_acc, a4_acc, a5_acc, a6_acc, all_acc


    if opts.mode == "eval":
        eval(model, a1_test_loader,a2_test_loader, a3_test_loader,a4_test_loader, a5_test_loader,a6_test_loader,0)
    else:
        a1_val_acc, a2_val_acc,a3_val_acc, a4_val_acc, a5_val_acc, a6_val_acc,all_val_acc = eval(model, a1_test_loader,a2_test_loader, a3_test_loader,a4_test_loader, a5_test_loader,a6_test_loader,0)
        a1_best_acc =a1_val_acc
        a2_best_acc =a2_val_acc
        a3_best_acc =a3_val_acc
        a4_best_acc =a4_val_acc
        a5_best_acc =a5_val_acc
        a6_best_acc =a6_val_acc
        all_best_acc =all_val_acc
        for epoch in range(1, opts.epochs+1):
            train(model, a1_train_loader, a2_train_loader, a3_train_loader,a4_train_loader, a5_train_loader,a6_train_loader, epoch)
            a1_val_acc, a2_val_acc, a3_val_acc, a4_val_acc, a5_val_acc, a6_val_acc, all_val_acc = eval(model, a1_test_loader,a2_test_loader, a3_test_loader,a4_test_loader, a5_test_loader,a6_test_loader,epoch)
            if a1_best_acc < a1_val_acc:
                a1_best_acc = a1_val_acc
                if opts.save_dir is not None:
                    if not os.path.isdir(opts.save_dir):
                        os.mkdir(opts.save_dir)
                    torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "a1_best_acc.pth"))
            if a2_best_acc < a2_val_acc:
                a2_best_acc = a2_val_acc
                if opts.save_dir is not None:
                    if not os.path.isdir(opts.save_dir):
                        os.mkdir(opts.save_dir)
                    torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "a2_best_acc.pth"))
            if a3_best_acc < a3_val_acc:
                a3_best_acc = a3_val_acc
                if opts.save_dir is not None:
                    if not os.path.isdir(opts.save_dir):
                        os.mkdir(opts.save_dir)
                    torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "a3_best_acc.pth"))
            if a4_best_acc < a4_val_acc:
                a4_best_acc = a4_val_acc
                if opts.save_dir is not None:
                    if not os.path.isdir(opts.save_dir):
                        os.mkdir(opts.save_dir)
                    torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "a4_best_acc.pth"))
            if a5_best_acc < a5_val_acc:
                a5_best_acc = a5_val_acc
                if opts.save_dir is not None:
                    if not os.path.isdir(opts.save_dir):
                        os.mkdir(opts.save_dir)
                    torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "a5_best_acc.pth"))
            if a6_best_acc < a6_val_acc:
                a6_best_acc = a6_val_acc
                if opts.save_dir is not None:
                    if not os.path.isdir(opts.save_dir):
                        os.mkdir(opts.save_dir)
                    torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "a6_best_acc.pth"))
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
                    f.write("Best Acc one acc: %.4f\n" % (a1_best_acc*100))
                    f.write("Best Acc two acc: %.4f\n" % (a2_best_acc*100))
                    f.write("Best Acc three acc: %.4f\n" % (a3_best_acc*100))
                    f.write("Best Acc four acc: %.4f\n" % (a4_best_acc*100))
                    f.write("Best Acc five acc: %.4f\n" % (a5_best_acc*100))
                    f.write("Best Acc six acc: %.4f\n" % (a6_best_acc*100))
                    f.write("Best Combined acc: %.4f\n" % (all_best_acc*100))
                    #f.write("Final recall@1: %.4f\n" % (val_recall * 100))
                    #f.write("Final Prec@1: %.4f\n" % (val_prec))
                    f.write("Final Acc one acc: %.4f\n" % (a1_val_acc*100))
                    f.write("Final Acc two acc: %.4f\n" % (a2_val_acc*100))
                    f.write("Final Acc three acc: %.4f\n" % (a3_val_acc*100))
                    f.write("Final Acc four acc: %.4f\n" % (a4_val_acc*100))
                    f.write("Final Acc five acc: %.4f\n" % (a5_val_acc*100))
                    f.write("Final Acc six acc: %.4f\n" % (a6_val_acc*100))
                    f.write("Final Combined acc: %.4f\n" % (all_val_acc*100))
                    #f.write("F-measure: %.4f\n" % (F_measure*100))

            #logging.info("Best Recall@1: %.4f" % (best_rec*100))
            #logging.info("Best Prec@1: %.4f" % best_prec)
            logging.info("Best Acc one acc: %.4f" % (a1_best_acc*100))
            logging.info("Best Acc two acc: %.4f" % (a2_best_acc*100))
            logging.info("Best Acc three acc: %.4f" % (a3_best_acc*100))
            logging.info("Best Acc four acc: %.4f" % (a4_best_acc*100))
            logging.info("Best Acc five acc: %.4f" % (a5_best_acc*100))
            logging.info("Best Acc six acc: %.4f" % (a6_best_acc*100))
            logging.info("Best Combined acc: %.4f\n" % (all_best_acc*100))
            #logging.info("F-measure: %.4f" % (F_measure*100))




if __name__ == '__main__':
    main()