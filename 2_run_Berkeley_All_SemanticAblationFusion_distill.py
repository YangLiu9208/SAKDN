from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import argparse
import logging
import dataset
import model.backbone as backbone
import metric.pairsampler as pair
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as tnn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from metric.utils import recall, count_parameters_in_MB, accuracy, AverageMeter
from metric.batchsampler import NPairs
from metric.loss import HardDarkRank, RkdDistance, RKdAngle, L2Triplet, AttentionTransfer, SP, SE_Fusion, SoftTarget,Gram_loss
from model.embedding import LinearEmbedding
from TSNdataset_berkeley import TSNDataSet
from transforms import *
from itertools import cycle
import warnings
warnings.filterwarnings("ignore")

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
parser.add_argument('--stu_video_train_path', type=str, default=r"data/BerkeleyMHAD_train_list.txt")
parser.add_argument('--stu_video_test_path', type=str, default=r"data/BerkeleyMHAD_val_list.txt")
parser.add_argument('--modality', type=str, default='a', choices=['a', 'g'])
parser.add_argument('--student_base',
                        choices=dict(googlenet=backbone.GoogleNet,
                                    inception_v1bn=backbone.InceptionV1BN,
                                    resnet18=backbone.ResNet18,
                                    resnet50=backbone.ResNet50,
                                    vggnet16=backbone.VggNet16,
                                    Sevggnet16=backbone.SeVggNet16,
                                    SeFusionVGG16=backbone.SeFusionVGG16,
                                    SemanticFusionVGG16=backbone.SemanticFusionVGG16,
                                    TSN=backbone.TSN,
                                    TRN=backbone.TRN,
                                    ),
                        default=backbone.ResNet50,
                        action=LookupChoices)
parser.add_argument('--teacher_base',
                        choices=dict(googlenet=backbone.GoogleNet,
                                    inception_v1bn=backbone.InceptionV1BN,
                                    resnet18=backbone.ResNet18,
                                    resnet50=backbone.ResNet50,
                                    Semantic_ResNet18=backbone.Semantic_ResNet18,
                                    Semantic_ResNet50=backbone.Semantic_ResNet50,
                                    vggnet16=backbone.VggNet16,
                                    Sevggnet16=backbone.SeVggNet16,
                                    SeFusionVGG16=backbone.SeFusionVGG16,
                                    SemanticFusionVGG16_MMAct=backbone.SemanticFusionVGG16_MMAct,
                                    SemanticFusionVGG16_Berkeley=backbone.SemanticFusionVGG16_Berkeley,
                                    ),
                        default=backbone.VggNet16,
                        action=LookupChoices)
parser.add_argument('--consensus_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn','TRN', 'TRNmultiscale'])
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")
parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--num_classes', default=25, type=int)

parser.add_argument('--triplet_ratio', default=0, type=float)
parser.add_argument('--dist_ratio', default=0, type=float)
parser.add_argument('--angle_ratio', default=0, type=float)
parser.add_argument('--sp_ratio', default=0, type=float)
parser.add_argument('--st_ratio', default=0, type=float)
parser.add_argument('--GCAM_ratio', default=0, type=float)
parser.add_argument('--semantic_ratio', default=0, type=float)
parser.add_argument('--dark_ratio', default=0, type=float)
parser.add_argument('--dark_alpha', default=2, type=float)
parser.add_argument('--dark_beta', default=3, type=float)
parser.add_argument('--at_ratio', default=0, type=float)

parser.add_argument('--triplet_sample',
                    choices=dict(random=pair.RandomNegative,
                                 hard=pair.HardNegative,
                                 all=pair.AllPairs,
                                 semihard=pair.SemiHardNegative,
                                 distance=pair.DistanceWeighted),
                    default=pair.DistanceWeighted,
                    action=LookupChoices)

parser.add_argument('--triplet_margin', type=float, default=0.2)
#parser.add_argument('--l2normalize', choices=['true', 'false'], default='true')
#parser.add_argument('--embedding_size', default=128, type=int)

#parser.add_argument('--teacher_load', default=None, required=True)
#parser.add_argument('--teacher_l2normalize', choices=['true', 'false'], default='true')
#parser.add_argument('--teacher_embedding_size', default=128, type=int)
#parser.add_argument('--recall', default=[1], type=int, nargs='+')

parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--data', default='data')
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--batch', default=64, type=int)
parser.add_argument('--iter_per_epoch', default=100, type=int)
parser.add_argument('--lr_decay_epochs', type=int, default=[40, 60], nargs='+')
parser.add_argument('--lr_decay_gamma', type=float, default=0.1)
parser.add_argument('--recall', default=[1], type=int, nargs='+')
parser.add_argument('--save_dir', default=None)
parser.add_argument('--load', default=None)
parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')

opts = parser.parse_args()
opts.dataset='Berkeley_graph'
opts.modality='6a_v'
opts.num_classes=11
opts.student_base=backbone.TRN
opts.consensus_type='TRNmultiscale'
opts.arch='BNInception'
opts.num_segments=8
opts.teacher_base=backbone.SemanticFusionVGG16_Berkeley

opts.epochs=30
opts.lr=0.001   #cross_subject: 0.001, cross_scene:0.0001
opts.dropout=0.8
opts.lr_decay_epochs=[20]
opts.lr_decay_gamma=0.1  
opts.batch=8
opts.img_feature_dim=300

#opts.triplet_ratio=0
opts.dist_ratio=0     #Relational knowledge distillation distance
opts.angle_ratio=0    #Relational knowledge distillation angle
opts.sp_ratio=0     #Similarity preserving distillation
opts.st_ratio=0       #0.1
opts.GCAM_ratio=1
opts.semantic_ratio=1

opts.print_freq=1
opts.output_dir='output/'
opts.teacher_load='output/Berkeley_6a_SemanticFusionVGG16_Berkeley_margin0.2_epochs100_batch8_lr0.0001/a4_best_acc.pth'
#opts.load='output/Berkeley_6a_v_teacher_SemanticFusionVGG16_Berkeley_student_TRN_arch_BNInception_seg8_epochs20_batch8_lr0.001_dropout0.8/best_acc.pth'
opts.save_dir= opts.output_dir+'_'.join(map(str, ['st',str(opts.st_ratio),opts.dataset, opts.modality,'teacher','SemanticFusionVGG16_Berkeley','student','TRN', 
            'arch',str(opts.arch),'seg'+str(opts.num_segments),'epochs'+str(opts.epochs),'batch'+str(opts.batch), 'lr'+str(opts.lr),'dropout'+str(opts.dropout)]))

if not os.path.exists(opts.save_dir):
    os.makedirs(opts.save_dir)
log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(opts.save_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def set_seed(self, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _init_fn(self):
    np.random.seed(0)

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
                                             shuffle=False, num_workers=2, drop_last=True)
    return testloader

def main():
    logging.info('----------- Network Initialization --------------')
    student = opts.student_base(opts.num_classes, opts.num_segments, 'RGB', 
    base_model=opts.arch, consensus_type=opts.consensus_type,  dropout=opts.dropout, img_feature_dim=opts.img_feature_dim, partial_bn=True).cuda()
    teacher = opts.teacher_base(n_classes=opts.num_classes)
    logging.info('Teacher: %s', teacher)
    logging.info('Student: %s', student)
    logging.info('Teacher param size = %fMB', count_parameters_in_MB(teacher))
    logging.info('Student param size = %fMB', count_parameters_in_MB(student))
    logging.info('-----------------------------------------------')
    Berkeley_Glove=np.load('data/Berkeley_Glove.npy')
    Berkeley_Glove=torch.from_numpy(Berkeley_Glove)
    Berkeley_Glove=Berkeley_Glove.float().cuda()

    crop_size = student.crop_size
    scale_size = student.scale_size
    input_mean = student.input_mean
    input_std = student.input_std
    policies = student.get_optim_policies()
    train_augmentation = student.get_augmentation()
    normalize = GroupNormalize(input_mean, input_std)

    video_train_loader = torch.utils.data.DataLoader(
        TSNDataSet("", opts.stu_video_train_path, num_segments=opts.num_segments,
                   new_length=1,
                   modality='Berkeley_RGB',
                   image_tmpl="img_l{:02d}_c{:02d}_s{:02d}_a{:02d}_r{:02d}_{:05d}.pgm",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=opts.arch == 'BNInception'),
                       ToTorchFormatTensor(div=opts.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=opts.batch, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)

    video_test_loader = torch.utils.data.DataLoader(
        TSNDataSet("", opts.stu_video_test_path, num_segments=opts.num_segments,
                   new_length=1,
                   modality='Berkeley_RGB',
                   image_tmpl="img_l{:02d}_c{:02d}_s{:02d}_a{:02d}_r{:02d}_{:05d}.pgm",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=opts.arch == 'BNInception'),
                       ToTorchFormatTensor(div=opts.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=opts.batch, shuffle=False, num_workers=2, pin_memory=False, drop_last=True)  

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

    torch.cuda.empty_cache()

    if torch.cuda.device_count() > 1:
        student = torch.nn.DataParallel(student, device_ids=[0,1]).cuda()
        teacher = torch.nn.DataParallel(teacher, device_ids=[0,1]).cuda()

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
    logging.info("Number of videos in Student Training Set Two: %d" % len(video_train_loader.dataset))
    logging.info("Number of videos in Student Testiing Set Two: %d" % len(video_test_loader.dataset))

    if opts.load is not None:
        student.load_state_dict(torch.load(opts.load))
        logging.info("Loaded Model from %s" % opts.load)

    teacher.load_state_dict(torch.load(opts.teacher_load))
    student = student.cuda()
    teacher = teacher.cuda()

    #optimizer = optim.Adam(student.parameters(), lr=opts.lr, weight_decay=1e-5)
    optimizer = torch.optim.SGD(policies,
                                opts.lr,
                                momentum=0.9,
                                weight_decay=5e-4)
    #lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma)

    dist_criterion = RkdDistance().cuda()
    angle_criterion = RKdAngle().cuda()
    sp_criterion=SP().cuda()
    st_criterion = SoftTarget(4).cuda()
    GCAM_criterion=Gram_loss().cuda()
    cls_criterion=torch.nn.CrossEntropyLoss().cuda()
    criterion_semantic=torch.nn.MSELoss().cuda()
    #se_fusion_criterion=SE_Fusion(128,256,4).cuda()
    #dark_criterion = HardDarkRank(alpha=opts.dark_alpha, beta=opts.dark_beta)

    #triplet_criterion = L2Triplet(sampler=opts.triplet_sample(), margin=opts.triplet_margin)
    #at_criterion = AttentionTransfer()


    def train(a1_loader,a2_loader,a3_loader,a4_loader,a5_loader,a6_loader,s_loader,ep):
        K = opts.recall
        batch_time = AverageMeter()
        data_time = AverageMeter()
        #loss_dist = AverageMeter()
        #loss_angle= AverageMeter()
        #loss_sp= AverageMeter()
        loss_gcam = AverageMeter()
        loss_st= AverageMeter()
        loss_cls= AverageMeter()
        loss_semantic= AverageMeter()
        top1_recall = AverageMeter()
        top1_prec = AverageMeter()
        
        student.train()
        teacher.eval()

        #dist_loss_all = []
        #angle_loss_all = []
        semantic_loss_all = []
        cls_loss_all=[]
        sp_loss_all = []
        st_loss_all = []
        gcam_loss_all = []
        loss_all = []
        train_acc=0.
        end = time.time()
        torch.cuda.empty_cache() 
        a1_dataloader_iterator=iter(a1_loader)
        a2_dataloader_iterator=iter(a2_loader)
        a3_dataloader_iterator=iter(a3_loader)
        a4_dataloader_iterator=iter(a4_loader)
        a5_dataloader_iterator=iter(a5_loader)
        a6_dataloader_iterator=iter(a6_loader)

        for i, (s_videos,s_labels)  in enumerate(s_loader,start=1):

            try:
                (a1_images, a1_labels) = next(a1_dataloader_iterator)
                (a2_images, a2_labels) = next(a2_dataloader_iterator)
                (a3_images, a3_labels) = next(a3_dataloader_iterator)
                (a4_images, a4_labels) = next(a4_dataloader_iterator)
                (a5_images, a5_labels) = next(a5_dataloader_iterator)
                (a6_images, a6_labels) = next(a6_dataloader_iterator)
            except StopIteration:
                a1_dataloader_iterator=iter(a1_loader)
                a2_dataloader_iterator=iter(a2_loader)
                a3_dataloader_iterator=iter(a3_loader)
                a4_dataloader_iterator=iter(a4_loader)
                a5_dataloader_iterator=iter(a5_loader)
                a6_dataloader_iterator=iter(a6_loader)
                (a1_images, a1_labels) = next(a1_dataloader_iterator)
                (a2_images, a2_labels) = next(a2_dataloader_iterator)
                (a3_images, a3_labels) = next(a3_dataloader_iterator)
                (a4_images, a4_labels) = next(a4_dataloader_iterator)
                (a5_images, a5_labels) = next(a5_dataloader_iterator)
                (a6_images, a6_labels) = next(a6_dataloader_iterator)

        #for (ap_images, ap_labels),(aw_images, aw_labels), (gyro_images, gyro_labels),(ori_images, ori_labels),(s_videos,s_labels) in zip(cycle(ap_loader),cycle(aw_loader),cycle(gyro_loader),cycle(ori_loader),s_loader):

            data_time.update(time.time() - end)
            
            a1_images_ablation = torch.zeros(a1_images.size()).cuda()
            a2_images_ablation = torch.zeros(a2_images.size()).cuda()
            a3_images_ablation = torch.zeros(a3_images.size()).cuda()
            a4_images_ablation = torch.zeros(a4_images.size()).cuda()
            a5_images_ablation = torch.zeros(a5_images.size()).cuda()
            a6_images_ablation = torch.zeros(a6_images.size()).cuda()
            s_videos_ablation = torch.zeros(s_videos.size()).cuda()

            a1_images, a1_labels = a1_images.cuda(), a1_labels.cuda()
            a2_images, a2_labels = a2_images.cuda(), a2_labels.cuda()
            a3_images, a3_labels = a3_images.cuda(), a3_labels.cuda()
            a4_images, a4_labels = a4_images.cuda(), a4_labels.cuda()
            a5_images, a5_labels = a5_images.cuda(), a5_labels.cuda()
            a6_images, a6_labels = a6_images.cuda(), a6_labels.cuda()
            s_videos, s_labels = s_videos.cuda(), s_labels.cuda()

            a1_images_combined = torch.cat((a1_images,a1_images_ablation),0)
            a2_images_combined = torch.cat((a2_images,a2_images_ablation),0)
            a3_images_combined = torch.cat((a3_images,a3_images_ablation),0)
            a4_images_combined = torch.cat((a4_images,a4_images_ablation),0)
            a5_images_combined = torch.cat((a5_images,a5_images_ablation),0)
            a6_images_combined = torch.cat((a6_images,a6_images_ablation),0)
            s_videos_combined = torch.cat((s_videos,s_videos_ablation),0)

            a1_semantic=Berkeley_Glove[a1_labels]
            a2_semantic=Berkeley_Glove[a2_labels]
            a3_semantic=Berkeley_Glove[a3_labels]
            a4_semantic=Berkeley_Glove[a4_labels]
            a5_semantic=Berkeley_Glove[a5_labels]
            a6_semantic=Berkeley_Glove[a6_labels]

            with torch.no_grad():
                a1_out1, a2_out1, a3_out1, a4_out1,a5_out1, a6_out1,\
                a1_out2, a2_out2, a3_out2, a4_out2, a5_out2, a6_out2,\
                a1_out3, a2_out3, a3_out3, a4_out3, a5_out3, a6_out3,\
                a1_out4, a2_out4, a3_out4, a4_out4, a5_out4, a6_out4,\
                a1_out5, a2_out5, a3_out5, a4_out5, a5_out5, a6_out5,\
                a1_out6, a2_out6, a3_out6, a4_out6, a5_out6, a6_out6,\
                a1_out7, a2_out7, a3_out7, a4_out7, a5_out7, a6_out7,\
                a1_out8, a2_out8, a3_out8, a4_out8, a5_out8, a6_out8\
                =teacher(a1_images_combined, a2_images_combined, a3_images_combined,a4_images_combined, a5_images_combined, a6_images_combined, True)

            conv_out_conv2,conv_out_3a,conv_out_3b,conv_out_3c,conv_out_4a,conv_out_4b,conv_out_4c,conv_out_4d,conv_out_4e,conv_out_5a,conv_out_5b,v_semantic,video_output = student(s_videos_combined)

            t_side=len(a1_labels)
            s_side=len(s_labels)

            ## W
            eps = np.finfo(float).eps
            emb_a1_1=torch.unsqueeze(a1_out8[0:t_side,:],1)
            emb_a2_1=torch.unsqueeze(a2_out8[0:t_side,:],1)
            emb_a3_1=torch.unsqueeze(a3_out8[0:t_side,:],1)
            emb_a4_1=torch.unsqueeze(a4_out8[0:t_side,:],1)
            emb_a5_1=torch.unsqueeze(a5_out8[0:t_side,:],1)
            emb_a6_1=torch.unsqueeze(a6_out8[0:t_side,:],1)
            emb_video1=torch.unsqueeze(video_output[0:s_side,:],1)

            emb_a1_2=torch.unsqueeze(a1_out8[0:t_side,:],0)
            emb_a2_2=torch.unsqueeze(a2_out8[0:t_side,:],0)
            emb_a3_2=torch.unsqueeze(a3_out8[0:t_side,:],0)
            emb_a4_2=torch.unsqueeze(a4_out8[0:t_side,:],0)
            emb_a5_2=torch.unsqueeze(a5_out8[0:t_side,:],0)
            emb_a6_2=torch.unsqueeze(a6_out8[0:t_side,:],0)
            emb_video2=torch.unsqueeze(video_output[0:s_side,:],0)

            emb_a1_1_ablation=torch.unsqueeze(a1_out8[t_side:,:],1)
            emb_a2_1_ablation=torch.unsqueeze(a2_out8[t_side:,:],1)
            emb_a3_1_ablation=torch.unsqueeze(a3_out8[t_side:,:],1)
            emb_a4_1_ablation=torch.unsqueeze(a4_out8[t_side:,:],1)
            emb_a5_1_ablation=torch.unsqueeze(a5_out8[t_side:,:],1)
            emb_a6_1_ablation=torch.unsqueeze(a6_out8[t_side:,:],1)
            emb_video1_ablation=torch.unsqueeze(video_output[s_side:,:],1)

            emb_a1_2_ablation=torch.unsqueeze(a1_out8[t_side:,:],0)
            emb_a2_2_ablation=torch.unsqueeze(a2_out8[t_side:,:],0)
            emb_a3_2_ablation=torch.unsqueeze(a3_out8[t_side:,:],0)
            emb_a4_2_ablation=torch.unsqueeze(a4_out8[t_side:,:],0)
            emb_a5_2_ablation=torch.unsqueeze(a5_out8[t_side:,:],0)
            emb_a6_2_ablation=torch.unsqueeze(a6_out8[t_side:,:],0)
            emb_video2_ablation=torch.unsqueeze(video_output[s_side:,:],0)

            GW_a1 = ((emb_a1_1-emb_a1_2)**2).mean(2)   # N*N*d -> N*N
            GW_a2 = ((emb_a2_1-emb_a2_2)**2).mean(2)   # N*N*d -> N*N
            GW_a3 = ((emb_a3_1-emb_a3_2)**2).mean(2)   # N*N*d -> N*N
            GW_a4 = ((emb_a4_1-emb_a4_2)**2).mean(2)   # N*N*d -> N*N
            GW_a5 = ((emb_a5_1-emb_a5_2)**2).mean(2)   # N*N*d -> N*N
            GW_a6 = ((emb_a6_1-emb_a6_2)**2).mean(2)   # N*N*d -> N*N
            GW_video = ((emb_video1-emb_video2)**2).mean(2)   # N*N*d -> N*N
            GW_a1_ablation = ((emb_a1_1_ablation-emb_a1_2_ablation)**2).mean(2)   # N*N*d -> N*N
            GW_a2_ablation = ((emb_a2_1_ablation-emb_a2_2_ablation)**2).mean(2)   # N*N*d -> N*N
            GW_a3_ablation = ((emb_a3_1_ablation-emb_a3_2_ablation)**2).mean(2)   # N*N*d -> N*N
            GW_a4_ablation = ((emb_a4_1_ablation-emb_a4_2_ablation)**2).mean(2)   # N*N*d -> N*N
            GW_a5_ablation = ((emb_a5_1_ablation-emb_a5_2_ablation)**2).mean(2)   # N*N*d -> N*N
            GW_a6_ablation = ((emb_a6_1_ablation-emb_a6_2_ablation)**2).mean(2)   # N*N*d -> N*N
            GW_video_ablation = ((emb_video1_ablation-emb_video2_ablation)**2).mean(2)   # N*N*d -> N*N
            GW_a1 = torch.exp(-GW_a1/2)
            GW_a2 = torch.exp(-GW_a2/2)
            GW_a3 = torch.exp(-GW_a3/2)
            GW_a4 = torch.exp(-GW_a4/2)
            GW_a5 = torch.exp(-GW_a5/2)
            GW_a6 = torch.exp(-GW_a6/2)
            GW_video = torch.exp(-GW_video/2)
            GW_a1_ablation = torch.exp(-GW_a1_ablation/2)
            GW_a2_ablation = torch.exp(-GW_a2_ablation/2)
            GW_a3_ablation = torch.exp(-GW_a3_ablation/2)
            GW_a4_ablation = torch.exp(-GW_a4_ablation/2)
            GW_a5_ablation = torch.exp(-GW_a5_ablation/2)
            GW_a6_ablation = torch.exp(-GW_a6_ablation/2)
            GW_video_ablation = torch.exp(-GW_video_ablation/2)

            ## normalize
            D_a1 = GW_a1.sum(0)
            D_a2 = GW_a2.sum(0)
            D_a3 = GW_a3.sum(0)
            D_a4 = GW_a4.sum(0)
            D_a5 = GW_a5.sum(0)
            D_a6 = GW_a6.sum(0)
            D_video = GW_video.sum(0)

            D_a1_ablation = GW_a1_ablation.sum(0)
            D_a2_ablation = GW_a2_ablation.sum(0)
            D_a3_ablation = GW_a3_ablation.sum(0)
            D_a4_ablation = GW_a4_ablation.sum(0)
            D_a5_ablation = GW_a5_ablation.sum(0)
            D_a6_ablation = GW_a6_ablation.sum(0)
            D_video_ablation = GW_video_ablation.sum(0)

            D_a1_sqrt_inv = torch.sqrt(1.0/(D_a1+eps))
            D_a2_sqrt_inv = torch.sqrt(1.0/(D_a2+eps))
            D_a3_sqrt_inv = torch.sqrt(1.0/(D_a3+eps))
            D_a4_sqrt_inv = torch.sqrt(1.0/(D_a4+eps))
            D_a5_sqrt_inv = torch.sqrt(1.0/(D_a5+eps))
            D_a6_sqrt_inv = torch.sqrt(1.0/(D_a6+eps))
            D_video_sqrt_inv = torch.sqrt(1.0/(D_video+eps))

            D_a1_sqrt_inv_ablation = torch.sqrt(1.0/(D_a1_ablation+eps))
            D_a2_sqrt_inv_ablation = torch.sqrt(1.0/(D_a2_ablation+eps))
            D_a3_sqrt_inv_ablation = torch.sqrt(1.0/(D_a3_ablation+eps))
            D_a4_sqrt_inv_ablation = torch.sqrt(1.0/(D_a4_ablation+eps))
            D_a5_sqrt_inv_ablation = torch.sqrt(1.0/(D_a5_ablation+eps))
            D_a6_sqrt_inv_ablation = torch.sqrt(1.0/(D_a6_ablation+eps))
            D_video_sqrt_inv_ablation = torch.sqrt(1.0/(D_video_ablation+eps))

            D_a1_1 = torch.unsqueeze(D_a1_sqrt_inv,1).repeat(1,t_side)
            D_a1_2 = torch.unsqueeze(D_a1_sqrt_inv,0).repeat(t_side,1)
            D_a2_1 = torch.unsqueeze(D_a2_sqrt_inv,1).repeat(1,t_side)
            D_a2_2 = torch.unsqueeze(D_a2_sqrt_inv,0).repeat(t_side,1)
            D_a3_1 = torch.unsqueeze(D_a3_sqrt_inv,1).repeat(1,t_side)
            D_a3_2 = torch.unsqueeze(D_a3_sqrt_inv,0).repeat(t_side,1)
            D_a4_1 = torch.unsqueeze(D_a4_sqrt_inv,1).repeat(1,t_side)
            D_a4_2 = torch.unsqueeze(D_a4_sqrt_inv,0).repeat(t_side,1)
            D_a5_1 = torch.unsqueeze(D_a5_sqrt_inv,1).repeat(1,t_side)
            D_a5_2 = torch.unsqueeze(D_a5_sqrt_inv,0).repeat(t_side,1)
            D_a6_1 = torch.unsqueeze(D_a6_sqrt_inv,1).repeat(1,t_side)
            D_a6_2 = torch.unsqueeze(D_a6_sqrt_inv,0).repeat(t_side,1)
            D_video_1 = torch.unsqueeze(D_video_sqrt_inv,1).repeat(1,s_side)
            D_video_2 = torch.unsqueeze(D_video_sqrt_inv,0).repeat(s_side,1)

            D_a1_1_ablation = torch.unsqueeze(D_a1_sqrt_inv_ablation,1).repeat(1,t_side)
            D_a1_2_ablation = torch.unsqueeze(D_a1_sqrt_inv_ablation,0).repeat(t_side,1)
            D_a2_1_ablation = torch.unsqueeze(D_a2_sqrt_inv_ablation,1).repeat(1,t_side)
            D_a2_2_ablation = torch.unsqueeze(D_a2_sqrt_inv_ablation,0).repeat(t_side,1)
            D_a3_1_ablation = torch.unsqueeze(D_a3_sqrt_inv_ablation,1).repeat(1,t_side)
            D_a3_2_ablation = torch.unsqueeze(D_a3_sqrt_inv_ablation,0).repeat(t_side,1)
            D_a4_1_ablation = torch.unsqueeze(D_a4_sqrt_inv_ablation,1).repeat(1,t_side)
            D_a4_2_ablation = torch.unsqueeze(D_a4_sqrt_inv_ablation,0).repeat(t_side,1)
            D_a5_1_ablation = torch.unsqueeze(D_a5_sqrt_inv_ablation,1).repeat(1,t_side)
            D_a5_2_ablation = torch.unsqueeze(D_a5_sqrt_inv_ablation,0).repeat(t_side,1)
            D_a6_1_ablation = torch.unsqueeze(D_a6_sqrt_inv_ablation,1).repeat(1,t_side)
            D_a6_2_ablation = torch.unsqueeze(D_a6_sqrt_inv_ablation,0).repeat(t_side,1)
            D_video_1_ablation = torch.unsqueeze(D_video_sqrt_inv_ablation,1).repeat(1,s_side)
            D_video_2_ablation = torch.unsqueeze(D_video_sqrt_inv_ablation,0).repeat(s_side,1)
            
            S_a1  = D_a1_1*GW_a1*D_a1_2
            S_a2  = D_a2_1*GW_a2*D_a2_2
            S_a3  = D_a3_1*GW_a3*D_a3_2
            S_a4  = D_a4_1*GW_a4*D_a4_2
            S_a5  = D_a5_1*GW_a5*D_a5_2
            S_a6  = D_a6_1*GW_a6*D_a6_2
            S_s_video  = D_video_1*GW_video*D_video_2
            S_a1_ablation  = D_a1_1_ablation*GW_a1_ablation*D_a1_2_ablation
            S_a2_ablation  = D_a2_1_ablation*GW_a2_ablation*D_a2_2_ablation
            S_a3_ablation  = D_a3_1_ablation*GW_a3_ablation*D_a3_2_ablation
            S_a4_ablation  = D_a4_1_ablation*GW_a4_ablation*D_a4_2_ablation
            S_a5_ablation  = D_a5_1_ablation*GW_a5_ablation*D_a5_2_ablation
            S_a6_ablation  = D_a6_1_ablation*GW_a6_ablation*D_a6_2_ablation
            S_s_video_ablation  = D_video_1_ablation*GW_video_ablation*D_video_2_ablation

            #w_a1=(a1_out8[0:t_side,:]-a1_out8[t_side:,:]).div(a1_out8[0:t_side,:]).cpu().data.numpy()
            #w_a2=(a2_out8[0:t_side,:]-a2_out8[t_side:,:]).div(a2_out8[0:t_side,:]).cpu().data.numpy()
            #w_a3=(a3_out8[0:t_side,:]-a3_out8[t_side:,:]).div(a3_out8[0:t_side,:]).cpu().data.numpy()
            #w_a4=(a4_out8[0:t_side,:]-a4_out8[t_side:,:]).div(a4_out8[0:t_side,:]).cpu().data.numpy()
            #w_a5=(a5_out8[0:t_side,:]-a5_out8[t_side:,:]).div(a5_out8[0:t_side,:]).cpu().data.numpy()
            #w_a6=(a6_out8[0:t_side,:]-a6_out8[t_side:,:]).div(a6_out8[0:t_side,:]).cpu().data.numpy()
            #w_s=(video_output[0:s_side,:]-video_output[s_side:,:]).div(video_output[0:s_side,:]).cpu().data.numpy()

            soft_a1_out8 = F.softmax(a1_out8, dim=1)
            soft_a2_out8 = F.softmax(a2_out8, dim=1)
            soft_a3_out8 = F.softmax(a3_out8, dim=1)
            soft_a4_out8 = F.softmax(a4_out8, dim=1)
            soft_a5_out8 = F.softmax(a5_out8, dim=1)
            soft_a6_out8 = F.softmax(a6_out8, dim=1)
            soft_video_output = F.softmax(video_output, dim=1)

            soft_a1_predict=torch.max(soft_a1_out8,1)[1]
            soft_a2_predict=torch.max(soft_a2_out8,1)[1]
            soft_a3_predict=torch.max(soft_a3_out8,1)[1]
            soft_a4_predict=torch.max(soft_a4_out8,1)[1]
            soft_a5_predict=torch.max(soft_a5_out8,1)[1]
            soft_a6_predict=torch.max(soft_a6_out8,1)[1]
            soft_video_predict=torch.max(soft_video_output,1)[1]

            soft_a1_semantic = Berkeley_Glove[soft_a1_predict]
            soft_a2_semantic = Berkeley_Glove[soft_a2_predict]
            soft_a3_semantic = Berkeley_Glove[soft_a3_predict]
            soft_a4_semantic = Berkeley_Glove[soft_a4_predict]
            soft_a5_semantic = Berkeley_Glove[soft_a5_predict]
            soft_a6_semantic = Berkeley_Glove[soft_a6_predict]
            video_semantic=Berkeley_Glove[soft_video_predict]

            w_a1=(torch.mm(S_a1,soft_a1_semantic[0:t_side,:])-torch.mm(S_a1_ablation,soft_a1_semantic[t_side:,:])).\
                div(torch.mm(S_a1,soft_a1_semantic[0:t_side,:])).cpu().data.numpy()

            w_a2=(torch.mm(S_a2,soft_a2_semantic[0:t_side,:])-torch.mm(S_a2_ablation,soft_a2_semantic[t_side:,:])).\
                div(torch.mm(S_a2,soft_a2_semantic[0:t_side,:])).cpu().data.numpy()

            w_a3=(torch.mm(S_a3,soft_a3_semantic[0:t_side,:])-torch.mm(S_a3_ablation,soft_a3_semantic[t_side:,:])).\
                div(torch.mm(S_a3,soft_a3_semantic[0:t_side,:])).cpu().data.numpy()

            w_a4=(torch.mm(S_a4,soft_a4_semantic[0:t_side,:])-torch.mm(S_a4_ablation,soft_a4_semantic[t_side:,:])).\
                div(torch.mm(S_a4,soft_a4_semantic[0:t_side,:])).cpu().data.numpy()

            w_a5=(torch.mm(S_a5,soft_a5_semantic[0:t_side,:])-torch.mm(S_a5_ablation,soft_a5_semantic[t_side:,:])).\
                div(torch.mm(S_a5,soft_a5_semantic[0:t_side,:])).cpu().data.numpy()

            w_a6=(torch.mm(S_a6,soft_a6_semantic[0:t_side,:])-torch.mm(S_a6_ablation,soft_a6_semantic[t_side:,:])).\
                div(torch.mm(S_a6,soft_a6_semantic[0:t_side,:])).cpu().data.numpy()

            w_s=(torch.mm(S_s_video,video_semantic[0:s_side,:])-torch.mm(S_s_video_ablation,video_semantic[s_side:,:])).\
                 div(torch.mm(S_s_video,video_semantic[0:s_side,:])).cpu().data.numpy()

            # w_a1=(0.99*torch.mm(S_a1,soft_a1_semantic[0:t_side,:])+0.01*soft_a1_semantic[0:t_side,:]-0.99*torch.mm(S_a1_ablation,soft_a1_semantic[t_side:,:])-0.01*soft_a1_semantic[t_side:,:]).\
            #     div(0.99*torch.mm(S_a1,soft_a1_semantic[0:t_side,:])+0.01*soft_a1_semantic[0:t_side,:]).cpu().data.numpy()

            # w_a2=(0.99*torch.mm(S_a2,soft_a2_semantic[0:t_side,:])+0.01*soft_a2_semantic[0:t_side,:]-0.99*torch.mm(S_a2_ablation,soft_a2_semantic[t_side:,:])-0.01*soft_a2_semantic[t_side:,:]).\
            #     div(0.99*torch.mm(S_a2,soft_a2_semantic[0:t_side,:])+0.01*soft_a2_semantic[0:t_side,:]).cpu().data.numpy()

            # w_a3=(0.99*torch.mm(S_a3,soft_a3_semantic[0:t_side,:])+0.01*soft_a3_semantic[0:t_side,:]-0.99*torch.mm(S_a3_ablation,soft_a3_semantic[t_side:,:])-0.01*soft_a3_semantic[t_side:,:]).\
            #     div(0.99*torch.mm(S_a3,soft_a3_semantic[0:t_side,:])+0.01*soft_a3_semantic[0:t_side,:]).cpu().data.numpy()

            # w_a4=(0.99*torch.mm(S_a4,soft_a4_semantic[0:t_side,:])+0.01*soft_a4_semantic[0:t_side,:]-0.99*torch.mm(S_a4_ablation,soft_a4_semantic[t_side:,:])-0.01*soft_a4_semantic[t_side:,:]).\
            #     div(0.99*torch.mm(S_a4,soft_a4_semantic[0:t_side,:])+0.01*soft_a4_semantic[0:t_side,:]).cpu().data.numpy()

            # w_a5=(0.99*torch.mm(S_a5,soft_a5_semantic[0:t_side,:])+0.01*soft_a5_semantic[0:t_side,:]-0.99*torch.mm(S_a5_ablation,soft_a5_semantic[t_side:,:])-0.01*soft_a5_semantic[t_side:,:]).\
            #     div(0.99*torch.mm(S_a5,soft_a5_semantic[0:t_side,:])+0.01*soft_a5_semantic[0:t_side,:]).cpu().data.numpy()

            # w_a6=(0.99*torch.mm(S_a6,soft_a6_semantic[0:t_side,:])+0.01*soft_a6_semantic[0:t_side,:]-0.99*torch.mm(S_a6_ablation,soft_a6_semantic[t_side:,:])-0.01*soft_a6_semantic[t_side:,:]).\
            #     div(0.99*torch.mm(S_a6,soft_a6_semantic[0:t_side,:])+0.01*soft_a6_semantic[0:t_side,:]).cpu().data.numpy()

            # w_s=(torch.mm(0.99*S_s_video,video_semantic[0:s_side,:])+0.01*video_semantic[0:s_side,:]-0.99*torch.mm(S_s_video_ablation,video_semantic[s_side:,:])-0.01*video_semantic[s_side:,:]).\
            #     div(torch.mm(0.99*S_s_video,video_semantic[0:s_side,:])+0.01*video_semantic[0:s_side,:]).cpu().data.numpy()


            # w_a1=(soft_a1_semantic[0:t_side,:]-soft_a1_semantic[t_side:,:]).div(soft_a1_semantic[0:t_side,:]).cpu().data.numpy()
            # w_a2=(soft_a2_semantic[0:t_side,:]-soft_a2_semantic[t_side:,:]).div(soft_a2_semantic[0:t_side,:]).cpu().data.numpy()
            # w_a3=(soft_a3_semantic[0:t_side,:]-soft_a3_semantic[t_side:,:]).div(soft_a3_semantic[0:t_side,:]).cpu().data.numpy()
            # w_a4=(soft_a4_semantic[0:t_side,:]-soft_a4_semantic[t_side:,:]).div(soft_a4_semantic[0:t_side,:]).cpu().data.numpy()
            # w_a5=(soft_a5_semantic[0:t_side,:]-soft_a5_semantic[t_side:,:]).div(soft_a5_semantic[0:t_side,:]).cpu().data.numpy()
            # w_a6=(soft_a6_semantic[0:t_side,:]-soft_a6_semantic[t_side:,:]).div(soft_a6_semantic[0:t_side,:]).cpu().data.numpy()
            # w_s=(video_semantic[0:s_side,:]-video_semantic[s_side:,:]).div(video_semantic[0:s_side,:]).cpu().data.numpy()

            #w_a1=np.maximum(w_a1,0)
            #w_a2=np.maximum(w_a2,0)
            #w_a3=np.maximum(w_a3,0)
            #w_a4=np.maximum(w_a4,0)
            #w_a5=np.maximum(w_a5,0)
            #w_a6=np.maximum(w_a6,0)
            #w_s=np.maximum(w_s,0)

            w_a1=np.mean(w_a1, axis=(1))
            w_a2=np.mean(w_a2, axis=(1))
            w_a3=np.mean(w_a3, axis=(1))
            w_a4=np.mean(w_a4, axis=(1))
            w_a5=np.mean(w_a5, axis=(1))
            w_a6=np.mean(w_a6, axis=(1))
            w_s=np.mean(w_s, axis=(1))

            video_semantic=Berkeley_Glove[s_labels]
            video_output=video_output[0:s_side,:]
            v_semantic=v_semantic[0:s_side,:]
            conv_out_conv2=conv_out_conv2[0:s_side,:]
            conv_out_3c=conv_out_3c[0:s_side,:]
            conv_out_4c=conv_out_4c[0:s_side,:]
            conv_out_5a=conv_out_5a[0:s_side,:]
            conv_out_5b=conv_out_5b[0:s_side,:]

            a1_out1=a1_out1[0:t_side,:]
            a2_out1=a2_out1[0:t_side,:]
            a3_out1=a3_out1[0:t_side,:]
            a4_out1=a4_out1[0:t_side,:]
            a5_out1=a5_out1[0:t_side,:]
            a6_out1=a6_out1[0:t_side,:]

            a1_out2=a1_out2[0:t_side,:]
            a2_out2=a2_out2[0:t_side,:]
            a3_out2=a3_out2[0:t_side,:]
            a4_out2=a4_out2[0:t_side,:]
            a5_out2=a5_out2[0:t_side,:]
            a6_out2=a6_out2[0:t_side,:]

            a1_out3=a1_out3[0:t_side,:]
            a2_out3=a2_out3[0:t_side,:]
            a3_out3=a3_out3[0:t_side,:]
            a4_out3=a4_out3[0:t_side,:]
            a5_out3=a5_out3[0:t_side,:]
            a6_out3=a6_out3[0:t_side,:]

            a1_out4=a1_out4[0:t_side,:]
            a2_out4=a2_out4[0:t_side,:]
            a3_out4=a3_out4[0:t_side,:]
            a4_out4=a4_out4[0:t_side,:]
            a5_out4=a5_out4[0:t_side,:]
            a6_out4=a6_out4[0:t_side,:]

            a1_out5=a1_out5[0:t_side,:]
            a2_out5=a2_out5[0:t_side,:]
            a3_out5=a3_out5[0:t_side,:]
            a4_out5=a4_out5[0:t_side,:]
            a5_out5=a5_out5[0:t_side,:]
            a6_out5=a6_out5[0:t_side,:]

            a1_out7=a1_out7[0:t_side,:]
            a2_out7=a2_out7[0:t_side,:]
            a3_out7=a3_out7[0:t_side,:]
            a4_out7=a4_out7[0:t_side,:]
            a5_out7=a5_out7[0:t_side,:]
            a6_out7=a6_out7[0:t_side,:]

            a1_out8=a1_out8[0:t_side,:]
            a2_out8=a2_out8[0:t_side,:]
            a3_out8=a3_out8[0:t_side,:]
            a4_out8=a4_out8[0:t_side,:]
            a5_out8=a5_out8[0:t_side,:]
            a6_out8=a6_out8[0:t_side,:]

            ablation_a1_out1=a1_out1.cpu().data.numpy() * w_a1[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a2_out1=a2_out1.cpu().data.numpy() * w_a2[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a3_out1=a3_out1.cpu().data.numpy() * w_a3[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a4_out1=a4_out1.cpu().data.numpy() * w_a4[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a5_out1=a5_out1.cpu().data.numpy() * w_a5[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a6_out1=a6_out1.cpu().data.numpy() * w_a6[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]


            ablation_a1_out2=a1_out2.cpu().data.numpy() * w_a1[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a2_out2=a2_out2.cpu().data.numpy() * w_a2[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a3_out2=a3_out2.cpu().data.numpy() * w_a3[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a4_out2=a4_out2.cpu().data.numpy() * w_a4[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a5_out2=a5_out2.cpu().data.numpy() * w_a5[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a6_out2=a6_out2.cpu().data.numpy() * w_a6[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]

            ablation_a1_out3=a1_out3.cpu().data.numpy() * w_a1[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a2_out3=a2_out3.cpu().data.numpy() * w_a2[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a3_out3=a3_out3.cpu().data.numpy() * w_a3[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a4_out3=a4_out3.cpu().data.numpy() * w_a4[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a5_out3=a5_out3.cpu().data.numpy() * w_a5[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a6_out3=a6_out3.cpu().data.numpy() * w_a6[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]

            ablation_a1_out4=a1_out4.cpu().data.numpy() * w_a1[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a2_out4=a2_out4.cpu().data.numpy() * w_a2[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a3_out4=a3_out4.cpu().data.numpy() * w_a3[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a4_out4=a4_out4.cpu().data.numpy() * w_a4[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a5_out4=a5_out4.cpu().data.numpy() * w_a5[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a6_out4=a6_out4.cpu().data.numpy() * w_a6[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]

            ablation_a1_out5=a1_out5.cpu().data.numpy() * w_a1[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a2_out5=a2_out5.cpu().data.numpy() * w_a2[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a3_out5=a3_out5.cpu().data.numpy() * w_a3[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a4_out5=a4_out5.cpu().data.numpy() * w_a4[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a5_out5=a5_out5.cpu().data.numpy() * w_a5[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a6_out5=a6_out5.cpu().data.numpy() * w_a6[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]

            ablation_a1_out7=a1_out7.cpu().data.numpy() * w_a1[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a2_out7=a2_out7.cpu().data.numpy() * w_a2[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a3_out7=a3_out7.cpu().data.numpy() * w_a3[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a4_out7=a4_out7.cpu().data.numpy() * w_a4[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a5_out7=a5_out7.cpu().data.numpy() * w_a5[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_a6_out7=a6_out7.cpu().data.numpy() * w_a6[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]

            ablation_cam_v_conv2=conv_out_conv2.cpu().data.numpy() * w_s[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_cam_v_inception_3c_pool_proj=conv_out_3c.cpu().data.numpy() * w_s[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_cam_v_inception_4c_pool_proj=conv_out_4c.cpu().data.numpy() * w_s[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            #ablation_cam_v_inception_4e_pool_proj=conv_out_4e.cpu().data.numpy() * w_s[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_cam_v_inception_5a_pool_proj=conv_out_5a.cpu().data.numpy() * w_s[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_cam_v_inception_5b_pool_proj=conv_out_5b.cpu().data.numpy() * w_s[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_cam_v_semantic=v_semantic.cpu().double().data.numpy() * w_s[:, np.newaxis] 

            ablation_a1_out1 = np.sum(ablation_a1_out1, axis=1)  # [N,H,W]
            ablation_a1_out1 = np.maximum(ablation_a1_out1, 0)  # ReLU
            ablation_a2_out1 = np.sum(ablation_a2_out1, axis=1)  # [N,H,W]
            ablation_a2_out1 = np.maximum(ablation_a2_out1, 0)  # ReLU
            ablation_a3_out1 = np.sum(ablation_a3_out1, axis=1)  # [N,H,W]
            ablation_a3_out1 = np.maximum(ablation_a3_out1, 0)  # ReLU
            ablation_a4_out1 = np.sum(ablation_a4_out1, axis=1)  # [N,H,W]
            ablation_a4_out1 = np.maximum(ablation_a4_out1, 0)  # ReLU
            ablation_a5_out1 = np.sum(ablation_a5_out1, axis=1)  # [N,H,W]
            ablation_a5_out1 = np.maximum(ablation_a5_out1, 0)  # ReLU
            ablation_a6_out1 = np.sum(ablation_a6_out1, axis=1)  # [N,H,W]
            ablation_a6_out1 = np.maximum(ablation_a6_out1, 0)  # ReLU

            ablation_a1_out2 = np.sum(ablation_a1_out2, axis=1)  # [N,H,W]
            ablation_a1_out2 = np.maximum(ablation_a1_out2, 0)  # ReLU
            ablation_a2_out2 = np.sum(ablation_a2_out2, axis=1)  # [N,H,W]
            ablation_a2_out2 = np.maximum(ablation_a2_out2, 0)  # ReLU
            ablation_a3_out2 = np.sum(ablation_a3_out2, axis=1)  # [N,H,W]
            ablation_a3_out2 = np.maximum(ablation_a3_out2, 0)  # ReLU
            ablation_a4_out2 = np.sum(ablation_a4_out2, axis=1)  # [N,H,W]
            ablation_a4_out2 = np.maximum(ablation_a4_out2, 0)  # ReLU
            ablation_a5_out2 = np.sum(ablation_a5_out2, axis=1)  # [N,H,W]
            ablation_a5_out2 = np.maximum(ablation_a5_out2, 0)  # ReLU
            ablation_a6_out2 = np.sum(ablation_a6_out2, axis=1)  # [N,H,W]
            ablation_a6_out2 = np.maximum(ablation_a6_out2, 0)  # ReLU

            ablation_a1_out3 = np.sum(ablation_a1_out3, axis=1)  # [N,H,W]
            ablation_a1_out3 = np.maximum(ablation_a1_out3, 0)  # ReLU
            ablation_a2_out3 = np.sum(ablation_a2_out3, axis=1)  # [N,H,W]
            ablation_a2_out3 = np.maximum(ablation_a2_out3, 0)  # ReLU
            ablation_a3_out3 = np.sum(ablation_a3_out3, axis=1)  # [N,H,W]
            ablation_a3_out3 = np.maximum(ablation_a3_out3, 0)  # ReLU
            ablation_a4_out3 = np.sum(ablation_a4_out3, axis=1)  # [N,H,W]
            ablation_a4_out3 = np.maximum(ablation_a4_out3, 0)  # ReLU
            ablation_a5_out3 = np.sum(ablation_a5_out3, axis=1)  # [N,H,W]
            ablation_a5_out3 = np.maximum(ablation_a5_out3, 0)  # ReLU
            ablation_a6_out3 = np.sum(ablation_a6_out3, axis=1)  # [N,H,W]
            ablation_a6_out3 = np.maximum(ablation_a6_out3, 0)  # ReLU

            ablation_a1_out4 = np.sum(ablation_a1_out4, axis=1)  # [N,H,W]
            ablation_a1_out4 = np.maximum(ablation_a1_out4, 0)  # ReLU
            ablation_a2_out4 = np.sum(ablation_a2_out4, axis=1)  # [N,H,W]
            ablation_a2_out4 = np.maximum(ablation_a2_out4, 0)  # ReLU
            ablation_a3_out4 = np.sum(ablation_a3_out4, axis=1)  # [N,H,W]
            ablation_a3_out4 = np.maximum(ablation_a3_out4, 0)  # ReLU
            ablation_a4_out4 = np.sum(ablation_a4_out4, axis=1)  # [N,H,W]
            ablation_a4_out4 = np.maximum(ablation_a4_out4, 0)  # ReLU
            ablation_a5_out4 = np.sum(ablation_a5_out4, axis=1)  # [N,H,W]
            ablation_a5_out4 = np.maximum(ablation_a5_out4, 0)  # ReLU
            ablation_a6_out4 = np.sum(ablation_a6_out4, axis=1)  # [N,H,W]
            ablation_a6_out4 = np.maximum(ablation_a6_out4, 0)  # ReLU

            ablation_a1_out5 = np.sum(ablation_a1_out5, axis=1)  # [N,H,W]
            ablation_a1_out5 = np.maximum(ablation_a1_out5, 0)  # ReLU
            ablation_a2_out5 = np.sum(ablation_a2_out5, axis=1)  # [N,H,W]
            ablation_a2_out5 = np.maximum(ablation_a2_out5, 0)  # ReLU
            ablation_a3_out5 = np.sum(ablation_a3_out5, axis=1)  # [N,H,W]
            ablation_a3_out5 = np.maximum(ablation_a3_out5, 0)  # ReLU
            ablation_a4_out5 = np.sum(ablation_a4_out5, axis=1)  # [N,H,W]
            ablation_a4_out5 = np.maximum(ablation_a4_out5, 0)  # ReLU
            ablation_a5_out5 = np.sum(ablation_a5_out5, axis=1)  # [N,H,W]
            ablation_a5_out5 = np.maximum(ablation_a5_out5, 0)  # ReLU
            ablation_a6_out5 = np.sum(ablation_a6_out5, axis=1)  # [N,H,W]
            ablation_a6_out5 = np.maximum(ablation_a6_out5, 0)  # ReLU

            ablation_a1_out7 = np.maximum(ablation_a1_out7, 0)  # ReLU
            ablation_a2_out7 = np.maximum(ablation_a2_out7, 0)  # ReLU
            ablation_a3_out7 = np.maximum(ablation_a3_out7, 0)  # ReLU
            ablation_a4_out7 = np.maximum(ablation_a4_out7, 0)  # ReLU
            ablation_a5_out7 = np.maximum(ablation_a5_out7, 0)  # ReLU
            ablation_a6_out7 = np.maximum(ablation_a6_out7, 0)  # ReLU

            ablation_cam_v_conv2= np.sum(ablation_cam_v_conv2, axis=1)  # [N,H,W]
            ablation_cam_v_conv2 = np.maximum(ablation_cam_v_conv2, 0)  # ReLU
            ablation_cam_v_inception_3c_pool_proj= np.sum(ablation_cam_v_inception_3c_pool_proj, axis=1)  # [N,H,W]
            ablation_cam_v_inception_3c_pool_proj = np.maximum(ablation_cam_v_inception_3c_pool_proj, 0)  # ReLU
            ablation_cam_v_inception_4c_pool_proj= np.sum(ablation_cam_v_inception_4c_pool_proj, axis=1)  # [N,H,W]
            ablation_cam_v_inception_4c_pool_proj = np.maximum(ablation_cam_v_inception_4c_pool_proj, 0)  # ReLU
            #ablation_cam_v_inception_4e_pool_proj= np.sum(ablation_cam_v_inception_4e_pool_proj, axis=1)  # [N,H,W]
            #ablation_cam_v_inception_4e_pool_proj = np.maximum(ablation_cam_v_inception_4e_pool_proj, 0)  # ReLU
            ablation_cam_v_inception_5a_pool_proj= np.sum(ablation_cam_v_inception_5a_pool_proj, axis=1)  # [N,H,W]
            ablation_cam_v_inception_5a_pool_proj = np.maximum(ablation_cam_v_inception_5a_pool_proj, 0)  # ReLU
            ablation_cam_v_inception_5b_pool_proj= np.sum(ablation_cam_v_inception_5b_pool_proj, axis=1)  # [N,H,W]
            ablation_cam_v_inception_5b_pool_proj = np.maximum(ablation_cam_v_inception_5b_pool_proj, 0)  # ReLU
            #ablation_cam_v_semantic= np.sum(ablation_cam_v_semantic, axis=1)  # [N,H,W]
            ablation_cam_v_semantic = np.maximum(ablation_cam_v_semantic, 0)  # ReLU

            # 数值归一化

            ablation_a1_out1 -= np.min(ablation_a1_out1)
            if np.max(ablation_a1_out1)!=0:
                ablation_a1_out1 /= np.max(ablation_a1_out1)

            ablation_a2_out1 -= np.min(ablation_a2_out1)
            if np.max(ablation_a2_out1)!=0:
                ablation_a2_out1 /= np.max(ablation_a2_out1)    

            ablation_a3_out1 -= np.min(ablation_a3_out1)
            if np.max(ablation_a3_out1)!=0:
                ablation_a3_out1 /= np.max(ablation_a3_out1)

            ablation_a4_out1 -= np.min(ablation_a4_out1)
            if np.max(ablation_a4_out1)!=0:
                ablation_a4_out1 /= np.max(ablation_a4_out1)  

            ablation_a5_out1 -= np.min(ablation_a5_out1)
            if np.max(ablation_a5_out1)!=0:
                ablation_a5_out1 /= np.max(ablation_a5_out1)

            ablation_a6_out1 -= np.min(ablation_a6_out1)
            if np.max(ablation_a6_out1)!=0:
                ablation_a6_out1 /= np.max(ablation_a6_out1)  

            ablation_a1_out2 -= np.min(ablation_a1_out2)
            if np.max(ablation_a1_out2)!=0:
                ablation_a1_out2 /= np.max(ablation_a1_out2)

            ablation_a2_out2 -= np.min(ablation_a2_out2)
            if np.max(ablation_a2_out2)!=0:
                ablation_a2_out2 /= np.max(ablation_a2_out2)    

            ablation_a3_out2 -= np.min(ablation_a3_out2)
            if np.max(ablation_a3_out2)!=0:
                ablation_a3_out2 /= np.max(ablation_a3_out2)

            ablation_a4_out2 -= np.min(ablation_a4_out2)
            if np.max(ablation_a4_out2)!=0:
                ablation_a4_out2 /= np.max(ablation_a4_out2)  

            ablation_a5_out2 -= np.min(ablation_a5_out2)
            if np.max(ablation_a5_out2)!=0:
                ablation_a5_out2 /= np.max(ablation_a5_out2)

            ablation_a6_out2 -= np.min(ablation_a6_out2)
            if np.max(ablation_a6_out2)!=0:
                ablation_a6_out2 /= np.max(ablation_a6_out2)  


            ablation_a1_out3 -= np.min(ablation_a1_out3)
            if np.max(ablation_a1_out3)!=0:
                ablation_a1_out3 /= np.max(ablation_a1_out3)

            ablation_a2_out3 -= np.min(ablation_a2_out3)
            if np.max(ablation_a2_out3)!=0:
                ablation_a2_out3 /= np.max(ablation_a2_out3)    

            ablation_a3_out3 -= np.min(ablation_a3_out3)
            if np.max(ablation_a3_out3)!=0:
                ablation_a3_out3 /= np.max(ablation_a3_out3)

            ablation_a4_out3 -= np.min(ablation_a4_out3)
            if np.max(ablation_a4_out3)!=0:
                ablation_a4_out3 /= np.max(ablation_a4_out3)  

            ablation_a5_out3 -= np.min(ablation_a5_out3)
            if np.max(ablation_a5_out3)!=0:
                ablation_a5_out3 /= np.max(ablation_a5_out3)

            ablation_a6_out3 -= np.min(ablation_a6_out3)
            if np.max(ablation_a6_out3)!=0:
                ablation_a6_out3 /= np.max(ablation_a6_out3)  


            ablation_a1_out4 -= np.min(ablation_a1_out4)
            if np.max(ablation_a1_out4)!=0:
                ablation_a1_out4 /= np.max(ablation_a1_out4)

            ablation_a2_out4 -= np.min(ablation_a2_out4)
            if np.max(ablation_a2_out4)!=0:
                ablation_a2_out4 /= np.max(ablation_a2_out4)    

            ablation_a3_out4 -= np.min(ablation_a3_out4)
            if np.max(ablation_a3_out4)!=0:
                ablation_a3_out4 /= np.max(ablation_a3_out4)

            ablation_a4_out4 -= np.min(ablation_a4_out4)
            if np.max(ablation_a4_out4)!=0:
                ablation_a4_out4 /= np.max(ablation_a4_out4)  

            ablation_a5_out4 -= np.min(ablation_a5_out4)
            if np.max(ablation_a5_out4)!=0:
                ablation_a5_out4 /= np.max(ablation_a5_out4)

            ablation_a6_out4 -= np.min(ablation_a6_out4)
            if np.max(ablation_a6_out4)!=0:
                ablation_a6_out4 /= np.max(ablation_a6_out4)  


            ablation_a1_out5 -= np.min(ablation_a1_out5)
            if np.max(ablation_a1_out5)!=0:
                ablation_a1_out5 /= np.max(ablation_a1_out5)

            ablation_a2_out5 -= np.min(ablation_a2_out5)
            if np.max(ablation_a2_out5)!=0:
                ablation_a2_out5 /= np.max(ablation_a2_out5)    

            ablation_a3_out5 -= np.min(ablation_a3_out5)
            if np.max(ablation_a3_out5)!=0:
                ablation_a3_out5 /= np.max(ablation_a3_out5)

            ablation_a4_out5 -= np.min(ablation_a4_out5)
            if np.max(ablation_a4_out5)!=0:
                ablation_a4_out5 /= np.max(ablation_a4_out5)  

            ablation_a5_out5 -= np.min(ablation_a5_out5)
            if np.max(ablation_a5_out5)!=0:
                ablation_a5_out5 /= np.max(ablation_a5_out5)

            ablation_a6_out5 -= np.min(ablation_a6_out5)
            if np.max(ablation_a6_out5)!=0:
                ablation_a6_out5 /= np.max(ablation_a6_out5)  


            ablation_a1_out7 -= np.min(ablation_a1_out7)
            if np.max(ablation_a1_out7)!=0:
                ablation_a1_out1 /= np.max(ablation_a1_out7)

            ablation_a2_out7 -= np.min(ablation_a2_out7)
            if np.max(ablation_a2_out7)!=0:
                ablation_a2_out7 /= np.max(ablation_a2_out7)    

            ablation_a3_out7 -= np.min(ablation_a3_out7)
            if np.max(ablation_a3_out7)!=0:
                ablation_a3_out7 /= np.max(ablation_a3_out7)

            ablation_a4_out7 -= np.min(ablation_a4_out7)
            if np.max(ablation_a4_out7)!=0:
                ablation_a4_out7 /= np.max(ablation_a4_out7)  

            ablation_a5_out7 -= np.min(ablation_a5_out7)
            if np.max(ablation_a5_out7)!=0:
                ablation_a5_out7 /= np.max(ablation_a5_out7)

            ablation_a6_out7 -= np.min(ablation_a6_out7)
            if np.max(ablation_a6_out7)!=0:
                ablation_a6_out7 /= np.max(ablation_a6_out7)  

            ablation_cam_v_conv2 -= np.min(ablation_cam_v_conv2)
            if np.max(ablation_cam_v_conv2)!=0:
                ablation_cam_v_conv2 /= np.max(ablation_cam_v_conv2)  

            ablation_cam_v_inception_3c_pool_proj -= np.min(ablation_cam_v_inception_3c_pool_proj)
            if np.max(ablation_cam_v_inception_3c_pool_proj)!=0:
                ablation_cam_v_inception_3c_pool_proj /= np.max(ablation_cam_v_inception_3c_pool_proj)       

            ablation_cam_v_inception_4c_pool_proj -= np.min(ablation_cam_v_inception_4c_pool_proj)
            if np.max(ablation_cam_v_inception_4c_pool_proj)!=0:
                ablation_cam_v_inception_4c_pool_proj /= np.max(ablation_cam_v_inception_4c_pool_proj)         

            #ablation_cam_v_inception_4e_pool_proj -= np.min(ablation_cam_v_inception_4e_pool_proj)
            #if np.max(ablation_cam_v_inception_4e_pool_proj)!=0:
            #    ablation_cam_v_inception_4e_pool_proj /= np.max(ablation_cam_v_inception_4e_pool_proj)       

            ablation_cam_v_inception_5a_pool_proj -= np.min(ablation_cam_v_inception_5a_pool_proj)
            if np.max(ablation_cam_v_inception_5a_pool_proj)!=0:
                ablation_cam_v_inception_5a_pool_proj /= np.max(ablation_cam_v_inception_5a_pool_proj)       

            ablation_cam_v_inception_5b_pool_proj -= np.min(ablation_cam_v_inception_5b_pool_proj)
            if np.max(ablation_cam_v_inception_5b_pool_proj)!=0:
                ablation_cam_v_inception_5b_pool_proj /= np.max(ablation_cam_v_inception_5b_pool_proj)     

            ablation_cam_v_semantic -= np.min(ablation_cam_v_semantic)
            if np.max(ablation_cam_v_semantic)!=0:
                ablation_cam_v_semantic /= np.max(ablation_cam_v_semantic)              

            ablation_a1_out1 = np.resize(ablation_a1_out1, (len(a1_labels),56, 56))
            ablation_a2_out1 = np.resize(ablation_a2_out1, (len(a2_labels),56, 56))
            ablation_a3_out1 = np.resize(ablation_a3_out1, (len(a3_labels),56, 56))
            ablation_a4_out1 = np.resize(ablation_a4_out1, (len(a4_labels),56, 56))
            ablation_a5_out1 = np.resize(ablation_a5_out1, (len(a5_labels),56, 56))
            ablation_a6_out1 = np.resize(ablation_a6_out1, (len(a6_labels),56, 56))

            ablation_a1_out2 = np.resize(ablation_a1_out2, (len(a1_labels),14, 14))
            ablation_a2_out2 = np.resize(ablation_a2_out2, (len(a2_labels),14, 14))
            ablation_a3_out2 = np.resize(ablation_a3_out2, (len(a3_labels),14, 14))
            ablation_a4_out2 = np.resize(ablation_a4_out2, (len(a4_labels),14, 14))
            ablation_a5_out2 = np.resize(ablation_a5_out2, (len(a5_labels),14, 14))
            ablation_a6_out2 = np.resize(ablation_a6_out2, (len(a6_labels),14, 14))

            ablation_a1_out3 = np.resize(ablation_a1_out3, (len(a1_labels),14, 14))
            ablation_a2_out3 = np.resize(ablation_a2_out3, (len(a2_labels),14, 14))
            ablation_a3_out3 = np.resize(ablation_a3_out3, (len(a3_labels),14, 14))
            ablation_a4_out3 = np.resize(ablation_a4_out3, (len(a4_labels),14, 14))
            ablation_a5_out3 = np.resize(ablation_a5_out3, (len(a5_labels),14, 14))
            ablation_a6_out3 = np.resize(ablation_a6_out3, (len(a6_labels),14, 14))

            ablation_a1_out4 = np.resize(ablation_a1_out4, (len(a1_labels),7, 7))
            ablation_a2_out4 = np.resize(ablation_a2_out4, (len(a2_labels),7, 7))
            ablation_a3_out4 = np.resize(ablation_a3_out4, (len(a3_labels),7, 7))
            ablation_a4_out4 = np.resize(ablation_a4_out4, (len(a4_labels),7, 7))
            ablation_a5_out4 = np.resize(ablation_a5_out4, (len(a5_labels),7, 7))
            ablation_a6_out4 = np.resize(ablation_a6_out4, (len(a6_labels),7, 7))

            ablation_a1_out1 = torch.from_numpy(ablation_a1_out1).cuda()
            ablation_a2_out1 = torch.from_numpy(ablation_a2_out1).cuda()
            ablation_a3_out1 = torch.from_numpy(ablation_a3_out1).cuda()
            ablation_a4_out1 = torch.from_numpy(ablation_a4_out1).cuda()
            ablation_a5_out1 = torch.from_numpy(ablation_a5_out1).cuda()
            ablation_a6_out1 = torch.from_numpy(ablation_a6_out1).cuda()

            ablation_a1_out2 = torch.from_numpy(ablation_a1_out2).cuda()
            ablation_a2_out2 = torch.from_numpy(ablation_a2_out2).cuda()
            ablation_a3_out2 = torch.from_numpy(ablation_a3_out2).cuda()
            ablation_a4_out2 = torch.from_numpy(ablation_a4_out2).cuda()
            ablation_a5_out2 = torch.from_numpy(ablation_a5_out2).cuda()
            ablation_a6_out2 = torch.from_numpy(ablation_a6_out2).cuda()

            ablation_a1_out3 = torch.from_numpy(ablation_a1_out3).cuda()
            ablation_a2_out3 = torch.from_numpy(ablation_a2_out3).cuda()
            ablation_a3_out3 = torch.from_numpy(ablation_a3_out3).cuda()
            ablation_a4_out3 = torch.from_numpy(ablation_a4_out3).cuda()
            ablation_a5_out3 = torch.from_numpy(ablation_a5_out3).cuda()
            ablation_a6_out3 = torch.from_numpy(ablation_a6_out3).cuda()

            ablation_a1_out4 = torch.from_numpy(ablation_a1_out4).cuda()
            ablation_a2_out4 = torch.from_numpy(ablation_a2_out4).cuda()
            ablation_a3_out4 = torch.from_numpy(ablation_a3_out4).cuda()
            ablation_a4_out4 = torch.from_numpy(ablation_a4_out4).cuda()
            ablation_a5_out4 = torch.from_numpy(ablation_a5_out4).cuda()
            ablation_a6_out4 = torch.from_numpy(ablation_a6_out4).cuda()

            ablation_a1_out5 = torch.from_numpy(ablation_a1_out5).cuda()
            ablation_a2_out5 = torch.from_numpy(ablation_a2_out5).cuda()
            ablation_a3_out5 = torch.from_numpy(ablation_a3_out5).cuda()
            ablation_a4_out5 = torch.from_numpy(ablation_a4_out5).cuda()
            ablation_a5_out5 = torch.from_numpy(ablation_a5_out5).cuda()
            ablation_a6_out5 = torch.from_numpy(ablation_a6_out5).cuda()

            ablation_a1_out7 = torch.from_numpy(ablation_a1_out7).cuda()
            ablation_a2_out7 = torch.from_numpy(ablation_a2_out7).cuda()
            ablation_a3_out7 = torch.from_numpy(ablation_a3_out7).cuda()
            ablation_a4_out7 = torch.from_numpy(ablation_a4_out7).cuda()
            ablation_a5_out7 = torch.from_numpy(ablation_a5_out7).cuda()
            ablation_a6_out7 = torch.from_numpy(ablation_a6_out7).cuda()

            ablation_cam_v_conv2=torch.from_numpy(ablation_cam_v_conv2).cuda()
            ablation_cam_v_inception_3c_pool_proj=torch.from_numpy(ablation_cam_v_inception_3c_pool_proj).cuda()
            ablation_cam_v_inception_4c_pool_proj=torch.from_numpy(ablation_cam_v_inception_4c_pool_proj).cuda()
            #ablation_cam_v_inception_4e_pool_proj=torch.from_numpy(ablation_cam_v_inception_4e_pool_proj).cuda()
            ablation_cam_v_inception_5a_pool_proj=torch.from_numpy(ablation_cam_v_inception_5a_pool_proj).cuda()
            ablation_cam_v_inception_5b_pool_proj=torch.from_numpy(ablation_cam_v_inception_5b_pool_proj).cuda()
            ablation_cam_v_semantic=torch.from_numpy(ablation_cam_v_semantic).cuda()

            pred=torch.max(video_output,1)[1]
            train_correct=(pred==s_labels).sum()
            train_acc+=train_correct.item()
            #triplet_loss = opts.triplet_ratio * triplet_criterion(e, labels)

            #sp_loss= opts.sp_ratio * (sp_criterion(ablation_cam_v_inception_5b_pool_proj, ablation_ap_out5)+sp_criterion(ablation_cam_v_inception_5b_pool_proj, ablation_aw_out5)\
            #    +sp_criterion(ablation_cam_v_inception_5b_pool_proj, ablation_gy_out5)+sp_criterion(ablation_cam_v_inception_5b_pool_proj, ablation_ori_out5))/4.0

            st_loss=opts.st_ratio * (st_criterion(video_output,a1_out8)+st_criterion(video_output,a2_out8)\
                +st_criterion(video_output,a3_out8)+st_criterion(video_output,a4_out8)+st_criterion(video_output,a5_out8)+st_criterion(video_output,a6_out8))/6.0

            gcam_loss=opts.GCAM_ratio * (GCAM_criterion(ablation_cam_v_conv2,ablation_a1_out1)+GCAM_criterion(ablation_cam_v_conv2,ablation_a2_out1)\
                +GCAM_criterion(ablation_cam_v_conv2,ablation_a3_out1)+GCAM_criterion(ablation_cam_v_conv2,ablation_a4_out1)\
                +GCAM_criterion(ablation_cam_v_conv2,ablation_a5_out1)+GCAM_criterion(ablation_cam_v_conv2,ablation_a6_out1)\
                +GCAM_criterion(ablation_cam_v_inception_3c_pool_proj,ablation_a1_out2)+GCAM_criterion(ablation_cam_v_inception_3c_pool_proj,ablation_a2_out2)\
                +GCAM_criterion(ablation_cam_v_inception_3c_pool_proj,ablation_a3_out2)+GCAM_criterion(ablation_cam_v_inception_3c_pool_proj,ablation_a4_out2)\
                +GCAM_criterion(ablation_cam_v_inception_3c_pool_proj,ablation_a5_out2)+GCAM_criterion(ablation_cam_v_inception_3c_pool_proj,ablation_a6_out2)\
                +GCAM_criterion(ablation_cam_v_inception_4c_pool_proj,ablation_a1_out3)+GCAM_criterion(ablation_cam_v_inception_4c_pool_proj,ablation_a2_out3)\
                +GCAM_criterion(ablation_cam_v_inception_4c_pool_proj,ablation_a3_out3)+GCAM_criterion(ablation_cam_v_inception_4c_pool_proj,ablation_a4_out3)\
                +GCAM_criterion(ablation_cam_v_inception_4c_pool_proj,ablation_a5_out3)+GCAM_criterion(ablation_cam_v_inception_4c_pool_proj,ablation_a6_out3)\
                +GCAM_criterion(ablation_cam_v_inception_5a_pool_proj,ablation_a1_out4)+GCAM_criterion(ablation_cam_v_inception_5a_pool_proj,ablation_a2_out4)\
                +GCAM_criterion(ablation_cam_v_inception_5a_pool_proj,ablation_a3_out4)+GCAM_criterion(ablation_cam_v_inception_5a_pool_proj,ablation_a4_out4)\
                +GCAM_criterion(ablation_cam_v_inception_5a_pool_proj,ablation_a5_out4)+GCAM_criterion(ablation_cam_v_inception_5a_pool_proj,ablation_a6_out4)\
                +GCAM_criterion(ablation_cam_v_inception_5b_pool_proj,ablation_a1_out5)+GCAM_criterion(ablation_cam_v_inception_5b_pool_proj,ablation_a2_out5)\
                +GCAM_criterion(ablation_cam_v_inception_5b_pool_proj,ablation_a3_out5)+GCAM_criterion(ablation_cam_v_inception_5b_pool_proj,ablation_a4_out5)\
                +GCAM_criterion(ablation_cam_v_inception_5b_pool_proj,ablation_a5_out5)+GCAM_criterion(ablation_cam_v_inception_5b_pool_proj,ablation_a6_out5))/30.0
            
            cls_loss=cls_criterion(video_output,s_labels)

            semantic_loss=opts.semantic_ratio*(criterion_semantic(v_semantic, a1_out7)+criterion_semantic(v_semantic, a2_out7)+\
            criterion_semantic(v_semantic, a3_out7)+criterion_semantic(v_semantic, a4_out7)+\
            criterion_semantic(v_semantic, a5_out7)+criterion_semantic(v_semantic, a6_out7))/6.0

            loss = st_loss+gcam_loss+cls_loss+semantic_loss

            #loss_dist.update(dist_loss.item(), s_videos.size(0))
            #loss_angle.update(angle_loss.item(), s_videos.size(0))
            #loss_sp.update(sp_loss.item(), s_videos.size(0))
            loss_st.update(st_loss.item(), s_videos.size(0))
            loss_gcam.update(gcam_loss.item(), s_videos.size(0))
            loss_cls.update(cls_loss.item(), s_videos.size(0))
            loss_semantic.update(semantic_loss.item(), s_videos.size(0))

            rec = recall(video_output, s_labels, K=K)
            prec = accuracy(video_output, s_labels, topk=(1,))
            top1_recall.update(rec[0], s_videos.size(0))
            top1_prec.update(prec[0]/100, s_videos.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()
            #sp_loss_all.append(sp_loss.item())
            #dist_loss_all.append(dist_loss.item())
            #angle_loss_all.append(angle_loss.item())
            st_loss_all.append(st_loss.item())
            gcam_loss_all.append(gcam_loss.item())
            semantic_loss_all.append(semantic_loss.item())
            cls_loss_all.append(cls_loss.item())
            loss_all.append(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()
            if i % opts.print_freq == 0:
                log_str=('Epoch[{0}]:[{1:03}/{2:03}] '
                        #'Batch:{batch_time.val:.4f} '
                        #'Data:{data_time.val:.4f}  '
                        #'Dist:{dist_loss.val:.5f}({dist_loss.avg:.4f}) '
                        #'Ang:{angle_loss.val:.5f}({angle_loss.avg:.4f}) '
                        #'Sp:{sp_loss.val:.5f} ({sp_loss.avg:.4f}) '
                        'St:{st_loss.val:.5f} ({st_loss.avg:.4f}) '
                        'Gcam:{gcam_loss.val:.5f} ({gcam_loss.avg:.4f}) '
                        'Cls:{cls_loss.val:.5f}({cls_loss.avg:.4f}) '
                        'Se:{semantic_loss.val:.5f} ({semantic_loss.avg:.4f}) '
                        'recall@1:{top1_recall.val:.2f}({top1_recall.avg:.2f}) '
                        'pre@1:{top1_prec.val:.2f}({top1_prec.avg:.2f}) '.format(
                        ep, i, len(s_loader),st_loss=loss_st,gcam_loss=loss_gcam,cls_loss=loss_cls,semantic_loss=loss_semantic,top1_recall=top1_recall,top1_prec=top1_prec))
                logging.info(log_str)
            #train_iter.set_description("[Train][Epoch %d]  Dist: %.5f, Angle: %.5f, Dark: %5f" %
            #                        (ep, dist_loss.item(), angle_loss.item(), dark_loss.item()))
        logging.info('[Epoch %d] Loss: %.5f, St: %.5f, Gcam: %.5f, Se: %.5f, Acc: %.5f\n' %\
            (ep, torch.Tensor(loss_all).mean(), torch.Tensor(st_loss_all).mean(), torch.Tensor(gcam_loss_all).mean(),torch.Tensor(semantic_loss_all).mean(),100*train_acc/(len(s_loader.dataset))))


    def eval_teacher(net, a1_loader,a2_loader,a3_loader,a4_loader,a5_loader,a6_loader,ep):
        torch.cuda.empty_cache()
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
        return a1_embeddings_all, a2_embeddings_all,a3_embeddings_all, a4_embeddings_all,a5_embeddings_all, \
                a6_embeddings_all, a1_labels_all, a2_labels_all, a3_labels_all, a4_labels_all, a5_labels_all, a6_labels_all
    
    def eval_student(net, loader, ep):
        torch.cuda.empty_cache() 
        K=opts.recall
        net.eval()
        correct = 0
        embeddings_all= []
        labels_all= []
        with torch.no_grad():
            for i,(images, labels) in enumerate(loader, start=1):
                images, labels = images.cuda(), labels.cuda()
                conv_out_conv2,conv_out_3a,conv_out_3b,conv_out_3c,conv_out_4a,conv_out_4b,conv_out_4c,conv_out_4d,conv_out_4e,conv_out_5a,conv_out_5b,video_semantic,video_output = net(images)
                pred=torch.max(video_output,1)[1]
                num_correct=(pred==labels).sum()
                correct+=num_correct.item()
                embeddings_all.append(video_output.data)
                labels_all.append(labels.data)  
            embeddings_all = torch.cat(embeddings_all).cpu()
            labels_all = torch.cat(labels_all).cpu()
            rec = recall(embeddings_all, labels_all, K=K)
            #prec = accuracy(embeddings_all, labels_all, topk=(1,))
            acc = correct/(len(loader.dataset))
            logging.info('[Epoch %d] recall@1: [%.4f]' % (ep, 100 * rec[0]))
            logging.info('[Epoch %d] acc: [%.4f]' % (ep, acc*100))    
        return rec[0], acc, embeddings_all, labels_all

    logging.info('----------- Teacher Network performance --------------')

    a1_prec, a2_prec, a3_prec, a4_prec, a5_prec, a6_prec,a1_labels, a2_labels, a3_labels, a4_labels,a5_labels, a6_labels\
    =eval_teacher(teacher, a1_test_loader,a2_test_loader,a3_test_loader, a4_test_loader, a5_test_loader, a6_test_loader, 0)

    logging.info('----------- Student Network performance  --------------')

    best_val_recall, best_val_acc, s_prec, video_labels = eval_student(student, video_test_loader, 0)
    
    #combined_acc=accuracy((ap_prec+aw_prec+gy_prec+ori_prec+s_prec), video_labels, topk=(1,))
    #best_combined_acc=combined_acc[0]
    #combined_rec = recall((ap_prec+aw_prec+gy_prec+ori_prec+s_prec), video_labels, K=[1])
    #best_combined_rec=combined_rec[0]
    #logging.info('----------- Combined Network performance  --------------')
    #logging.info('Combined acc: [%.4f]\n' % (best_combined_acc)) 
    #logging.info('Combined rec: [%.4f]\n' % (best_combined_rec)) 

    for epoch in range(1, opts.epochs+1):
        adjust_learning_rate(optimizer, epoch, opts.lr_decay_epochs)
        train(a1_train_loader, a2_train_loader, a3_train_loader,a4_train_loader,a5_train_loader,a6_train_loader,video_train_loader, epoch)
        val_recall, val_acc, s_prec, video_labels = eval_student(student, video_test_loader, epoch)
        #val_combined_acc=accuracy((ap_prec+aw_prec+gy_prec+s_prec), video_labels, topk=(1,))
        #val_comb_acc=val_combined_acc[0]
        #val_combined_rec = recall((ap_prec+aw_prec+gy_prec+s_prec), video_labels, K=[1])
        #val_comb_rec=val_combined_rec[0]
        #logging.info('[Epoch %d] combined acc: [%.4f]' % (epoch, val_comb_acc)) 
        #logging.info('[Epoch %d] combined rec: [%.4f]\n' % (epoch, val_comb_rec*100)) 
        if best_val_recall < val_recall:
            best_val_recall = val_recall
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(student.state_dict(), "%s/%s"%(opts.save_dir, "best_recall.pth"))

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(student.state_dict(), "%s/%s"%(opts.save_dir, "best_acc.pth"))

        #if best_combined_acc < val_comb_acc:
        #    best_combined_acc = val_comb_acc
        #    if opts.save_dir is not None:
        #        if not os.path.isdir(opts.save_dir):
        #            os.mkdir(opts.save_dir)
        #        torch.save(student.state_dict(), "%s/%s"%(opts.save_dir, "best_combined.pth"))

        #if best_combined_rec < val_comb_rec:
        #    best_combined_rec = val_comb_rec
        #    if opts.save_dir is not None:
        #        if not os.path.isdir(opts.save_dir):
        #            os.mkdir(opts.save_dir)
        #        torch.save(student.state_dict(), "%s/%s"%(opts.save_dir, "best_combined_rec.pth"))  

        F_measure=(2*best_val_acc*best_val_recall)/(best_val_acc+best_val_recall)
        #combined_F_measure=(2*best_combined_acc/100*best_combined_rec)/(best_combined_acc/100+best_combined_rec)
        if opts.save_dir is not None:
            if not os.path.isdir(opts.save_dir):
                os.mkdir(opts.save_dir)
            torch.save(student.state_dict(), "%s/%s" % (opts.save_dir, "last.pth"))
            with open("%s/result.txt" % opts.save_dir, 'w') as f:
                f.write("Best Test recall@1: %.4f\n" % (best_val_recall * 100))
                f.write("Final Recall@1: %.4f\n" % (val_recall * 100))
                f.write("Best Test Acc: %.4f\n" % (best_val_acc * 100))
                f.write("Final Acc: %.4f\n" % (val_acc * 100))
                #f.write("Best Combined Acc: %.4f\n" % (best_combined_acc))
                #f.write("Final Combined Acc: %.4f\n" % (val_comb_acc ))
                f.write("F-measure: %.4f\n" % (F_measure*100))
                #f.write("Combined F-measure: %.4f\n" % (combined_F_measure*100))

        logging.info("Best Eval Recall@1: %.4f" % (best_val_recall*100))
        logging.info("Best Eval Acc: %.4f" % (best_val_acc*100))
        #logging.info("Best Eval Combined Acc: %.4f" % (best_combined_acc))
        logging.info("Eval F-measure: %.4f" % (F_measure*100))
        #logging.info("Eval Combined F-measure: %.4f\n" % (combined_F_measure*100))


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = opts.lr * decay
    decay = 5e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

if __name__ == '__main__':
    main()