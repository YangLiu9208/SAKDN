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
from TSNdataset import TSNDataSet
from transforms import *
from itertools import cycle
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--dataset', type=str, default='UTD', choices=['UTD', 'MMAct'])
parser.add_argument('--acc_phone_train_path', type=str, default=r"E:/Multi-modal Action Recognition/MMAct_sensors/acc_phone_cross_view_train/")
parser.add_argument('--acc_phone_test_path', type=str, default=r"E:/Multi-modal Action Recognition/MMAct_sensors/acc_phone_cross_view_test/")
parser.add_argument('--acc_watch_train_path', type=str, default=r"E:/Multi-modal Action Recognition/MMAct_sensors/acc_watch_cross_view_train/")
parser.add_argument('--acc_watch_test_path', type=str, default=r"E:/Multi-modal Action Recognition/MMAct_sensors/acc_watch_cross_view_test/")
parser.add_argument('--gyro_train_path', type=str, default=r"E:/Multi-modal Action Recognition/MMAct_sensors/gyro_cross_view_train/")
parser.add_argument('--gyro_test_path', type=str, default=r"E:/Multi-modal Action Recognition/MMAct_sensors/gyro_cross_view_test/")
parser.add_argument('--orientation_train_path', type=str, default=r"E:/Multi-modal Action Recognition/MMAct_sensors/orientation_cross_view_train/")
parser.add_argument('--orientation_test_path', type=str, default=r"E:/Multi-modal Action Recognition/MMAct_sensors/orientation_cross_view_test/")
parser.add_argument('--stu_video_train_path', type=str, default=r"data/MMAct_cross_view_train_list.txt")
parser.add_argument('--stu_video_test_path', type=str, default=r"data/MMAct_cross_view_val_list.txt")
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
opts.dataset='MMAct_Cross_View_AblationCAM'
opts.modality='a_g_o_v'
opts.num_classes=36
opts.student_base=backbone.TRN
opts.consensus_type='TRNmultiscale'
opts.arch='BNInception'
opts.num_segments=3
opts.teacher_base=backbone.SemanticFusionVGG16_MMAct

opts.epochs=60
opts.lr=0.001   #cross_subject: 0.001, cross_scene:0.0001
opts.dropout=0.8
opts.lr_decay_epochs=[30]
opts.lr_decay_gamma=0.5  
opts.batch=32
opts.img_feature_dim=300

#opts.triplet_ratio=0
opts.dist_ratio=0     #Relational knowledge distillation distance
opts.angle_ratio=0    #Relational knowledge distillation angle
opts.sp_ratio=0     #Similarity preserving distillation
opts.st_ratio=0      #0.1
opts.GCAM_ratio=1
opts.semantic_ratio=1

opts.print_freq=1
opts.output_dir='output/'
opts.teacher_load='output/MMAct_cross_view_a_g_o_SemanticFusionVGG16_MMAct_margin0.2_epochs70_batch16_lr0.0001/ap_best_acc.pth'
#opts.load='output/MMAct_Cross_View_AblationCAM_a_g_o_v_teacher_SemanticFusionVGG16_MMAct_student_TRN_arch_BNInception_seg8_epochs60_batch16_lr0.001_st0.1_se1_dropout0.8/best_acc.pth'
opts.save_dir= opts.output_dir+'_'.join(map(str, ['st',str(opts.st_ratio),opts.dataset, opts.modality,'teacher','SemanticFusionVGG16_MMAct','student','TRN', 
            'arch',str(opts.arch),'seg'+str(opts.num_segments),'epochs'+str(opts.epochs),'batch'+str(opts.batch), 'lr'+str(opts.lr),'st'+str(opts.st_ratio),'se'+str(opts.semantic_ratio),'dropout'+str(opts.dropout)]))

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
    torch.backends.cudnn.benchmark = True

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
                                              shuffle=True, num_workers=4, drop_last=True)
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
                                             shuffle=False, num_workers=4, drop_last=True)
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
    MMAct_Glove=np.load('data/MMAct_Glove.npy')
    MMAct_Glove=torch.from_numpy(MMAct_Glove)
    MMAct_Glove=MMAct_Glove.float().cuda()

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
                   modality='RGB',
                   image_tmpl="{:06d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=opts.arch == 'BNInception'),
                       ToTorchFormatTensor(div=opts.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=opts.batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    video_test_loader = torch.utils.data.DataLoader(
        TSNDataSet("", opts.stu_video_test_path, num_segments=opts.num_segments,
                   new_length=1,
                   modality='RGB',
                   image_tmpl="{:06d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=opts.arch == 'BNInception'),
                       ToTorchFormatTensor(div=opts.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=opts.batch, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)  

    acc_phone_train_loader = loadtraindata(opts.acc_phone_train_path)
    acc_phone_test_loader = loadtestdata(opts.acc_phone_test_path)
    acc_watch_train_loader = loadtraindata(opts.acc_watch_train_path)
    acc_watch_test_loader = loadtestdata(opts.acc_watch_test_path)
    gyro_train_loader = loadtraindata(opts.gyro_train_path)
    gyro_test_loader = loadtestdata(opts.gyro_test_path)
    ori_train_loader = loadtraindata(opts.orientation_train_path)
    ori_test_loader = loadtestdata(opts.orientation_test_path)

    torch.cuda.empty_cache()

    if torch.cuda.device_count() > 1:
        student = torch.nn.DataParallel(student, device_ids=[0,1]).cuda()
        teacher = torch.nn.DataParallel(teacher, device_ids=[0,1]).cuda()

    logging.info("Number of images in Acc Phone Training Set: %d" % len(acc_phone_train_loader.dataset))
    logging.info("Number of images in Acc Phone Testing set: %d" % len(acc_phone_test_loader.dataset))
    logging.info("Number of images in Acc Watch Training Set: %d" % len(acc_watch_train_loader.dataset))
    logging.info("Number of images in Acc Watch Testing set: %d" % len(acc_watch_test_loader.dataset))
    logging.info("Number of images in Gyro Training Set: %d" % len(gyro_train_loader.dataset))
    logging.info("Number of images in Gyro Testing set: %d" % len(gyro_test_loader.dataset))
    logging.info("Number of images in Ori Training Set: %d" % len(ori_train_loader.dataset))
    logging.info("Number of images in Ori Testing set: %d" % len(ori_test_loader.dataset))
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


    def train(ap_loader, aw_loader, gyro_loader, ori_loader,s_loader,ep):
        K = opts.recall
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_dist = AverageMeter()
        loss_angle= AverageMeter()
        loss_sp= AverageMeter()
        loss_gcam = AverageMeter()
        loss_st= AverageMeter()
        loss_cls= AverageMeter()
        loss_semantic= AverageMeter()
        top1_recall = AverageMeter()
        top1_prec = AverageMeter()
        
        student.train()
        teacher.eval()

        dist_loss_all = []
        angle_loss_all = []
        semantic_loss_all = []
        cls_loss_all=[]
        sp_loss_all = []
        st_loss_all = []
        gcam_loss_all = []
        loss_all = []
        train_acc=0.
        end = time.time()
        torch.cuda.empty_cache() 
        ap_dataloader_iterator=iter(ap_loader)
        aw_dataloader_iterator=iter(aw_loader)
        gyro_dataloader_iterator=iter(gyro_loader)
        ori_dataloader_iterator=iter(ori_loader)

        for i, (s_videos,s_labels)  in enumerate(s_loader,start=1):

            try:
                (ap_images, ap_labels) = next(ap_dataloader_iterator)
                (aw_images, aw_labels) = next(aw_dataloader_iterator)
                (gyro_images, gyro_labels) = next(gyro_dataloader_iterator)
                (ori_images, ori_labels) = next(ori_dataloader_iterator)
            except StopIteration:
                ap_dataloader_iterator=iter(ap_loader)
                aw_dataloader_iterator=iter(aw_loader)
                gyro_dataloader_iterator=iter(gyro_loader)
                ori_dataloader_iterator=iter(ori_loader)
                (ap_images, ap_labels) = next(ap_dataloader_iterator)
                (aw_images, aw_labels) = next(aw_dataloader_iterator)
                (gyro_images, gyro_labels) = next(gyro_dataloader_iterator)
                (ori_images, ori_labels) = next(ori_dataloader_iterator)

        #for (ap_images, ap_labels),(aw_images, aw_labels), (gyro_images, gyro_labels),(ori_images, ori_labels),(s_videos,s_labels) in zip(cycle(ap_loader),cycle(aw_loader),cycle(gyro_loader),cycle(ori_loader),s_loader):

            data_time.update(time.time() - end)
            
            ap_images_ablation = torch.zeros(ap_images.size()).cuda()
            aw_images_ablation = torch.zeros(aw_images.size()).cuda()
            gyro_images_ablation = torch.zeros(gyro_images.size()).cuda()
            ori_images_ablation = torch.zeros(ori_images.size()).cuda()
            s_videos_ablation = torch.zeros(s_videos.size()).cuda()

            ap_images, ap_labels = ap_images.cuda(), ap_labels.cuda()
            aw_images, aw_labels = aw_images.cuda(), aw_labels.cuda()
            gyro_images, gyro_labels = gyro_images.cuda(), gyro_labels.cuda()
            ori_images, ori_labels = ori_images.cuda(), ori_labels.cuda()
            s_videos, s_labels = s_videos.cuda(), s_labels.cuda()

            ap_images_combined = torch.cat((ap_images,ap_images_ablation),0)
            aw_images_combined = torch.cat((aw_images,aw_images_ablation),0)
            gyro_images_combined = torch.cat((gyro_images,gyro_images_ablation),0)
            ori_images_combined = torch.cat((ori_images,ori_images_ablation),0)
            s_videos_combined = torch.cat((s_videos,s_videos_ablation),0)

            ap_semantic=MMAct_Glove[ap_labels]
            aw_semantic=MMAct_Glove[aw_labels]
            gyro_semantic=MMAct_Glove[gyro_labels]
            ori_semantic=MMAct_Glove[ori_labels]

            with torch.no_grad():
                ap_out1, aw_out1, gy_out1, ori_out1,\
                ap_out2, aw_out2, gy_out2, ori_out2,\
                ap_out3, aw_out3, gy_out3, ori_out3,\
                ap_out4, aw_out4, gy_out4, ori_out4,\
                ap_out5, aw_out5, gy_out5, ori_out5,\
                ap_out6, aw_out6, gy_out6,ori_out6,\
                ap_out7, aw_out7, gy_out7, ori_out7,\
                ap_out8, aw_out8, gy_out8, ori_out8= \
                teacher(ap_images_combined,aw_images_combined,gyro_images_combined,ori_images_combined,True)

            conv_out_conv2,conv_out_3a,conv_out_3b,conv_out_3c,conv_out_4a,conv_out_4b,conv_out_4c,conv_out_4d,conv_out_4e,conv_out_5a,conv_out_5b,v_semantic,video_output = student(s_videos_combined)

            t_side=len(ap_labels)
            s_side=len(s_labels)

            ## W
            eps = np.finfo(float).eps
            emb_ap_1=torch.unsqueeze(ap_out8[0:t_side,:],1)
            emb_aw_1=torch.unsqueeze(aw_out8[0:t_side,:],1)
            emb_gy_1=torch.unsqueeze(gy_out8[0:t_side,:],1)
            emb_ori_1=torch.unsqueeze(ori_out8[0:t_side,:],1)
            emb_video1=torch.unsqueeze(video_output[0:s_side,:],1)

            emb_ap_2=torch.unsqueeze(ap_out8[0:t_side,:],0)
            emb_aw_2=torch.unsqueeze(aw_out8[0:t_side,:],0)
            emb_gy_2=torch.unsqueeze(gy_out8[0:t_side,:],0)
            emb_ori_2=torch.unsqueeze(ori_out8[0:t_side,:],0)
            emb_video2=torch.unsqueeze(video_output[0:s_side,:],0)

            emb_ap_1_ablation=torch.unsqueeze(ap_out8[t_side:,:],1)
            emb_aw_1_ablation=torch.unsqueeze(aw_out8[t_side:,:],1)
            emb_gy_1_ablation=torch.unsqueeze(gy_out8[t_side:,:],1)
            emb_ori_1_ablation=torch.unsqueeze(ori_out8[t_side:,:],1)
            emb_video1_ablation=torch.unsqueeze(video_output[s_side:,:],1)

            emb_ap_2_ablation=torch.unsqueeze(ap_out8[t_side:,:],0)
            emb_aw_2_ablation=torch.unsqueeze(aw_out8[t_side:,:],0)
            emb_gy_2_ablation=torch.unsqueeze(gy_out8[t_side:,:],0)
            emb_ori_2_ablation=torch.unsqueeze(ori_out8[t_side:,:],0)
            emb_video2_ablation=torch.unsqueeze(video_output[s_side:,:],0)

            GW_ap = ((emb_ap_1-emb_ap_2)**2).mean(2)   # N*N*d -> N*N
            GW_aw = ((emb_aw_1-emb_aw_2)**2).mean(2)   # N*N*d -> N*N
            GW_gy = ((emb_gy_1-emb_gy_2)**2).mean(2)   # N*N*d -> N*N
            GW_ori = ((emb_ori_1-emb_ori_2)**2).mean(2)   # N*N*d -> N*N
            GW_video = ((emb_video1-emb_video2)**2).mean(2)   # N*N*d -> N*N
            GW_ap_ablation = ((emb_ap_1_ablation-emb_ap_2_ablation)**2).mean(2)   # N*N*d -> N*N
            GW_aw_ablation = ((emb_aw_1_ablation-emb_aw_2_ablation)**2).mean(2)   # N*N*d -> N*N
            GW_gy_ablation = ((emb_gy_1_ablation-emb_gy_2_ablation)**2).mean(2)   # N*N*d -> N*N
            GW_ori_ablation = ((emb_ori_1_ablation-emb_ori_2_ablation)**2).mean(2)   # N*N*d -> N*N
            GW_video_ablation = ((emb_video1_ablation-emb_video2_ablation)**2).mean(2)   # N*N*d -> N*N
            GW_ap = torch.exp(-GW_ap/2)
            GW_aw = torch.exp(-GW_aw/2)
            GW_gy = torch.exp(-GW_gy/2)
            GW_ori = torch.exp(-GW_ori/2)
            GW_video = torch.exp(-GW_video/2)
            GW_ap_ablation = torch.exp(-GW_ap_ablation/2)
            GW_aw_ablation = torch.exp(-GW_aw_ablation/2)
            GW_gy_ablation = torch.exp(-GW_gy_ablation/2)
            GW_ori_ablation = torch.exp(-GW_ori_ablation/2)
            GW_video_ablation = torch.exp(-GW_video_ablation/2)

            ## normalize
            D_ap = GW_ap.sum(0)
            D_aw = GW_aw.sum(0)
            D_gy = GW_gy.sum(0)
            D_ori = GW_ori.sum(0)
            D_video = GW_video.sum(0)

            D_ap_ablation = GW_ap_ablation.sum(0)
            D_aw_ablation = GW_aw_ablation.sum(0)
            D_gy_ablation = GW_gy_ablation.sum(0)
            D_ori_ablation = GW_ori_ablation.sum(0)
            D_video_ablation = GW_video_ablation.sum(0)

            D_ap_sqrt_inv = torch.sqrt(1.0/(D_ap+eps))
            D_aw_sqrt_inv = torch.sqrt(1.0/(D_aw+eps))
            D_gy_sqrt_inv = torch.sqrt(1.0/(D_gy+eps))
            D_ori_sqrt_inv = torch.sqrt(1.0/(D_ori+eps))
            D_video_sqrt_inv = torch.sqrt(1.0/(D_video+eps))

            D_ap_sqrt_inv_ablation = torch.sqrt(1.0/(D_ap_ablation+eps))
            D_aw_sqrt_inv_ablation = torch.sqrt(1.0/(D_aw_ablation+eps))
            D_gy_sqrt_inv_ablation = torch.sqrt(1.0/(D_gy_ablation+eps))
            D_ori_sqrt_inv_ablation = torch.sqrt(1.0/(D_ori_ablation+eps))
            D_video_sqrt_inv_ablation = torch.sqrt(1.0/(D_video_ablation+eps))

            D_ap_1 = torch.unsqueeze(D_ap_sqrt_inv,1).repeat(1,t_side)
            D_ap_2 = torch.unsqueeze(D_ap_sqrt_inv,0).repeat(t_side,1)
            D_aw_1 = torch.unsqueeze(D_aw_sqrt_inv,1).repeat(1,t_side)
            D_aw_2 = torch.unsqueeze(D_aw_sqrt_inv,0).repeat(t_side,1)
            D_gy_1 = torch.unsqueeze(D_gy_sqrt_inv,1).repeat(1,t_side)
            D_gy_2 = torch.unsqueeze(D_gy_sqrt_inv,0).repeat(t_side,1)
            D_ori_1 = torch.unsqueeze(D_ori_sqrt_inv,1).repeat(1,t_side)
            D_ori_2 = torch.unsqueeze(D_ori_sqrt_inv,0).repeat(t_side,1)
            D_video_1 = torch.unsqueeze(D_video_sqrt_inv,1).repeat(1,s_side)
            D_video_2 = torch.unsqueeze(D_video_sqrt_inv,0).repeat(s_side,1)

            D_ap_1_ablation = torch.unsqueeze(D_ap_sqrt_inv_ablation,1).repeat(1,t_side)
            D_ap_2_ablation = torch.unsqueeze(D_ap_sqrt_inv_ablation,0).repeat(t_side,1)
            D_aw_1_ablation = torch.unsqueeze(D_aw_sqrt_inv_ablation,1).repeat(1,t_side)
            D_aw_2_ablation = torch.unsqueeze(D_aw_sqrt_inv_ablation,0).repeat(t_side,1)
            D_gy_1_ablation = torch.unsqueeze(D_gy_sqrt_inv_ablation,1).repeat(1,t_side)
            D_gy_2_ablation = torch.unsqueeze(D_gy_sqrt_inv_ablation,0).repeat(t_side,1)
            D_ori_1_ablation = torch.unsqueeze(D_ori_sqrt_inv_ablation,1).repeat(1,t_side)
            D_ori_2_ablation = torch.unsqueeze(D_ori_sqrt_inv_ablation,0).repeat(t_side,1)
            D_video_1_ablation = torch.unsqueeze(D_video_sqrt_inv_ablation,1).repeat(1,s_side)
            D_video_2_ablation = torch.unsqueeze(D_video_sqrt_inv_ablation,0).repeat(s_side,1)
            
            S_ap  = D_ap_1*GW_ap*D_ap_2
            S_aw  = D_aw_1*GW_aw*D_aw_2
            S_gy  = D_gy_1*GW_gy*D_gy_2
            S_ori  = D_ori_1*GW_ori*D_ori_2
            S_s_video  = D_video_1*GW_video*D_video_2
            S_ap_ablation  = D_ap_1_ablation*GW_ap_ablation*D_ap_2_ablation
            S_aw_ablation  = D_aw_1_ablation*GW_aw_ablation*D_aw_2_ablation
            S_gy_ablation  = D_gy_1_ablation*GW_gy_ablation*D_gy_2_ablation
            S_ori_ablation  = D_ori_1_ablation*GW_ori_ablation*D_ori_2_ablation
            S_s_video_ablation  = D_video_1_ablation*GW_video_ablation*D_video_2_ablation

            soft_ap_out8 = F.softmax(ap_out8, dim=1)
            soft_aw_out8 = F.softmax(aw_out8, dim=1)
            soft_gy_out8 = F.softmax(gy_out8, dim=1)
            soft_ori_out8 = F.softmax(ori_out8, dim=1)
            soft_video_output = F.softmax(video_output, dim=1)

            soft_ap_predict=torch.max(soft_ap_out8,1)[1]
            soft_aw_predict=torch.max(soft_aw_out8,1)[1]
            soft_gy_predict=torch.max(soft_gy_out8,1)[1]
            soft_ori_predict=torch.max(soft_ori_out8,1)[1]
            soft_video_predict=torch.max(soft_video_output,1)[1]

            soft_ap_semantic=MMAct_Glove[soft_ap_predict]
            soft_aw_semantic=MMAct_Glove[soft_aw_predict]
            soft_gy_semantic=MMAct_Glove[soft_gy_predict]
            soft_ori_semantic=MMAct_Glove[soft_ori_predict]
            video_semantic=MMAct_Glove[soft_video_predict]

            w_ap=((torch.mm(S_ap,soft_ap_semantic[0:t_side,:])-torch.mm(S_ap_ablation,soft_ap_semantic[t_side:,:]))/(torch.mm(S_ap,soft_ap_semantic[0:t_side,:]))).cpu().data.numpy()

            w_aw=((torch.mm(S_aw,soft_aw_semantic[0:t_side,:])-torch.mm(S_aw_ablation,soft_aw_semantic[t_side:,:]))/(torch.mm(S_aw,soft_aw_semantic[0:t_side,:]))).cpu().data.numpy()

            w_gy=((torch.mm(S_gy,soft_gy_semantic[0:t_side,:])-torch.mm(S_gy_ablation,soft_gy_semantic[t_side:,:]))/(torch.mm(S_gy,soft_gy_semantic[0:t_side,:]))).cpu().data.numpy()

            w_ori=((torch.mm(S_ori,soft_ori_semantic[0:t_side,:])-torch.mm(S_ori_ablation,soft_ori_semantic[t_side:,:]))/(torch.mm(S_ori,soft_ori_semantic[0:t_side,:]))).cpu().data.numpy()

            w_s=((torch.mm(S_s_video,video_semantic[0:s_side,:])-torch.mm(S_s_video_ablation,video_semantic[s_side:,:]))/(torch.mm(S_s_video,video_semantic[0:s_side,:]))).cpu().data.numpy()

            w_aw=np.maximum(w_aw,0)
            w_ap=np.maximum(w_ap,0)
            w_gy=np.maximum(w_gy,0)
            w_ori=np.maximum(w_ori,0)
            w_s=np.maximum(w_s,0)

            w_aw=np.mean(w_aw, axis=(1))
            w_ap=np.mean(w_ap, axis=(1))
            w_gy=np.mean(w_gy, axis=(1))
            w_ori=np.mean(w_ori, axis=(1))
            w_s=np.mean(w_s, axis=(1))

            video_semantic=MMAct_Glove[s_labels]
            video_output=video_output[0:s_side,:]
            v_semantic=v_semantic[0:s_side,:]
            conv_out_conv2=conv_out_conv2[0:s_side,:]
            conv_out_3c=conv_out_3c[0:s_side,:]
            conv_out_4c=conv_out_4c[0:s_side,:]
            conv_out_5a=conv_out_5a[0:s_side,:]
            conv_out_5b=conv_out_5b[0:s_side,:]

            ap_out1=ap_out1[0:t_side,:]
            aw_out1=aw_out1[0:t_side,:]
            gy_out1=gy_out1[0:t_side,:]
            ori_out1=ori_out1[0:t_side,:]

            ap_out2=ap_out2[0:t_side,:]
            aw_out2=aw_out2[0:t_side,:]
            gy_out2=gy_out2[0:t_side,:]
            ori_out2=ori_out2[0:t_side,:]

            ap_out3=ap_out3[0:t_side,:]
            aw_out3=aw_out3[0:t_side,:]
            gy_out3=gy_out3[0:t_side,:]
            ori_out3=ori_out3[0:t_side,:]

            ap_out4=ap_out4[0:t_side,:]
            aw_out4=aw_out4[0:t_side,:]
            gy_out4=gy_out4[0:t_side,:]
            ori_out4=ori_out4[0:t_side,:]

            ap_out5=ap_out5[0:t_side,:]
            aw_out5=aw_out5[0:t_side,:]
            gy_out5=gy_out5[0:t_side,:]
            ori_out5=ori_out5[0:t_side,:]

            ap_out7=ap_out7[0:t_side,:]
            aw_out7=aw_out7[0:t_side,:]
            gy_out7=gy_out7[0:t_side,:]
            ori_out7=ori_out7[0:t_side,:]

            ap_out8=ap_out8[0:t_side,:]
            aw_out8=aw_out8[0:t_side,:]
            gy_out8=gy_out8[0:t_side,:]
            ori_out8=ori_out8[0:t_side,:]

            ablation_ap_out1=ap_out1.cpu().data.numpy() * w_ap[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_aw_out1=aw_out1.cpu().data.numpy() * w_aw[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_gy_out1=gy_out1.cpu().data.numpy() * w_gy[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_ori_out1=ori_out1.cpu().data.numpy() * w_ori[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]

            ablation_ap_out2=ap_out2.cpu().data.numpy() * w_ap[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_aw_out2=aw_out2.cpu().data.numpy() * w_aw[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_gy_out2=gy_out2.cpu().data.numpy() * w_gy[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_ori_out2=ori_out2.cpu().data.numpy() * w_ori[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]

            ablation_ap_out3=ap_out3.cpu().data.numpy() * w_ap[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_aw_out3=aw_out3.cpu().data.numpy() * w_aw[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_gy_out3=gy_out3.cpu().data.numpy() * w_gy[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_ori_out3=ori_out3.cpu().data.numpy() * w_ori[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]

            ablation_ap_out4=ap_out4.cpu().data.numpy() * w_ap[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_aw_out4=aw_out4.cpu().data.numpy() * w_aw[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_gy_out4=gy_out4.cpu().data.numpy() * w_gy[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_ori_out4=ori_out4.cpu().data.numpy() * w_ori[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]

            ablation_ap_out5=ap_out5.cpu().data.numpy() * w_ap[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_aw_out5=aw_out5.cpu().data.numpy() * w_aw[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_gy_out5=gy_out5.cpu().data.numpy() * w_gy[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_ori_out5=ori_out5.cpu().data.numpy() * w_ori[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]

            ablation_ap_out7=ap_out7.cpu().data.numpy() * w_ap[:, np.newaxis]  # [N,C,H,W]
            ablation_aw_out7=aw_out7.cpu().data.numpy() * w_aw[:, np.newaxis]  # [N,C,H,W]
            ablation_gy_out7=gy_out7.cpu().data.numpy() * w_gy[:, np.newaxis]  # [N,C,H,W]
            ablation_ori_out7=ori_out7.cpu().data.numpy() * w_ori[:, np.newaxis]  # [N,C,H,W]

            ablation_cam_v_conv2=conv_out_conv2.cpu().data.numpy() * w_s[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_cam_v_inception_3c_pool_proj=conv_out_3c.cpu().data.numpy() * w_s[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_cam_v_inception_4c_pool_proj=conv_out_4c.cpu().data.numpy() * w_s[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            #ablation_cam_v_inception_4e_pool_proj=conv_out_4e.cpu().data.numpy() * w_s[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_cam_v_inception_5a_pool_proj=conv_out_5a.cpu().data.numpy() * w_s[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_cam_v_inception_5b_pool_proj=conv_out_5b.cpu().data.numpy() * w_s[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_cam_v_semantic=v_semantic.cpu().double().data.numpy() * w_s[:, np.newaxis] 

            ablation_ap_out1 = np.sum(ablation_ap_out1, axis=1)  # [N,H,W]
            ablation_ap_out1 = np.maximum(ablation_ap_out1, 0)  # ReLU
            ablation_aw_out1 = np.sum(ablation_aw_out1, axis=1)  # [N,H,W]
            ablation_aw_out1 = np.maximum(ablation_aw_out1, 0)  # ReLU
            ablation_gy_out1 = np.sum(ablation_gy_out1, axis=1)  # [N,H,W]
            ablation_gy_out1 = np.maximum(ablation_gy_out1, 0)  # ReLU
            ablation_ori_out1 = np.sum(ablation_ori_out1, axis=1)  # [N,H,W]
            ablation_ori_out1 = np.maximum(ablation_ori_out1, 0)  # ReLU

            ablation_ap_out2 = np.sum(ablation_ap_out2, axis=1)  # [N,H,W]
            ablation_ap_out2 = np.maximum(ablation_ap_out2, 0)  # ReLU
            ablation_aw_out2 = np.sum(ablation_aw_out2, axis=1)  # [N,H,W]
            ablation_aw_out2 = np.maximum(ablation_aw_out2, 0)  # ReLU
            ablation_gy_out2 = np.sum(ablation_gy_out2, axis=1)  # [N,H,W]
            ablation_gy_out2 = np.maximum(ablation_gy_out2, 0)  # ReLU
            ablation_ori_out2 = np.sum(ablation_ori_out2, axis=1)  # [N,H,W]
            ablation_ori_out2 = np.maximum(ablation_ori_out2, 0)  # ReLU

            ablation_ap_out3 = np.sum(ablation_ap_out3, axis=1)  # [N,H,W]
            ablation_ap_out3 = np.maximum(ablation_ap_out3, 0)  # ReLU
            ablation_aw_out3 = np.sum(ablation_aw_out3, axis=1)  # [N,H,W]
            ablation_aw_out3 = np.maximum(ablation_aw_out3, 0)  # ReLU
            ablation_gy_out3 = np.sum(ablation_gy_out3, axis=1)  # [N,H,W]
            ablation_gy_out3 = np.maximum(ablation_gy_out3, 0)  # ReLU
            ablation_ori_out3 = np.sum(ablation_ori_out3, axis=1)  # [N,H,W]
            ablation_ori_out3 = np.maximum(ablation_ori_out3, 0)  # ReLU

            ablation_ap_out4 = np.sum(ablation_ap_out4, axis=1)  # [N,H,W]
            ablation_ap_out4 = np.maximum(ablation_ap_out4, 0)  # ReLU
            ablation_aw_out4 = np.sum(ablation_aw_out4, axis=1)  # [N,H,W]
            ablation_aw_out4 = np.maximum(ablation_aw_out4, 0)  # ReLU
            ablation_gy_out4 = np.sum(ablation_gy_out4, axis=1)  # [N,H,W]
            ablation_gy_out4 = np.maximum(ablation_gy_out4, 0)  # ReLU
            ablation_ori_out4 = np.sum(ablation_ori_out4, axis=1)  # [N,H,W]
            ablation_ori_out4 = np.maximum(ablation_ori_out4, 0)  # ReLU

            ablation_ap_out5 = np.sum(ablation_ap_out5, axis=1)  # [N,H,W]
            ablation_ap_out5 = np.maximum(ablation_ap_out5, 0)  # ReLU
            ablation_aw_out5 = np.sum(ablation_aw_out5, axis=1)  # [N,H,W]
            ablation_aw_out5 = np.maximum(ablation_aw_out5, 0)  # ReLU
            ablation_gy_out5 = np.sum(ablation_gy_out5, axis=1)  # [N,H,W]
            ablation_gy_out5 = np.maximum(ablation_gy_out5, 0)  # ReLU
            ablation_ori_out5 = np.sum(ablation_ori_out5, axis=1)  # [N,H,W]
            ablation_ori_out5 = np.maximum(ablation_ori_out5, 0)  # ReLU

            #ablation_ap_out7 = np.sum(ablation_ap_out7, axis=1)  # [N,H,W]
            ablation_ap_out7 = np.maximum(ablation_ap_out7, 0)  # ReLU
            #ablation_aw_out7 = np.sum(ablation_aw_out7, axis=1)  # [N,H,W]
            ablation_aw_out7 = np.maximum(ablation_aw_out7, 0)  # ReLU
            #ablation_gy_out7 = np.sum(ablation_gy_out7, axis=1)  # [N,H,W]
            ablation_gy_out7 = np.maximum(ablation_gy_out7, 0)  # ReLU
            #ablation_ori_out7 = np.sum(ablation_ori_out7, axis=1)  # [N,H,W]
            ablation_ori_out7 = np.maximum(ablation_ori_out7, 0)  # ReLU

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

            ablation_ap_out1 -= np.min(ablation_ap_out1)
            ablation_ap_out1 /= (np.max(ablation_ap_out1)+eps)

            ablation_aw_out1 -= np.min(ablation_aw_out1)
            ablation_aw_out1 /= (np.max(ablation_aw_out1) +eps)   

            ablation_gy_out1 -= np.min(ablation_gy_out1)
            ablation_gy_out1 /= (np.max(ablation_gy_out1)+eps)

            ablation_ori_out1 -= np.min(ablation_ori_out1)
            ablation_ori_out1 /= (np.max(ablation_ori_out1)+eps)  

            ablation_ap_out2 -= np.min(ablation_ap_out2)
            ablation_ap_out2 /= (np.max(ablation_ap_out2)+eps)

            ablation_aw_out2 -= np.min(ablation_aw_out2)
            ablation_aw_out2 /= (np.max(ablation_aw_out2)+eps)    

            ablation_gy_out2 -= np.min(ablation_gy_out2)
            ablation_gy_out2 /= (np.max(ablation_gy_out2)+eps)

            ablation_ori_out2 -= np.min(ablation_ori_out2)
            ablation_ori_out2 /= (np.max(ablation_ori_out2)+eps)  

            ablation_ap_out3 -= np.min(ablation_ap_out3)
            ablation_ap_out3 /= (np.max(ablation_ap_out3)+eps)

            ablation_aw_out3 -= np.min(ablation_aw_out3)
            ablation_aw_out3 /= (np.max(ablation_aw_out3)+eps)    

            ablation_gy_out3 -= np.min(ablation_gy_out3)
            ablation_gy_out3 /= (np.max(ablation_gy_out3)+eps)

            ablation_ori_out3 -= np.min(ablation_ori_out3)
            ablation_ori_out3 /= (np.max(ablation_ori_out3)+eps)  

            ablation_ap_out4 -= np.min(ablation_ap_out4)
            ablation_ap_out4 /= (np.max(ablation_ap_out4)+eps)

            ablation_aw_out4 -= np.min(ablation_aw_out4)
            ablation_aw_out4 /= (np.max(ablation_aw_out4)+eps)    

            ablation_gy_out4 -= np.min(ablation_gy_out4)
            ablation_gy_out4 /= (np.max(ablation_gy_out4)+eps)

            ablation_ori_out4 -= np.min(ablation_ori_out4)
            ablation_ori_out4 /= (np.max(ablation_ori_out4)+eps)  

            ablation_ap_out5 -= np.min(ablation_ap_out5)
            ablation_ap_out5 /= (np.max(ablation_ap_out5)+eps)

            ablation_aw_out5 -= np.min(ablation_aw_out5)
            ablation_aw_out5 /= (np.max(ablation_aw_out5)+eps)    

            ablation_gy_out5 -= np.min(ablation_gy_out5)
            ablation_gy_out5 /= (np.max(ablation_gy_out5)+eps)

            ablation_ori_out5 -= np.min(ablation_ori_out5)
            ablation_ori_out5 /= (np.max(ablation_ori_out5)+eps)  

            ablation_ap_out7 -= np.min(ablation_ap_out7)
            ablation_ap_out7 /= (np.max(ablation_ap_out7)+eps)

            ablation_aw_out7 -= np.min(ablation_aw_out7)
            ablation_aw_out7 /= (np.max(ablation_aw_out7)+eps)    

            ablation_gy_out7 -= np.min(ablation_gy_out7)
            ablation_gy_out7 /= (np.max(ablation_gy_out7)+eps)

            ablation_ori_out7 -= np.min(ablation_ori_out7)
            ablation_ori_out7 /= (np.max(ablation_ori_out7)+eps)        

            ablation_cam_v_conv2 -= np.min(ablation_cam_v_conv2)
            ablation_cam_v_conv2 /= (np.max(ablation_cam_v_conv2)+eps)  

            ablation_cam_v_inception_3c_pool_proj -= np.min(ablation_cam_v_inception_3c_pool_proj)
            ablation_cam_v_inception_3c_pool_proj /= (np.max(ablation_cam_v_inception_3c_pool_proj)+eps)       

            ablation_cam_v_inception_4c_pool_proj -= np.min(ablation_cam_v_inception_4c_pool_proj)
            ablation_cam_v_inception_4c_pool_proj /= (np.max(ablation_cam_v_inception_4c_pool_proj)+eps)         

            ablation_cam_v_inception_5a_pool_proj -= np.min(ablation_cam_v_inception_5a_pool_proj)
            ablation_cam_v_inception_5a_pool_proj /= (np.max(ablation_cam_v_inception_5a_pool_proj)+eps)       

            ablation_cam_v_inception_5b_pool_proj -= np.min(ablation_cam_v_inception_5b_pool_proj)
            ablation_cam_v_inception_5b_pool_proj /= (np.max(ablation_cam_v_inception_5b_pool_proj)+eps)     

            ablation_ap_out1 = np.resize(ablation_ap_out1, (len(ap_labels),56, 56))
            ablation_aw_out1 = np.resize(ablation_aw_out1, (len(ap_labels),56, 56))
            ablation_gy_out1 = np.resize(ablation_gy_out1, (len(ap_labels),56, 56))
            ablation_ori_out1 = np.resize(ablation_ori_out1, (len(ap_labels),56, 56))

            ablation_ap_out2 = np.resize(ablation_ap_out2, (len(ap_labels),14, 14))
            ablation_aw_out2 = np.resize(ablation_aw_out2, (len(ap_labels),14, 14))
            ablation_gy_out2 = np.resize(ablation_gy_out2, (len(ap_labels),14, 14))
            ablation_ori_out2 = np.resize(ablation_ori_out2, (len(ap_labels),14, 14))

            ablation_ap_out3 = np.resize(ablation_ap_out3, (len(ap_labels),14, 14))
            ablation_aw_out3 = np.resize(ablation_aw_out3, (len(ap_labels),14, 14))
            ablation_gy_out3 = np.resize(ablation_gy_out3, (len(ap_labels),14, 14))
            ablation_ori_out3 = np.resize(ablation_ori_out3, (len(ap_labels),14, 14))

            ablation_ap_out4 = np.resize(ablation_ap_out4, (len(ap_labels),7, 7))
            ablation_aw_out4 = np.resize(ablation_aw_out4, (len(ap_labels),7, 7))
            ablation_gy_out4 = np.resize(ablation_gy_out4, (len(ap_labels),7, 7))
            ablation_ori_out4 = np.resize(ablation_ori_out4, (len(ap_labels),7, 7))

            ablation_ap_out1 = torch.from_numpy(ablation_ap_out1).cuda()
            ablation_aw_out1 = torch.from_numpy(ablation_aw_out1).cuda()
            ablation_gy_out1 = torch.from_numpy(ablation_gy_out1).cuda()
            ablation_ori_out1 = torch.from_numpy(ablation_ori_out1).cuda()
            ablation_ap_out2 = torch.from_numpy(ablation_ap_out2).cuda()
            ablation_aw_out2 = torch.from_numpy(ablation_aw_out2).cuda()
            ablation_gy_out2 = torch.from_numpy(ablation_gy_out2).cuda()
            ablation_ori_out2 = torch.from_numpy(ablation_ori_out2).cuda()
            ablation_ap_out3 = torch.from_numpy(ablation_ap_out3).cuda()
            ablation_aw_out3 = torch.from_numpy(ablation_aw_out3).cuda()
            ablation_gy_out3 = torch.from_numpy(ablation_gy_out3).cuda()
            ablation_ori_out3 = torch.from_numpy(ablation_ori_out3).cuda()
            ablation_ap_out4 = torch.from_numpy(ablation_ap_out4).cuda()
            ablation_aw_out4 = torch.from_numpy(ablation_aw_out4).cuda()
            ablation_gy_out4 = torch.from_numpy(ablation_gy_out4).cuda()
            ablation_ori_out4 = torch.from_numpy(ablation_ori_out4).cuda()
            ablation_ap_out5 = torch.from_numpy(ablation_ap_out5).cuda()
            ablation_aw_out5 = torch.from_numpy(ablation_aw_out5).cuda()
            ablation_gy_out5 = torch.from_numpy(ablation_gy_out5).cuda()
            ablation_ori_out5 = torch.from_numpy(ablation_ori_out5).cuda()
            ablation_ap_out7 = torch.from_numpy(ablation_ap_out7).cuda()
            ablation_aw_out7 = torch.from_numpy(ablation_aw_out7).cuda()
            ablation_gy_out7 = torch.from_numpy(ablation_gy_out7).cuda()
            ablation_ori_out7 = torch.from_numpy(ablation_ori_out7).cuda()

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

            st_loss=opts.st_ratio * (st_criterion(video_output,ap_out8)+st_criterion(video_output,aw_out8)\
                +st_criterion(video_output,gy_out8)+st_criterion(video_output,ori_out8))/4.0

            gcam_loss=opts.GCAM_ratio * (GCAM_criterion(ablation_cam_v_conv2,ablation_ap_out1)+GCAM_criterion(ablation_cam_v_conv2,ablation_aw_out1)\
                +GCAM_criterion(ablation_cam_v_conv2,ablation_gy_out1)+GCAM_criterion(ablation_cam_v_conv2,ablation_ori_out1)\
                +GCAM_criterion(ablation_cam_v_inception_3c_pool_proj,ablation_ap_out2)+GCAM_criterion(ablation_cam_v_inception_3c_pool_proj,ablation_aw_out2)\
                +GCAM_criterion(ablation_cam_v_inception_3c_pool_proj,ablation_gy_out2)+GCAM_criterion(ablation_cam_v_inception_3c_pool_proj,ablation_ori_out2)\
                +GCAM_criterion(ablation_cam_v_inception_4c_pool_proj,ablation_ap_out3)+GCAM_criterion(ablation_cam_v_inception_4c_pool_proj,ablation_aw_out3)\
                +GCAM_criterion(ablation_cam_v_inception_4c_pool_proj,ablation_gy_out3)+GCAM_criterion(ablation_cam_v_inception_4c_pool_proj,ablation_ori_out3)\
                +GCAM_criterion(ablation_cam_v_inception_5a_pool_proj,ablation_ap_out4)+GCAM_criterion(ablation_cam_v_inception_5a_pool_proj,ablation_aw_out4)\
                +GCAM_criterion(ablation_cam_v_inception_5a_pool_proj,ablation_gy_out4)+GCAM_criterion(ablation_cam_v_inception_5a_pool_proj,ablation_ori_out4)\
                +GCAM_criterion(ablation_cam_v_inception_5b_pool_proj,ablation_ap_out5)+GCAM_criterion(ablation_cam_v_inception_5b_pool_proj,ablation_aw_out5)\
                +GCAM_criterion(ablation_cam_v_inception_5b_pool_proj,ablation_gy_out5)+GCAM_criterion(ablation_cam_v_inception_5b_pool_proj,ablation_ori_out5))/20.0
            
            cls_loss=cls_criterion(video_output,s_labels)

            semantic_loss=opts.semantic_ratio*(criterion_semantic(v_semantic, ap_out7)+criterion_semantic(v_semantic, aw_out7)+\
                criterion_semantic(v_semantic, gy_out7)+criterion_semantic(v_semantic, ori_out7))/4.0

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
                        'Batch:{batch_time.val:.4f} '
                        'Data:{data_time.val:.4f}  '
                        #'Dist:{dist_loss.val:.5f}({dist_loss.avg:.4f}) '
                        #'Ang:{angle_loss.val:.5f}({angle_loss.avg:.4f}) '
                        #'Sp:{sp_loss.val:.5f} ({sp_loss.avg:.4f}) '
                        'St:{st_loss.val:.5f} ({st_loss.avg:.4f}) '
                        'Gcam:{gcam_loss.val:.5f} ({gcam_loss.avg:.4f}) '
                        'Cls:{cls_loss.val:.5f}({cls_loss.avg:.4f}) '
                        'Se:{semantic_loss.val:.5f} ({semantic_loss.avg:.4f}) '
                        'recall@1:{top1_recall.val:.2f}({top1_recall.avg:.2f}) '
                        'pre@1:{top1_prec.val:.2f}({top1_prec.avg:.2f}) '.format(
                        ep, i, len(s_loader),batch_time=batch_time,data_time=data_time,st_loss=loss_st,gcam_loss=loss_gcam,cls_loss=loss_cls,semantic_loss=loss_semantic,top1_recall=top1_recall,top1_prec=top1_prec))
                logging.info(log_str)
            #train_iter.set_description("[Train][Epoch %d]  Dist: %.5f, Angle: %.5f, Dark: %5f" %
            #                        (ep, dist_loss.item(), angle_loss.item(), dark_loss.item()))
        logging.info('[Epoch %d] Loss: %.5f, St: %.5f, Gcam: %.5f, Se: %.5f, Acc: %.5f\n' %\
            (ep, torch.Tensor(loss_all).mean(), torch.Tensor(st_loss_all).mean(), torch.Tensor(gcam_loss_all).mean(),torch.Tensor(semantic_loss_all).mean(),100*train_acc/(len(s_loader.dataset))))


    def eval_teacher(net, ap_loader, aw_loader, gyro_loader, ori_loader, ep):
        torch.cuda.empty_cache()
        K = opts.recall
        net.eval()
        #test_iter = tqdm(loader, ncols=80)
        ap_embeddings_all, aw_embeddings_all,gyro_embeddings_all,ori_embeddings_all,\
        ap_labels_all,aw_labels_all,gyro_labels_all,ori_labels_all = [], [], [], [], [], [], [], []
        ap_correct = 0
        aw_correct = 0
        gyro_correct = 0
        ori_correct = 0
        #test_iter.set_description("[Eval][Epoch %d]" % ep)
        with torch.no_grad():
            for (ap_images, ap_labels),(aw_images, aw_labels),(gyro_images, gyro_labels),(ori_images, ori_labels) in zip(ap_loader,cycle(aw_loader),gyro_loader,ori_loader):

                ap_images, ap_labels = ap_images.cuda(), ap_labels.cuda()
                aw_images, aw_labels = aw_images.cuda(), aw_labels.cuda()
                gyro_images, gyro_labels = gyro_images.cuda(), gyro_labels.cuda()
                ori_images, ori_labels = ori_images.cuda(), ori_labels.cuda()

                ap_embedding,aw_embedding,gyro_embedding,ori_embedding = net(ap_images,aw_images,gyro_images,ori_images)

                ap_pred=torch.max(ap_embedding,1)[1]
                aw_pred=torch.max(aw_embedding,1)[1]
                gyro_pred=torch.max(gyro_embedding,1)[1]
                ori_pred=torch.max(ori_embedding,1)[1]

                ap_num_correct=(ap_pred==ap_labels).sum()
                aw_num_correct=(aw_pred==ap_labels).sum()
                gyro_num_correct=(gyro_pred==gyro_labels).sum()
                ori_num_correct=(ori_pred==ori_labels).sum()

                ap_correct+=ap_num_correct.item()
                aw_correct+=aw_num_correct.item()
                gyro_correct+=gyro_num_correct.item()
                ori_correct+=ori_num_correct.item()

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

            ap_acc = ap_correct/(len(ap_loader.dataset))
            aw_acc = aw_correct/(len(aw_loader.dataset))
            gyro_acc = gyro_correct/(len(gyro_loader.dataset))
            ori_acc = ori_correct/(len(ori_loader.dataset))

            logging.info('[Epoch %d] Ap acc: [%.4f] Aw acc: [%.4f] Gyro acc: [%.4f] Ori acc: [%.4f]' % (ep, ap_acc*100, aw_acc*100, gyro_acc*100, ori_acc*100))    
        return ap_embeddings_all, aw_embeddings_all, gyro_embeddings_all, ori_embeddings_all, ap_labels_all, aw_labels_all, gyro_labels_all, ori_labels_all
    
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

    ap_prec, aw_prec, gy_prec, ori_prec, ap_labels, aw_labels, gyro_labels, ori_labels\
    =eval_teacher(teacher, acc_phone_test_loader,acc_watch_test_loader,gyro_test_loader, ori_test_loader, 0)

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
        train(acc_phone_train_loader, acc_watch_train_loader, gyro_train_loader,ori_train_loader,video_train_loader, epoch)
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