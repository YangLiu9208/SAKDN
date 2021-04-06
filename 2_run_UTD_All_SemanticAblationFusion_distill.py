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
from metric.batchsampler import NPairs
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from metric.utils import recall, count_parameters_in_MB, accuracy, AverageMeter
from metric.batchsampler import NPairs
from metric.loss import HardDarkRank, RkdDistance, RKdAngle, L2Triplet, AttentionTransfer, SP, SE_Fusion,SoftTarget,Gram_loss
from model.embedding import LinearEmbedding
from TSNdataset import TSNDataSet
from transforms import *
from grad_cam_module import GradCAM, GradCamPlusPlus,GradCAM_two_one,GradCAM_two_two
import warnings
warnings.filterwarnings("ignore")
#torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--dataset', type=str, default='UTD', choices=['UTD', 'MMAct'])
parser.add_argument('--tea_train_path_one', type=str, default=r"E:/Multi-modal Action Recognition/UTD-MHAD/Inertial_a_GASF_subject_specific_train/")
parser.add_argument('--tea_test_path_one', type=str, default=r"E:/Multi-modal Action Recognition/UTD-MHAD/Inertial_a_GASF_subject_specific_test/")
parser.add_argument('--tea_train_path_two', type=str, default=r"E:/Multi-modal Action Recognition/UTD-MHAD/Inertial_g_GASF_subject_specific_train/")
parser.add_argument('--tea_test_path_two', type=str, default=r"E:/Multi-modal Action Recognition/UTD-MHAD/Inertial_g_GASF_subject_specific_test/")
parser.add_argument('--stu_video_train_path', type=str, default=r"data/UTD_rgb_train_list_subject_specific.txt")
parser.add_argument('--stu_video_test_path', type=str, default=r"data/UTD_rgb_val_list_subject_specific.txt")
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
parser.add_argument('--tea_layer_name', type=str, default=None,help='last convolutional layer name')
parser.add_argument('--stu_layer_name', type=str, default=None,help='last convolutional layer name')

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
opts.dataset='UTD_subject_specific_AblationCAM'
opts.modality='a_g_v'
opts.num_classes=27
opts.student_base=backbone.TRN
opts.consensus_type='TRNmultiscale'
opts.arch='BNInception' #BNInception
opts.num_segments=8
opts.teacher_base=backbone.SemanticFusionVGG16

opts.epochs=100
opts.lr=0.001
opts.dropout=0.8
opts.lr_decay_epochs=[50]
opts.lr_decay_gamma=0.5
opts.batch=16
opts.img_feature_dim=300

opts.sp_ratio=0    #Similarity preserving distillation   1
opts.st_ratio=0.01       # 0.01
opts.GCAM_ratio=1
opts.semantic_ratio=1

opts.print_freq=1
opts.output_dir='output/'
opts.teacher_load='output/UTD_subject_specific_a_g_SemanticFusionVGG16_margin0.2_epochs100_batch16_lr0.0002/tea_best_acc.pth'
#opts.load='output/UTD_subject_specific_AblationCAM_a_g_v_teacher_SemanticFusionVGG16_student_TRN_arch_BNInception_seg8_epochs100_batch16_lr0.001_dropout0.8/best_acc.pth'
opts.save_dir= opts.output_dir+'_'.join(map(str, ['st',str(opts.st_ratio),opts.dataset, opts.modality,'teacher','SemanticFusionVGG16','student','TRN', 
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
                                              shuffle=True, num_workers=3 , worker_init_fn=_init_fn)
                                              
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
                                             shuffle=False, num_workers=3, worker_init_fn=_init_fn)
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
    UTD_Glove=np.load('data/UTD_Glove.npy')
    UTD_Glove=torch.from_numpy(UTD_Glove)
    UTD_Glove=UTD_Glove.float().cuda()

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
        batch_size=opts.batch, shuffle=True, num_workers=3, pin_memory=False,worker_init_fn=_init_fn)

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
        batch_size=opts.batch, shuffle=False, num_workers=3, pin_memory=False,worker_init_fn=_init_fn)  

    #diy_tea_train_loader_one=DIY_loadtraindata(opts.tea_train_path_one)

    tea_train_loader_one = loadtraindata(opts.tea_train_path_one)
    tea_test_loader_one = loadtestdata(opts.tea_test_path_one)
    tea_train_loader_two = loadtraindata(opts.tea_train_path_two)
    tea_test_loader_two = loadtestdata(opts.tea_test_path_two)

    torch.cuda.empty_cache()

    if torch.cuda.device_count() > 1:
        student = torch.nn.DataParallel(student, device_ids=[0,1]).cuda()
        teacher = torch.nn.DataParallel(teacher, device_ids=[0,1]).cuda()

    logging.info("Number of images in Teacher Training Set One: %d" % len(tea_train_loader_one.dataset))
    logging.info("Number of images in Teacher Testing Set One: %d" % len(tea_test_loader_one.dataset))
    logging.info("Number of images in Teacher Training Set Two: %d" % len(tea_train_loader_two.dataset))
    logging.info("Number of images in Teacher Testing Set Two: %d" % len(tea_test_loader_two.dataset))
    logging.info("Number of videos in Student Training Set Two: %d" % len(video_train_loader.dataset))
    logging.info("Number of videos in Student Testing Set Two: %d" % len(video_test_loader.dataset))

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

    #dist_criterion = RkdDistance().cuda()
    #angle_criterion = RKdAngle().cuda()
    sp_criterion=SP().cuda()
    st_criterion = SoftTarget(4).cuda()
    GCAM_criterion=Gram_loss().cuda()
    cls_criterion=torch.nn.CrossEntropyLoss().cuda()
    criterion_semantic=torch.nn.MSELoss().cuda()
     
    #dark_criterion = HardDarkRank(alpha=opts.dark_alpha, beta=opts.dark_beta)

    #triplet_criterion = L2Triplet(sampler=opts.triplet_sample(), margin=opts.triplet_margin)
    #at_criterion = AttentionTransfer()


    def train(t_loader_one, t_loader_two, s_loader,ep):
        K = opts.recall
        batch_time = AverageMeter()
        data_time = AverageMeter()
        epoch_time = AverageMeter()
        #loss_dist = AverageMeter()
        #loss_angle= AverageMeter()
        loss_sp= AverageMeter()
        loss_gcam = AverageMeter()
        loss_st= AverageMeter()
        loss_cls= AverageMeter()
        loss_semantic= AverageMeter()
        top1_recall = AverageMeter()
        top1_prec = AverageMeter()
        
        teacher.eval()
        student.train()

        #dist_loss_all = []
        #angle_loss_all = []
        semantic_loss_all = []
        cls_loss_all=[]
        sp_loss_all = []
        st_loss_all = []
        gcam_loss_all = []
        loss_all = []
        train_acc=0.
        #train_iter = tqdm(loader)
        end = time.time()
        i=1
        torch.cuda.empty_cache() 
        for (t_images_one, t_labels_one),(t_images_two, t_labels_two),(s_videos,s_labels) in zip(t_loader_one,t_loader_two,s_loader):
            #data_time.update(time.time() - end)
            t_images_one_ablation = torch.zeros(t_images_one.size()).cuda()
            t_images_two_ablation = torch.zeros(t_images_two.size()).cuda()
            s_videos_ablation = torch.zeros(s_videos.size()).cuda()

            t_images_one, t_labels_one = t_images_one.cuda(), t_labels_one.cuda()
            t_images_two, t_labels_two = t_images_two.cuda(), t_labels_two.cuda()
            s_videos, s_labels = s_videos.cuda(), s_labels.cuda()

            t_images_one_combined=torch.cat((t_images_one,t_images_one_ablation),0)
            t_images_two_combined=torch.cat((t_images_two,t_images_two_ablation),0)
            s_videos_combiend=torch.cat((s_videos,s_videos_ablation),0)

            with torch.no_grad():
                s_out1,t_out1,s_out2,t_out2,s_out3,t_out3,s_out4,t_out4,s_out5,t_out5,s_out6,t_out6,s_out7,t_out7,s_out8,t_out8 = teacher(t_images_one_combined,t_images_two_combined, True)

            conv_out_conv2,conv_out_3a,conv_out_3b,conv_out_3c,conv_out_4a,conv_out_4b,conv_out_4c,conv_out_4d,conv_out_4e,conv_out_5a,conv_out_5b,v_semantic,video_output = student(s_videos_combiend)

            #conv_layer_name=get_last_conv_name(student)
            #conv_extract=ConvFeatureExtraction(student,conv_layer_name)
            #conv_out=conv_extract(s_videos.view((-1, 3) + s_videos.size()[-2:]))
            #conv_out=conv_out.view((-1, opts.num_segments) + conv_out.size()[1:])

            t_side=len(t_labels_one)
            s_side=len(s_labels)
            
            #w_t_one=(s_out8[0:t_side,:]-s_out8[t_side:,:]).div(s_out8[0:t_side,:]).cpu().data.numpy()
            #w_t_two=(t_out8[0:t_side,:]-t_out8[t_side:,:]).div(t_out8[0:t_side,:]).cpu().data.numpy()
            #w_s=(video_output[0:s_side,:]-video_output[s_side:,:]).div(video_output[0:s_side,:]).cpu().data.numpy()
            soft_s_out8 = F.softmax(s_out8, dim=1)
            soft_t_out8 = F.softmax(t_out8, dim=1)
            soft_video_output = F.softmax(video_output, dim=1)
            soft_s_predict=torch.max(soft_s_out8,1)[1]
            soft_t_predict=torch.max(soft_t_out8,1)[1]
            soft_video_predict=torch.max(soft_video_output,1)[1]
            soft_s_semantic=UTD_Glove[soft_s_predict]
            soft_t_semantic=UTD_Glove[soft_t_predict]
            video_semantic=UTD_Glove[soft_video_predict]

            w_t_one=(soft_s_semantic[0:t_side,:]-soft_s_semantic[t_side:,:]).div(soft_s_semantic[0:t_side,:]).cpu().data.numpy()
            w_t_two=(soft_t_semantic[0:t_side,:]-soft_t_semantic[t_side:,:]).div(soft_t_semantic[0:t_side,:]).cpu().data.numpy()
            w_s=(video_semantic[0:s_side,:]-video_semantic[s_side:,:]).div(video_semantic[0:s_side,:]).cpu().data.numpy()

            #w_t_one=np.maximum(w_t_one,0)
            #w_t_two=np.maximum(w_t_two,0)
            #w_s=np.maximum(w_s,0)

            #w_t_one -= np.min(w_t_one)
            #if np.max(w_t_one)!=0:
            #    w_t_one /= np.max(w_t_one)

            #w_t_two -= np.min(w_t_two)
            #if np.max(w_t_two)!=0:
            #    w_t_two /= np.max(w_t_two)   

            #w_s -= np.min(w_s)
            #if np.max(w_s)!=0:
            #    w_s /= np.max(w_s)         

            w_t_one=np.mean(w_t_one, axis=(1))
            w_t_two=np.mean(w_t_two, axis=(1))
            w_s=np.mean(w_s, axis=(1))

            video_semantic=UTD_Glove[s_labels]
            video_output=video_output[0:s_side,:]
            v_semantic=v_semantic[0:s_side,:]
            conv_out_conv2=conv_out_conv2[0:s_side,:]
            #conv_out_3a=conv_out_3a[0:s_side,:]
            conv_out_3c=conv_out_3c[0:s_side,:]
            #conv_out_4a=conv_out_4a[0:s_side,:]
            conv_out_4c=conv_out_4c[0:s_side,:]
            #conv_out_4e=conv_out_4e[0:s_side,:]
            conv_out_5a=conv_out_5a[0:s_side,:]
            conv_out_5b=conv_out_5b[0:s_side,:]
            s_out1=s_out1[0:t_side,:]
            t_out1=t_out1[0:t_side,:]
            s_out2=s_out2[0:t_side,:]
            t_out2=t_out2[0:t_side,:]
            s_out3=s_out3[0:t_side,:]
            t_out3=t_out3[0:t_side,:]
            s_out4=s_out4[0:t_side,:]
            t_out4=t_out4[0:t_side,:]
            s_out5=s_out5[0:t_side,:]
            t_out5=t_out5[0:t_side,:]
            s_out7=s_out7[0:t_side,:]
            t_out7=t_out7[0:t_side,:]
            s_out8=s_out8[0:t_side,:]
            t_out8=t_out8[0:t_side,:]

            s_out1_temp=s_out1.cpu().data.numpy()
            t_out1_temp=t_out1.cpu().data.numpy()
            #s_out1_temp.dtype='float64'
            #t_out1_temp.dtype='float64'
            ablation_s_out1=s_out1_temp * w_t_one[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_t_out1=t_out1_temp * w_t_two[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_s_out2=s_out2.cpu().data.numpy() * w_t_one[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_t_out2=t_out2.cpu().data.numpy() * w_t_two[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_s_out3=s_out3.cpu().data.numpy() * w_t_one[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_t_out3=t_out3.cpu().data.numpy() * w_t_two[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_s_out4=s_out4.cpu().data.numpy() * w_t_one[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_t_out4=t_out4.cpu().data.numpy() * w_t_two[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_s_out5=s_out5.cpu().data.numpy() * w_t_one[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_t_out5=t_out5.cpu().data.numpy() * w_t_two[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_s_out7=s_out7.cpu().data.numpy() * w_t_one[:, np.newaxis]  # [N,C,H,W]
            ablation_t_out7=t_out7.cpu().data.numpy() * w_t_two[:, np.newaxis]  # [N,C,H,W]
            ablation_cam_v_conv2=conv_out_conv2.cpu().data.numpy() * w_s[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_cam_v_inception_3c_pool_proj=conv_out_3c.cpu().data.numpy() * w_s[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_cam_v_inception_4c_pool_proj=conv_out_4c.cpu().data.numpy() * w_s[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            #ablation_cam_v_inception_4e_pool_proj=conv_out_4e.cpu().data.numpy() * w_s[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_cam_v_inception_5a_pool_proj=conv_out_5a.cpu().data.numpy() * w_s[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_cam_v_inception_5b_pool_proj=conv_out_5b.cpu().data.numpy() * w_s[:, np.newaxis, np.newaxis, np.newaxis]  # [N,C,H,W]
            ablation_cam_v_semantic=v_semantic.cpu().double().data.numpy() * w_s[:, np.newaxis] 

            ablation_s_out1 = np.sum(ablation_s_out1, axis=1)  # [N,H,W]
            ablation_s_out1 = np.maximum(ablation_s_out1, 0)  # ReLU
            ablation_t_out1 = np.sum(ablation_t_out1, axis=1)  # [N,H,W]
            ablation_t_out1 = np.maximum(ablation_t_out1, 0)  # ReLU
            ablation_s_out2 = np.sum(ablation_s_out2, axis=1)  # [N,H,W]
            ablation_s_out2 = np.maximum(ablation_s_out2, 0)  # ReLU
            ablation_t_out2 = np.sum(ablation_t_out2, axis=1)  # [N,H,W]
            ablation_t_out2 = np.maximum(ablation_t_out2, 0)  # ReLU
            ablation_s_out3 = np.sum(ablation_s_out3, axis=1)  # [N,H,W]
            ablation_s_out3 = np.maximum(ablation_s_out3, 0)  # ReLU
            ablation_t_out3 = np.sum(ablation_t_out3, axis=1)  # [N,H,W]
            ablation_t_out3 = np.maximum(ablation_t_out3, 0)  # ReLU
            ablation_s_out4 = np.sum(ablation_s_out4, axis=1)  # [N,H,W]
            ablation_s_out4 = np.maximum(ablation_s_out4, 0)  # ReLU
            ablation_t_out4 = np.sum(ablation_t_out4, axis=1)  # [N,H,W]
            ablation_t_out4 = np.maximum(ablation_t_out4, 0)  # ReLU
            ablation_s_out5 = np.sum(ablation_s_out5, axis=1)  # [N,H,W]
            ablation_s_out5 = np.maximum(ablation_s_out5, 0)  # ReLU
            ablation_t_out5 = np.sum(ablation_t_out5, axis=1)  # [N,H,W]
            ablation_t_out5 = np.maximum(ablation_t_out5, 0)  # ReLU
            #ablation_s_out7 = np.sum(ablation_s_out7, axis=1)  # [N,H,W]
            ablation_s_out7 = np.maximum(ablation_s_out7, 0)  # ReLU
            #ablation_t_out7 = np.sum(ablation_t_out7, axis=1)  # [N,H,W]
            ablation_t_out7 = np.maximum(ablation_t_out7, 0)  # ReLU

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
            ablation_s_out1 -= np.min(ablation_s_out1)
            if np.max(ablation_s_out1)!=0:
                ablation_s_out1 /= np.max(ablation_s_out1)

            ablation_t_out1 -= np.min(ablation_t_out1)
            if np.max(ablation_t_out1)!=0:
                ablation_t_out1 /= np.max(ablation_t_out1)  

            ablation_s_out2 -= np.min(ablation_s_out2)
            if np.max(ablation_s_out2)!=0:
                ablation_s_out2 /= np.max(ablation_s_out2)

            ablation_t_out2 -= np.min(ablation_t_out2)
            if np.max(ablation_t_out2)!=0:
                ablation_t_out2 /= np.max(ablation_t_out2)   

            ablation_s_out3 -= np.min(ablation_s_out3)
            if np.max(ablation_s_out3)!=0:
                ablation_s_out3 /= np.max(ablation_s_out3)

            ablation_t_out3 -= np.min(ablation_t_out3)
            if np.max(ablation_t_out3)!=0:
                ablation_t_out3 /= np.max(ablation_t_out3)   

            ablation_s_out4 -= np.min(ablation_s_out4)
            if np.max(ablation_s_out4)!=0:
                ablation_s_out4 /= np.max(ablation_s_out4)

            ablation_t_out4 -= np.min(ablation_t_out4)
            if np.max(ablation_t_out4)!=0:
                ablation_t_out4 /= np.max(ablation_t_out4)   

            ablation_s_out5 -= np.min(ablation_s_out5)
            if np.max(ablation_s_out5)!=0:
                ablation_s_out5 /= np.max(ablation_s_out5)

            ablation_t_out5 -= np.min(ablation_t_out5)
            if np.max(ablation_t_out5)!=0:
                ablation_t_out5 /= np.max(ablation_t_out5)    

            ablation_s_out7 -= np.min(ablation_s_out7)
            if np.max(ablation_s_out7)!=0:
                ablation_s_out7 /= np.max(ablation_s_out7)

            ablation_t_out7 -= np.min(ablation_t_out7)
            if np.max(ablation_t_out7)!=0:
                ablation_t_out7 /= np.max(ablation_t_out7)        

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
            
            ablation_s_out1 = np.resize(ablation_s_out1, (len(t_labels_one),56, 56))
            ablation_t_out1 = np.resize(ablation_t_out1, (len(t_labels_one),56, 56))
            ablation_s_out2 = np.resize(ablation_s_out2, (len(t_labels_one),14, 14))
            ablation_t_out2 = np.resize(ablation_t_out2, (len(t_labels_one),14, 14))
            ablation_s_out3 = np.resize(ablation_s_out3, (len(t_labels_one),14, 14))
            ablation_t_out3 = np.resize(ablation_t_out3, (len(t_labels_one),14, 14))
            ablation_s_out4 = np.resize(ablation_s_out4, (len(t_labels_one),7, 7))
            ablation_t_out4 = np.resize(ablation_t_out4, (len(t_labels_one),7, 7))
            ablation_5b_visualize=cv2.resize(np.squeeze(ablation_cam_v_inception_5b_pool_proj[0,:,:]), (224, 224))
            ablation_t_out5_visualize=cv2.resize(np.squeeze(ablation_t_out5[0,:,:]), (224, 224))

            ablation_s_out1 = torch.from_numpy(ablation_s_out1).cuda()
            ablation_t_out1 = torch.from_numpy(ablation_t_out1).cuda()
            ablation_s_out2 = torch.from_numpy(ablation_s_out2).cuda()
            ablation_t_out2 = torch.from_numpy(ablation_t_out2).cuda()
            ablation_s_out3 = torch.from_numpy(ablation_s_out3).cuda()
            ablation_t_out3 = torch.from_numpy(ablation_t_out3).cuda()
            ablation_s_out4 = torch.from_numpy(ablation_s_out4).cuda()
            ablation_t_out4 = torch.from_numpy(ablation_t_out4).cuda()
            ablation_s_out5 = torch.from_numpy(ablation_s_out5).cuda()
            ablation_t_out5 = torch.from_numpy(ablation_t_out5).cuda()
            ablation_s_out7 = torch.from_numpy(ablation_s_out7).cuda()
            ablation_t_out7 = torch.from_numpy(ablation_t_out7).cuda()

            ablation_cam_v_conv2=torch.from_numpy(ablation_cam_v_conv2).cuda()
            ablation_cam_v_inception_3c_pool_proj=torch.from_numpy(ablation_cam_v_inception_3c_pool_proj).cuda()
            ablation_cam_v_inception_4c_pool_proj=torch.from_numpy(ablation_cam_v_inception_4c_pool_proj).cuda()
            #ablation_cam_v_inception_4e_pool_proj=torch.from_numpy(ablation_cam_v_inception_4e_pool_proj).cuda()
            ablation_cam_v_inception_5a_pool_proj=torch.from_numpy(ablation_cam_v_inception_5a_pool_proj).cuda()
            ablation_cam_v_inception_5b_pool_proj=torch.from_numpy(ablation_cam_v_inception_5b_pool_proj).cuda()
            ablation_cam_v_semantic=torch.from_numpy(ablation_cam_v_semantic).cuda()

            #visualize 
            #CAM_image = gen_cam(s_videos[0,0:3,:,:].permute(1,2,0).cpu().data.numpy(), ablation_5b_visualize)
            #CAM_image2 = gen_cam(t_images_one[0,:].permute(1,2,0).cpu().data.numpy(), ablation_t_out5_visualize)

            pred=torch.max(video_output,1)[1]
            train_correct=(pred==s_labels).sum()
            train_acc+=train_correct.item()

            #dist_loss = opts.dist_ratio * (dist_criterion(v_semantic, s_out7)+dist_criterion(v_semantic, t_out7))/2.0

            #angle_loss = opts.angle_ratio * (angle_criterion(v_semantic, s_out7)+angle_criterion(v_semantic, t_out7))/2.0

            #sp_loss= opts.sp_ratio * (sp_criterion( ablation_cam_v_semantic,ablation_s_out7)+sp_criterion(ablation_cam_v_semantic,ablation_t_out7))/2.0

            st_loss=opts.st_ratio * (st_criterion( video_output,s_out8)+st_criterion(video_output,t_out8))/2.0

            gcam_loss=opts.GCAM_ratio * (GCAM_criterion(ablation_cam_v_conv2,ablation_s_out1)+GCAM_criterion(ablation_cam_v_conv2,ablation_t_out1)+\
                GCAM_criterion(ablation_cam_v_inception_3c_pool_proj,ablation_s_out2)+GCAM_criterion(ablation_cam_v_inception_3c_pool_proj,ablation_t_out2)+\
                GCAM_criterion(ablation_cam_v_inception_4c_pool_proj,ablation_s_out3)+GCAM_criterion(ablation_cam_v_inception_4c_pool_proj,ablation_t_out3)+\
                #GCAM_criterion(ablation_cam_v_inception_4e_pool_proj,ablation_s_out4)+GCAM_criterion(ablation_cam_v_inception_4e_pool_proj,ablation_t_out4)+\
                GCAM_criterion(ablation_cam_v_inception_5a_pool_proj,ablation_s_out4)+GCAM_criterion(ablation_cam_v_inception_5a_pool_proj,ablation_t_out4)+\
                GCAM_criterion(ablation_cam_v_inception_5b_pool_proj,ablation_s_out5)+GCAM_criterion(ablation_cam_v_inception_5b_pool_proj,ablation_t_out5))/10.0

            cls_loss=cls_criterion(video_output,s_labels)#+cls_criterion(video_output,s_out8)+cls_criterion(video_output,t_out8)

            semantic_loss=opts.semantic_ratio*(criterion_semantic(v_semantic, s_out7)+criterion_semantic(v_semantic, t_out7))/2.0

            loss = st_loss+gcam_loss+cls_loss+semantic_loss #+ at_loss +
            #loss = cls_loss+semantic_loss #+ at_loss

            #loss_dist.update(dist_loss.item(), s_videos.size(0))
            #loss_angle.update(angle_loss.item(), s_videos.size(0))
            #loss_sp.update(sp_loss.item(), s_videos.size(0))
            loss_gcam.update(gcam_loss.item(), s_videos.size(0))
            loss_st.update(st_loss.item(), s_videos.size(0))
            loss_cls.update(cls_loss.item(), s_videos.size(0))
            loss_semantic.update(semantic_loss.item(), s_videos.size(0))

            rec = recall(video_output, s_labels, K=K)
            prec = accuracy(video_output, s_labels, topk=(1,))
            top1_recall.update(rec[0], s_videos.size(0))
            top1_prec.update(prec[0]/100, s_videos.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #sp_loss_all.append(sp_loss.item())
            st_loss_all.append(st_loss.item())
            gcam_loss_all.append(gcam_loss.item())
            cls_loss_all.append(cls_loss.item())
            semantic_loss_all.append(semantic_loss.item())
            loss_all.append(loss.item())
            #batch_time.update(time.time() - end)
            #end = time.time()
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
            i=i+1
        epoch_time.update(time.time() - end)
        logging.info('[Epoch %d] Time:  %.5f, Loss: %.5f, St: %.5f, Gcam: %.5f, Se: %.5f, Acc: %.5f\n' %\
            (ep, epoch_time.val, torch.Tensor(loss_all).mean(), torch.Tensor(st_loss_all).mean(), torch.Tensor(gcam_loss_all).mean(),torch.Tensor(semantic_loss_all).mean(),100*train_acc/(len(s_loader.dataset))))


    def eval_teacher(net, t_loader, s_loader, ep):
        torch.cuda.empty_cache() 
        K = opts.recall
        net.eval()
        #test_iter = tqdm(loader, ncols=80)
        s_embeddings_all, t_embeddings_all,s_labels_all,t_labels_all = [], [], [], []
        s_correct = 0
        t_correct = 0
        #test_iter.set_description("[Eval][Epoch %d]" % ep)
        with torch.no_grad():
            for (t_images, t_labels),(s_images, s_labels) in zip(t_loader,s_loader):
                t_images, t_labels = t_images.cuda(), t_labels.cuda()
                s_images, s_labels = s_images.cuda(), s_labels.cuda()
                s_embedding, t_embedding = net(s_images,t_images)
                s_pred=torch.max(s_embedding,1)[1]
                t_pred=torch.max(t_embedding,1)[1]
                s_num_correct=(s_pred==s_labels).sum()
                t_num_correct=(t_pred==t_labels).sum()
                s_correct+=s_num_correct.item()
                t_correct+=t_num_correct.item()
                s_embeddings_all.append(s_embedding.data)
                t_embeddings_all.append(t_embedding.data)
                s_labels_all.append(s_labels.data)  
                t_labels_all.append(t_labels.data)
            s_embeddings_all = torch.cat(s_embeddings_all).cpu()
            t_embeddings_all = torch.cat(t_embeddings_all).cpu()
            s_labels_all = torch.cat(s_labels_all).cpu()
            t_labels_all = torch.cat(t_labels_all).cpu()
            #rec = recall(embeddings_all, labels_all, K=K)
            s_prec = accuracy(s_embeddings_all, s_labels_all, topk=(1,))
            s_acc = s_correct/(len(s_loader.dataset))
            t_acc = t_correct/(len(t_loader.dataset))
            logging.info('[Epoch %d] Teacher 1 acc: [%.4f] Teacher 2 acc: [%.4f]' % (ep, s_acc*100, t_acc*100))    
        return s_embeddings_all, t_embeddings_all
    
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
                conv_out_conv2,conv_out_3a,conv_out_3b,conv_out_3c,conv_out_4a,conv_out_4b,\
                conv_out_4c,conv_out_4d,conv_out_4e,conv_out_5a,conv_out_5b,video_semantic,video_output = net(images)
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

    one_prec, two_prec=eval_teacher(teacher, tea_test_loader_one,tea_test_loader_two, 0)

    logging.info('----------- Student Network performance  --------------')

    best_val_recall, best_val_acc, s_prec, labels_all  = eval_student(student, video_test_loader, 0)

    combined_acc=accuracy((one_prec+two_prec+s_prec), labels_all, topk=(1,))
    best_combined_acc=combined_acc[0]
    combined_rec = recall((one_prec+two_prec+s_prec), labels_all, K=[1])
    best_combined_rec=combined_rec[0]
    logging.info('----------- Combined Network performance  --------------')
    logging.info('Combined acc: [%.4f]\n' % (best_combined_acc)) 
    logging.info('Combined rec: [%.4f]\n' % (best_combined_rec)) 
    for epoch in range(1, opts.epochs+1):
        adjust_learning_rate(optimizer, epoch, opts.lr_decay_epochs)
        train(tea_train_loader_one, tea_train_loader_two,video_train_loader, epoch)
        val_recall, val_acc, s_prec, labels_all = eval_student(student, video_test_loader, epoch)
        val_combined_acc=accuracy((one_prec+two_prec+s_prec), labels_all, topk=(1,))
        val_comb_acc=val_combined_acc[0]
        val_combined_rec = recall((one_prec+two_prec+s_prec), labels_all, K=[1])
        val_comb_rec=val_combined_rec[0]
        logging.info('[Epoch %d] combined acc: [%.4f]' % (epoch, val_comb_acc)) 
        logging.info('[Epoch %d] combined rec: [%.4f]\n' % (epoch, val_comb_rec*100)) 
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

        if best_combined_acc < val_comb_acc:
            best_combined_acc = val_comb_acc
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(student.state_dict(), "%s/%s"%(opts.save_dir, "best_combined.pth"))

        if best_combined_rec < val_comb_rec:
            best_combined_rec = val_comb_rec
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(student.state_dict(), "%s/%s"%(opts.save_dir, "best_combined_rec.pth"))        
   
        F_measure=(2*best_val_acc*best_val_recall)/(best_val_acc+best_val_recall)
        combined_F_measure=(2*best_combined_acc/100*best_combined_rec)/(best_combined_acc/100+best_combined_rec)
        if opts.save_dir is not None:
            if not os.path.isdir(opts.save_dir):
                os.mkdir(opts.save_dir)
            torch.save(student.state_dict(), "%s/%s" % (opts.save_dir, "last.pth"))
            with open("%s/result.txt" % opts.save_dir, 'w') as f:
                f.write("Best Test recall@1: %.4f\n" % (best_val_recall * 100))
                f.write("Final Recall@1: %.4f\n" % (val_recall * 100))
                f.write("Best Test Acc: %.4f\n" % (best_val_acc * 100))
                f.write("Final Acc: %.4f\n" % (val_acc * 100))
                f.write("Best Combined Acc: %.4f\n" % (best_combined_acc))
                f.write("Final Combined Acc: %.4f\n" % (val_comb_acc ))
                f.write("F-measure: %.4f\n" % (F_measure*100))
                f.write("Combined F-measure: %.4f\n" % (combined_F_measure*100))

        logging.info("Best Eval Recall@1: %.4f" % (best_val_recall*100))
        logging.info("Best Eval Acc: %.4f" % (best_val_acc*100))
        logging.info("Best Eval Combined Acc: %.4f" % (best_combined_acc))
        logging.info("Eval F-measure: %.4f" % (F_measure*100))
        logging.info("Eval Combined F-measure: %.4f\n" % (combined_F_measure*100))

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = opts.lr * decay
    decay = 5e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name        

class ConvFeatureExtraction(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs):
        """
        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]

        feature = self.feature  # [C,H,W]

        return feature  

def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def gen_cam(image, mask):#, masked_image):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb
    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
        
    #cv2.imwrite('cam_img.jpg',norm_image(cam))
    #masked_cam_image=masked_image.squeeze().permute(2,1,0).numpy()
    # 显示图片
    plt.imshow(norm_image(heatmap))
    plt.show()
    plt.imshow(norm_image(cam))
    plt.show()

    #plt.imshow(norm_image(masked_cam_image))
    #plt.show()

    return norm_image(cam)#,norm_image(masked_cam_image)        

if __name__ == '__main__':
    main()