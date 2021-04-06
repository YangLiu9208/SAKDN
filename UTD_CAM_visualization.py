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
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image
from grad_cam_module import GradCAM, GradCamPlusPlus,GradCAM_two_one,GradCAM_two_two
from torchcam.utils import overlay_mask
import warnings
from torchcam.cams import CAM
from skimage import io
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
opts.batch=1
opts.img_feature_dim=300

opts.sp_ratio=0    #Similarity preserving distillation   1
opts.st_ratio=0.01       # 0.01
opts.GCAM_ratio=1
opts.semantic_ratio=1

opts.print_freq=1
opts.output_dir='output/'
opts.teacher_load='output/UTD_subject_specific_a_g_SemanticFusionVGG16_margin0.2_epochs100_batch16_lr0.0002/tea_best_acc.pth'
opts.load='save_results/UTD_SAKDN_subject_specific_AblationCAM_a_g_v_teacher_SemanticFusionVGG16_student_TRN_arch_BNInception_seg8_epochs100_batch16_lr0.001_dropout0.8/best_acc.pth'
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
    torch.backends.cudnn.benchmark = True
    # Force the pytorch to create context on the specific device 
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    logging.info('----------- Network Initialization --------------')
    model = opts.student_base(opts.num_classes, opts.num_segments, 'RGB', 
    base_model=opts.arch, consensus_type=opts.consensus_type,  dropout=opts.dropout, img_feature_dim=opts.img_feature_dim, partial_bn=True).to(device)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()
    normalize = GroupNormalize(input_mean, input_std)

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

    if opts.load is not None:
        pretrained_weights = torch.load(opts.load)
        model.load_state_dict({k.replace('module.',''):v for k,v in pretrained_weights.items()},strict=True) 
        #student.load_state_dict(torch.load(opts.load))
        logging.info("Loaded Model from %s" % opts.load)

    #if torch.cuda.device_count() > 1:
    #    model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()

    #for name, module in student._modules.items():
    #    print (name," : ",module)
    conv_layer_name = get_last_conv_name(model)
    fc_layer_name = get_last_fc_name(model)

    #activation = {}
    #def get_activation(name):
    #    def hook(model, input, output):
    #        activation[name] = output.detach()
    #    return hook

    #cam_extractor= CAM(model,conv_layer_name,fc_layer_name)

    torch.cuda.empty_cache() 
    model.eval()
    #with torch.no_grad():
        #cam_extractor._hooks_enabled = True
    for i,(images, labels) in enumerate(video_test_loader, start=1):
            #cam_extractor._hooks_enabled = True
            #pil_img = images.convert('RGB')
        raw_images=images[:,12:15,:,:].squeeze(0).permute(2,1,0)
        #sub_images = torch.unsqueeze(images[:,0:3,:,:], 0).to(device)
        sub_labels = labels[0].to(device)
        grad_cam = GradCAM(model, conv_layer_name)

        images, labels = images.cuda(), labels.cuda()
            #conv_out_conv2,conv_out_3a,conv_out_3b,conv_out_3c,conv_out_4a,conv_out_4b,\
            #conv_out_4c,conv_out_4d,conv_out_4e,conv_out_5a,conv_out_5b,video_semantic,scores = model(images)
            
        mask = grad_cam(images, sub_labels.item())  # cam mask
        GCAM_image = gen_cam(raw_images, mask)
        save_image(GCAM_image,i,sub_labels.item())
            #activation_map = cam_extractor(scores.squeeze(0).argmax().item(), scores).cpu()
            #cam_extractor.clear_hooks()
            #cam_extractor._hooks_enabled = False
            # Convert it to PIL image
            # The indexing below means first image in batch
            #heatmap = to_pil_image(activation_map, mode='F')
            # Plot the result
            #result = overlay_mask(pil_img, heatmap)
            #visualize 
            


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

def get_last_fc_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Linear):
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
    cam = heatmap + np.float32(image)/255
        
    #cv2.imwrite('cam_img.jpg',norm_image(cam))
    #masked_cam_image=masked_image.squeeze().permute(2,1,0).numpy()
    # 显示图片
    #plt.imshow(norm_image(heatmap))
    #plt.show()
    #plt.imshow(norm_image(cam))
    #plt.show()

    #plt.imshow(norm_image(masked_cam_image))
    #plt.show()

    return norm_image(cam)#,norm_image(masked_cam_image)        

def save_image(image,index,label):
    #prefix = os.path.splitext(input_image_name)[0]
    io.imsave(os.path.join('UTD_visualization', '{}_class{}.jpg'.format(index,label)), image)

if __name__ == '__main__':
    main()