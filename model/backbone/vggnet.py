import torchvision
import torch.nn as nn
import argparse
import torch
import torch.nn.functional as F
import torch.nn.init as init
__all__ = ['VggNet16','SeVggNet16','SeFusionVGG16','SeFusionVGG16_MMAct','SeFusionVGG16_Berkeley','SemanticFusionVGG16','SemanticFusionVGG16_MMAct','SemanticFusionVGG16_Berkeley']

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

class SelfAttention(nn.Module):
    r"""
        Self attention Layer.
        Source paper: https://arxiv.org/abs/1805.08318
    """
    def __init__(self, in_dim, activation=F.relu):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.f = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        init_conv(self.f)
        init_conv(self.g)
        init_conv(self.h)
        
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention feature maps
                
        """
        m_batchsize, C, width, height = x.size()
        
        f = self.f(x).view(m_batchsize, -1, width * height) # B * (C//8) * (W * H)
        g = self.g(x).view(m_batchsize, -1, width * height) # B * (C//8) * (W * H)
        h = self.h(x).view(m_batchsize, -1, width * height) # B * C * (W * H)
        
        attention = torch.bmm(f.permute(0, 2, 1), g) # B * (W * H) * (W * H)
        attention = self.softmax(attention)
        
        self_attetion = torch.bmm(h, attention) # B * C * (W * H)
        self_attetion = self_attetion.view(m_batchsize, C, width, height) # B * C * W * H
        
        out = self.gamma * self_attetion + x
        return out

class SE_Fusion(nn.Module):
    def __init__(self,channel_s=128,channel_t=256,reduction=4):
        super(SE_Fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_s = nn.Sequential(
            nn.Linear((channel_s+channel_t), (channel_s+channel_t) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_s+channel_t) // reduction, channel_s, bias=True),
            nn.Sigmoid()
        )
        self.fc_t = nn.Sequential(
            nn.Linear((channel_s+channel_t), (channel_s+channel_t) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_s+channel_t) // reduction, channel_t, bias=True),
            nn.Sigmoid()
        )

    def forward(self, student, teacher):

        #squeeze
        if len(student.size())<4:
            b_s, c_s= student.size()
            y_s = student
            b_t, c_t= teacher.size()
            y_t = teacher
        else:
            b_s, c_s, _, _ = student.size()
            y_s = self.avg_pool(student).view(b_s, c_s)
            b_t, c_t, _, _ = teacher.size()
            y_t = self.avg_pool(teacher).view(b_t, c_t)
        #joint
        z=torch.cat((y_s,y_t),1)
        #excitation
        if len(student.size())<4:
            y_s = self.fc_s(z).view(b_s, c_s)
            y_t = self.fc_t(z).view(b_t, c_t)   
        else:
            y_s = self.fc_s(z).view(b_s, c_s, 1, 1)
            y_t = self.fc_t(z).view(b_t, c_t, 1, 1)        

        
        return 2*student * y_s.expand_as(student), 2*teacher * y_t.expand_as(teacher)

class SE_semantic_Fusion(nn.Module):
    def __init__(self,channel_s=128,channel_t=256,reduction=4):
        super(SE_semantic_Fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_s = nn.Sequential(
            nn.Linear((channel_s+channel_t), (channel_s+channel_t) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_s+channel_t) // reduction, channel_s, bias=True),
            nn.Sigmoid()
        )
        self.fc_s_add = nn.Sequential(
            nn.Linear((channel_s), (channel_s) //reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_s) //reduction, channel_s, bias=True),
            nn.Sigmoid()
        )
        self.fc_s_mul = nn.Sequential(
            nn.Linear((channel_s), (channel_s) //reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_s) //reduction, channel_s, bias=True),
            nn.Sigmoid()
        )
        self.fc_t = nn.Sequential(
            nn.Linear((channel_s+channel_t), (channel_s+channel_t) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_s+channel_t) //reduction, channel_t, bias=True),
            nn.Sigmoid()
        )
        self.fc_t_add = nn.Sequential(
            nn.Linear((channel_t), (channel_t) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_t) // reduction, channel_t, bias=True),
            nn.Sigmoid()
        )
        self.fc_t_mul = nn.Sequential(
            nn.Linear((channel_t), (channel_t) //reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_t) //reduction, channel_t, bias=True),
            nn.Sigmoid()
        )
        #self.fc_s_all = nn.Sequential(
        #    nn.Linear((channel_s+channel_t), channel_s, bias=True),
        #    nn.Sigmoid()
        #)
        #self.fc_t_all = nn.Sequential(
        #    nn.Linear((channel_s+channel_t), channel_t , bias=True),
        #    nn.Sigmoid()
        #)

    def forward(self, student, teacher):

        #squeeze
        if len(student.size())<4:
            b_s, c_s= student.size()
            y_s = student

            fm_s = student.view(student.size(0), -1)
            G_s  = torch.mm(fm_s, fm_s.t())
            norm_G_s = F.normalize(G_s, p=2, dim=1)

            b_t, c_t= teacher.size()
            y_t = teacher

            fm_t = teacher.view(teacher.size(0), -1)
            G_t  = torch.mm(fm_t, fm_t.t())
            norm_G_t = F.normalize(G_t, p=2, dim=1)

            #G_st = torch.mm(fm_s,fm_t.t())
            #norm_G_st = F.normalize(G_st, p=2, dim=1)

        else:
            b_s, c_s, _, _ = student.size()
            fm_s = student.view(student.size(0), -1)
            G_s  = torch.mm(fm_s, fm_s.t())
            norm_G_s = F.normalize(G_s, p=2, dim=1)
            y_s = self.avg_pool(student).view(b_s, c_s)

            b_t, c_t, _, _ = teacher.size()
            fm_t = teacher.view(teacher.size(0), -1)
            G_t  = torch.mm(fm_t, fm_t.t())
            norm_G_t = F.normalize(G_t, p=2, dim=1)
            y_t = self.avg_pool(teacher).view(b_t, c_t)

            #G_st = torch.mm(fm_s,fm_t.t())
            #norm_G_st = F.normalize(G_st, p=2, dim=1)

        #joint
        z_add=torch.add(torch.mm(norm_G_s,y_s),torch.mm(norm_G_t,y_t))
        z_mul=torch.mul(torch.mm(norm_G_s,y_s),torch.mm(norm_G_t,y_t))
        z=torch.cat((torch.mm(norm_G_s,y_s),torch.mm(norm_G_t,y_t)),1)
        #z=F.normalize(z, p=2, dim=1)
        #excitation
        if len(student.size())<4:
            y_s =self.fc_s(z).view(b_s, c_s)
            y_s_add =  self.fc_s_add(z_add).view(b_s, c_s)
            y_s_mul =  self.fc_s_mul(z_mul).view(b_s, c_s)
            #y_s_all = self.fc_s_all(torch.cat((self.fc_s(z),self.fc_s_add(z_add)),1)).view(b_s, c_s)
            y_t = self.fc_t(z).view(b_t, c_t)  
            y_t_add=self.fc_t_add(z_add).view(b_t, c_t)
            y_t_mul=self.fc_t_mul(z_mul).view(b_t, c_t)
            #y_t_all = self.fc_t_all(torch.cat((self.fc_t(z),self.fc_t_add(z_add)),1)).view(b_t, c_t)
        else:
            y_s = self.fc_s(z).view(b_s, c_s, 1 ,1)
            y_s_add=  self.fc_s_add(z_add).view(b_s, c_s,1,1)
            y_s_mul =  self.fc_s_mul(z_mul).view(b_s, c_s,1,1)
            #y_s_all = self.fc_s_all(torch.cat((self.fc_s(z),self.fc_s_add(z_add)),1)).view(b_s, c_s, 1, 1)
            y_t = self.fc_t(z).view(b_t, c_t, 1, 1)  
            y_t_add= self.fc_t_add(z_add).view(b_t, c_t, 1, 1)   
            y_t_mul=self.fc_t_mul(z_mul).view(b_t, c_t,1,1) 
            #y_t_all = self.fc_t_all(torch.cat((self.fc_t(z),self.fc_t_add(z_add)),1)).view(b_t, c_t, 1, 1)
        
        return torch.mul((F.relu(y_s)+F.relu(y_s_add)+F.relu(y_s_mul)).expand_as(student),student), torch.mul((F.relu(y_t)+F.relu(y_t_add)+F.relu(y_t_mul)).expand_as(teacher),teacher)  #torch.add(torch.mul(F.sigmoid(y_s).expand_as(student),student),torch.mul(F.sigmoid(y_s_add).expand_as(student),student)), torch.add(torch.mul(F.sigmoid(y_t).expand_as(teacher),teacher),torch.mul(F.sigmoid(y_t_add).expand_as(teacher),teacher))   #2*student * y_s.expand_as(student), 2*teacher * y_t.expand_as(teacher)

class SE_semantic_Fusion_MMAct(nn.Module):
    def __init__(self,channel_ap=128,channel_aw=128,channel_gyro=128,channel_ori=256,reduction=4):
        super(SE_semantic_Fusion_MMAct, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_ap = nn.Sequential(
            nn.Linear((channel_ap+channel_aw+channel_gyro+channel_ori), (channel_ap+channel_aw+channel_gyro+channel_ori) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_ap+channel_aw+channel_gyro+channel_ori) // reduction, channel_ap, bias=True),
            nn.Sigmoid()
        )
        self.fc_ap_add = nn.Sequential(
            nn.Linear((channel_ap), (channel_ap) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_ap) // reduction, channel_ap, bias=True),
            nn.Sigmoid()
        )
        self.fc_ap_mul = nn.Sequential(
            nn.Linear((channel_ap), (channel_ap) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_ap) // reduction, channel_ap, bias=True),
            nn.Sigmoid()
        )
        self.fc_aw = nn.Sequential(
            nn.Linear((channel_ap+channel_aw+channel_gyro+channel_ori), (channel_ap+channel_aw+channel_gyro+channel_ori) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_ap+channel_aw+channel_gyro+channel_ori) // reduction, channel_aw, bias=True),
            nn.Sigmoid()
        )
        self.fc_aw_add = nn.Sequential(
            nn.Linear((channel_aw), (channel_aw) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_aw) // reduction, channel_aw, bias=True),
            nn.Sigmoid()
        )
        self.fc_aw_mul = nn.Sequential(
            nn.Linear((channel_aw), (channel_aw) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_aw) // reduction, channel_aw, bias=True),
            nn.Sigmoid()
        )
        self.fc_gy = nn.Sequential(
            nn.Linear((channel_ap+channel_aw+channel_gyro+channel_ori), (channel_ap+channel_aw+channel_gyro+channel_ori) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_ap+channel_aw+channel_gyro+channel_ori) // reduction, channel_gyro, bias=True),
            nn.Sigmoid()
        )
        self.fc_gy_add = nn.Sequential(
            nn.Linear((channel_gyro), (channel_gyro) //reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_gyro) // reduction, channel_gyro, bias=True),
            nn.Sigmoid()
        )
        self.fc_gy_mul = nn.Sequential(
            nn.Linear((channel_gyro), (channel_gyro) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_gyro) // reduction, channel_gyro, bias=True),
            nn.Sigmoid()
        )
        self.fc_ori = nn.Sequential(
            nn.Linear((channel_ap+channel_aw+channel_gyro+channel_ori), (channel_ap+channel_aw+channel_gyro+channel_ori) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_ap+channel_aw+channel_gyro+channel_ori) // reduction, channel_ori, bias=True),
            nn.Sigmoid()
        )
        self.fc_ori_add = nn.Sequential(
            nn.Linear((channel_ori), (channel_ori) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_ori) // reduction, channel_ori, bias=True),
            nn.Sigmoid()
        )
        self.fc_ori_mul = nn.Sequential(
            nn.Linear((channel_ori), (channel_ori) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_ori) // reduction, channel_ori, bias=True),
            nn.Sigmoid()
        )

    def forward(self, ap, aw, gy, ori):

        #squeeze
        if len(ap.size())<4:
            b_ap, c_ap= ap.size()
            b_aw, c_aw= aw.size()
            b_gy, c_gy= gy.size()
            b_ori, c_ori= ori.size()

            y_ap = ap
            fm_ap = ap.view(ap.size(0), -1)
            G_ap  = torch.mm(fm_ap, fm_ap.t())
            norm_G_ap = F.normalize(G_ap, p=2, dim=1)

            y_aw = aw
            fm_aw = aw.view(aw.size(0), -1)
            G_aw  = torch.mm(fm_aw, fm_aw.t())
            norm_G_aw = F.normalize(G_aw, p=2, dim=1)

            y_gy =gy
            fm_gy = gy.view(gy.size(0), -1)
            G_gy  = torch.mm(fm_gy, fm_gy.t())
            norm_G_gy = F.normalize(G_gy, p=2, dim=1)

            y_ori = ori
            fm_ori = ori.view(ori.size(0), -1)
            G_ori  = torch.mm(fm_ori, fm_ori.t())
            norm_G_ori = F.normalize(G_ori, p=2, dim=1)
        else:
            b_ap, c_ap, _, _ = ap.size()
            fm_ap = ap.view(ap.size(0), -1)
            G_ap  = torch.mm(fm_ap, fm_ap.t())
            norm_G_ap = F.normalize(G_ap, p=2, dim=1)
            y_ap = self.avg_pool(ap).view(b_ap, c_ap)

            b_aw, c_aw, _, _ = aw.size()
            fm_aw = aw.view(aw.size(0), -1)
            G_aw  = torch.mm(fm_aw, fm_aw.t())
            norm_G_aw = F.normalize(G_aw, p=2, dim=1)
            y_aw = self.avg_pool(aw).view(b_aw, c_aw)

            b_gy, c_gy, _, _ = gy.size()
            fm_gy = gy.view(gy.size(0), -1)
            G_gy  = torch.mm(fm_gy, fm_gy.t())
            norm_G_gy = F.normalize(G_gy, p=2, dim=1)
            y_gy = self.avg_pool(gy).view(b_gy, c_gy)

            b_ori, c_ori, _, _ = ori.size()
            fm_ori = ori.view(ori.size(0), -1)
            G_ori  = torch.mm(fm_ori, fm_ori.t())
            norm_G_ori = F.normalize(G_ori, p=2, dim=1)
            y_ori = self.avg_pool(ori).view(b_ori, c_ori)

        #joint
        z=torch.cat((torch.mm(norm_G_ap,y_ap),torch.mm(norm_G_aw,y_aw),torch.mm(norm_G_gy,y_gy),torch.mm(norm_G_ori,y_ori)),1)
        z_add=torch.mm(norm_G_ap,y_ap)+torch.mm(norm_G_aw,y_aw)+torch.mm(norm_G_gy,y_gy)+torch.mm(norm_G_ori,y_ori)
        z_mul=torch.mm(norm_G_ap,y_ap)*torch.mm(norm_G_aw,y_aw)*torch.mm(norm_G_gy,y_gy)*torch.mm(norm_G_ori,y_ori)

        #z=F.normalize(z, p=2, dim=1)
        #excitation
        if len(ap.size())<4:
            y_ap = self.fc_ap(z).view(b_ap, c_ap)
            y_ap_add=self.fc_ap_add(z_add).view(b_ap, c_ap)
            y_ap_mul=self.fc_ap_mul(z_mul).view(b_ap, c_ap)
            y_aw = self.fc_aw(z).view(b_aw, c_aw)   
            y_aw_add=self.fc_aw_add(z_add).view(b_aw, c_aw)
            y_aw_mul=self.fc_aw_mul(z_mul).view(b_aw, c_aw)
            y_gy = self.fc_gy(z).view(b_gy, c_gy)
            y_gy_add = self.fc_gy_add(z_add).view(b_gy, c_gy)
            y_gy_mul = self.fc_gy_mul(z_mul).view(b_gy, c_gy)
            y_ori = self.fc_ori(z).view(b_ori, c_ori)   
            y_ori_add = self.fc_ori_add(z_add).view(b_ori, c_ori) 
            y_ori_mul = self.fc_ori_mul(z_mul).view(b_ori, c_ori) 
        else:
            y_ap = self.fc_ap(z).view(b_ap, c_ap, 1 , 1)
            y_ap_add=self.fc_ap_add(z_add).view(b_ap, c_ap,1,1)
            y_ap_mul=self.fc_ap_mul(z_mul).view(b_ap, c_ap,1,1)
            y_aw = self.fc_aw(z).view(b_aw, c_aw, 1 , 1) 
            y_aw_add=self.fc_aw_add(z_add).view(b_aw, c_aw,1,1)
            y_aw_mul=self.fc_aw_mul(z_mul).view(b_aw, c_aw,1,1)  
            y_gy = self.fc_gy(z).view(b_gy, c_gy, 1 , 1)
            y_gy_add = self.fc_gy_add(z_add).view(b_gy, c_gy,1,1)
            y_gy_mul = self.fc_gy_mul(z_mul).view(b_gy, c_gy,1,1)
            y_ori = self.fc_ori(z).view(b_ori, c_ori, 1 , 1)
            y_ori_add = self.fc_ori_add(z_add).view(b_ori, c_ori,1,1) 
            y_ori_mul = self.fc_ori_mul(z_mul).view(b_ori, c_ori,1,1)        

        
        return torch.mul((F.relu(y_ap)+F.relu(y_ap_add)+F.relu(y_ap_mul)).expand_as(ap),ap), \
               torch.mul((F.relu(y_aw)+F.relu(y_aw_add)+F.relu(y_aw_mul)).expand_as(aw),aw), \
               torch.mul((F.relu(y_gy)+F.relu(y_gy_add)+F.relu(y_gy_mul)).expand_as(gy),gy),\
               torch.mul((F.relu(y_ori)+F.relu(y_ori_add)+F.relu(y_ori_mul)).expand_as(ori),ori) #2*ap*y_ap.expand_as(ap), 2*aw*y_aw.expand_as(aw),2*gy*y_gy.expand_as(gy), 2*ori*y_ori.expand_as(ori)

class SE_semantic_Fusion_Berkeley(nn.Module):
    def __init__(self,channel_1=128,channel_2=128,channel_3=128,channel_4=256,channel_5=128,channel_6=256,reduction=4):
        super(SE_semantic_Fusion_Berkeley, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_a1 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4+channel_5+channel_6), (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, channel_1, bias=True),
            nn.Sigmoid()
        )
        self.fc_a1_add = nn.Sequential(
            nn.Linear((channel_1), (channel_1) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1) // reduction, channel_1, bias=True),
            nn.Sigmoid()
        )
        self.fc_a1_mul = nn.Sequential(
            nn.Linear((channel_1), (channel_1) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1) // reduction, channel_1, bias=True),
            nn.Sigmoid()
        )
        self.fc_a2 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4+channel_5+channel_6), (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, channel_2, bias=True),
            nn.Sigmoid()
        )
        self.fc_a2_add = nn.Sequential(
            nn.Linear((channel_2), (channel_2) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_2) // reduction, channel_2, bias=True),
            nn.Sigmoid()
        )
        self.fc_a2_mul = nn.Sequential(
            nn.Linear((channel_2), (channel_2) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_2) // reduction, channel_2, bias=True),
            nn.Sigmoid()
        )
        self.fc_a3 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4+channel_5+channel_6), (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, channel_3, bias=True),
            nn.Sigmoid()
        )
        self.fc_a3_add = nn.Sequential(
            nn.Linear((channel_3), (channel_3) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_3) // reduction, channel_3, bias=True),
            nn.Sigmoid()
        )
        self.fc_a3_mul = nn.Sequential(
            nn.Linear((channel_3), (channel_3) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_3) // reduction, channel_3, bias=True),
            nn.Sigmoid()
        )
        self.fc_a4 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4+channel_5+channel_6), (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, channel_4, bias=True),
            nn.Sigmoid()
        )
        self.fc_a4_add = nn.Sequential(
            nn.Linear((channel_4), (channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_4) // reduction, channel_4, bias=True),
            nn.Sigmoid()
        )
        self.fc_a4_mul = nn.Sequential(
            nn.Linear((channel_4), (channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_4) // reduction, channel_4, bias=True),
            nn.Sigmoid()
        )
        self.fc_a5 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4+channel_5+channel_6), (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, channel_5, bias=True),
            nn.Sigmoid()
        )
        self.fc_a5_add = nn.Sequential(
            nn.Linear((channel_5), (channel_5) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_5) // reduction, channel_5, bias=True),
            nn.Sigmoid()
        )
        self.fc_a5_mul = nn.Sequential(
            nn.Linear((channel_5), (channel_5) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_5) // reduction, channel_5, bias=True),
            nn.Sigmoid()
        )
        self.fc_a6 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4+channel_5+channel_6), (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4+channel_5+channel_6) // reduction, channel_6, bias=True),
            nn.Sigmoid()
        )
        self.fc_a6_add = nn.Sequential(
            nn.Linear((channel_6), (channel_6) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_6) // reduction, channel_6, bias=True),
            nn.Sigmoid()
        )
        self.fc_a6_mul = nn.Sequential(
            nn.Linear((channel_6), (channel_6) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_6) // reduction, channel_6, bias=True),
            nn.Sigmoid()
        )

    def forward(self, a1, a2, a3, a4, a5, a6):

        #squeeze
        if len(a1.size())<4:
            b_a1, c_a1= a1.size()
            b_a2, c_a2= a2.size()
            b_a3, c_a3= a3.size()
            b_a4, c_a4= a4.size()
            b_a5, c_a5= a5.size()
            b_a6, c_a6= a6.size()

            y_a1 = a1
            fm_a1 = a1.view(a1.size(0), -1)
            G_a1  = torch.mm(fm_a1, fm_a1.t())
            norm_G_a1 = F.normalize(G_a1, p=2, dim=1)

            y_a2 = a2
            fm_a2= a2.view(a2.size(0), -1)
            G_a2  = torch.mm(fm_a2, fm_a2.t())
            norm_G_a2 = F.normalize(G_a2, p=2, dim=1)

            y_a3 = a3
            fm_a3 = a3.view(a3.size(0), -1)
            G_a3  = torch.mm(fm_a3, fm_a3.t())
            norm_G_a3 = F.normalize(G_a3, p=2, dim=1)

            y_a4 = a4
            fm_a4= a4.view(a4.size(0), -1)
            G_a4  = torch.mm(fm_a4, fm_a4.t())
            norm_G_a4 = F.normalize(G_a4, p=2, dim=1)

            y_a5 = a5
            fm_a5 = a5.view(a5.size(0), -1)
            G_a5  = torch.mm(fm_a5, fm_a5.t())
            norm_G_a5 = F.normalize(G_a5, p=2, dim=1)

            y_a6 = a6
            fm_a6= a6.view(a6.size(0), -1)
            G_a6  = torch.mm(fm_a6, fm_a6.t())
            norm_G_a6 = F.normalize(G_a6, p=2, dim=1)
        else:
            b_a1, c_a1, _, _ = a1.size()
            fm_a1 = a1.view(a1.size(0), -1)
            G_a1  = torch.mm(fm_a1, fm_a1.t())
            norm_G_a1 = F.normalize(G_a1, p=2, dim=1)
            y_a1 = self.avg_pool(a1).view(b_a1, c_a1)

            b_a2, c_a2, _, _ = a2.size()
            fm_a2 = a2.view(a2.size(0), -1)
            G_a2  = torch.mm(fm_a2, fm_a2.t())
            norm_G_a2 = F.normalize(G_a2, p=2, dim=1)
            y_a2 = self.avg_pool(a2).view(b_a2, c_a2)

            b_a3, c_a3, _, _ = a3.size()
            fm_a3 = a3.view(a3.size(0), -1)
            G_a3  = torch.mm(fm_a3, fm_a3.t())
            norm_G_a3 = F.normalize(G_a3, p=2, dim=1)
            y_a3 = self.avg_pool(a3).view(b_a3, c_a3)

            b_a4, c_a4, _, _ = a4.size()
            fm_a4 = a4.view(a4.size(0), -1)
            G_a4  = torch.mm(fm_a4, fm_a4.t())
            norm_G_a4 = F.normalize(G_a4, p=2, dim=1)
            y_a4 = self.avg_pool(a4).view(b_a4, c_a4)

            b_a5, c_a5, _, _ = a5.size()
            fm_a5 = a5.view(a5.size(0), -1)
            G_a5  = torch.mm(fm_a5, fm_a5.t())
            norm_G_a5 = F.normalize(G_a5, p=2, dim=1)
            y_a5 = self.avg_pool(a5).view(b_a5, c_a5)

            b_a6, c_a6, _, _ = a6.size()
            fm_a6 = a6.view(a6.size(0), -1)
            G_a6  = torch.mm(fm_a6, fm_a6.t())
            norm_G_a6 = F.normalize(G_a6, p=2, dim=1)
            y_a6 = self.avg_pool(a6).view(b_a6, c_a6)

        #joint
        z=torch.cat((torch.mm(norm_G_a1,y_a1),torch.mm(norm_G_a2,y_a2),torch.mm(norm_G_a3,y_a3),torch.mm(norm_G_a4,y_a4),torch.mm(norm_G_a5,y_a5),torch.mm(norm_G_a6,y_a6)),1)
        z_add=(torch.mm(norm_G_a1,y_a1)+torch.mm(norm_G_a2,y_a2)+torch.mm(norm_G_a3,y_a3)+torch.mm(norm_G_a4,y_a4)+torch.mm(norm_G_a5,y_a5)+torch.mm(norm_G_a6,y_a6))
        z_mul=torch.mm(norm_G_a1,y_a1)*torch.mm(norm_G_a2,y_a2)*torch.mm(norm_G_a3,y_a3)*torch.mm(norm_G_a4,y_a4)*torch.mm(norm_G_a5,y_a5)*torch.mm(norm_G_a6,y_a6)
        #z=F.normalize(z, p=2, dim=1)
        #excitation
        if len(a1.size())<4:
            y_a1 = self.fc_a1(z).view(b_a1, c_a1)
            y_a1_add = self.fc_a1_add(z_add).view(b_a1, c_a1)
            y_a1_mul = self.fc_a1_mul(z_mul).view(b_a1, c_a1)
            y_a2 = self.fc_a2(z).view(b_a2, c_a2)  
            y_a2_add = self.fc_a2_add(z_add).view(b_a2, c_a2)
            y_a2_mul = self.fc_a2_mul(z_mul).view(b_a2, c_a2)
            y_a3 = self.fc_a3(z).view(b_a3, c_a3)
            y_a3_add = self.fc_a3_add(z_add).view(b_a3, c_a3)
            y_a3_mul = self.fc_a3_mul(z_mul).view(b_a3, c_a3)
            y_a4 = self.fc_a4(z).view(b_a4, c_a4)   
            y_a4_add = self.fc_a4_add(z_add).view(b_a4, c_a4)
            y_a4_mul = self.fc_a4_mul(z_mul).view(b_a4, c_a4)
            y_a5 = self.fc_a5(z).view(b_a5, c_a5)
            y_a5_add = self.fc_a5_add(z_add).view(b_a5, c_a5)
            y_a5_mul = self.fc_a5_mul(z_mul).view(b_a5, c_a5)
            y_a6 = self.fc_a6(z).view(b_a6, c_a6)  
            y_a6_add = self.fc_a6_add(z_add).view(b_a6, c_a6)
            y_a6_mul = self.fc_a6_mul(z_mul).view(b_a6, c_a6) 
        else:
            y_a1 = self.fc_a1(z).view(b_a1, c_a1, 1, 1)
            y_a1_add = self.fc_a1_add(z_add).view(b_a1, c_a1, 1, 1)
            y_a1_mul = self.fc_a1_mul(z_mul).view(b_a1, c_a1, 1, 1)
            y_a2 = self.fc_a2(z).view(b_a2, c_a2, 1, 1)  
            y_a2_add = self.fc_a2_add(z_add).view(b_a2, c_a2, 1, 1)
            y_a2_mul = self.fc_a2_mul(z_mul).view(b_a2, c_a2, 1, 1) 
            y_a3 = self.fc_a3(z).view(b_a3, c_a3, 1, 1)
            y_a3_add = self.fc_a3_add(z_add).view(b_a3, c_a3, 1, 1)
            y_a3_mul = self.fc_a3_mul(z_mul).view(b_a3, c_a3, 1, 1)
            y_a4 = self.fc_a4(z).view(b_a4, c_a4, 1, 1)   
            y_a4_add = self.fc_a4_add(z_add).view(b_a4, c_a4, 1, 1)
            y_a4_mul = self.fc_a4_mul(z_mul).view(b_a4, c_a4, 1, 1)
            y_a5 = self.fc_a5(z).view(b_a5, c_a5, 1, 1)
            y_a5_add = self.fc_a5_add(z_add).view(b_a5, c_a5, 1, 1)
            y_a5_mul = self.fc_a5_mul(z_mul).view(b_a5, c_a5, 1, 1)
            y_a6 = self.fc_a6(z).view(b_a6, c_a6, 1, 1)  
            y_a6_add = self.fc_a6_add(z_add).view(b_a6, c_a6, 1, 1)
            y_a6_mul = self.fc_a6_mul(z_mul).view(b_a6, c_a6, 1, 1)      

        return torch.mul(((F.relu(y_a1)+F.relu(y_a1_add)+F.relu(y_a1_mul))).expand_as(a1),a1), \
               torch.mul(((F.relu(y_a2)+F.relu(y_a2_add)+F.relu(y_a2_mul))).expand_as(a2),a2), \
               torch.mul(((F.relu(y_a3)+F.relu(y_a3_add)+F.relu(y_a3_mul))).expand_as(a3),a3), \
               torch.mul(((F.relu(y_a4)+F.relu(y_a4_add)+F.relu(y_a4_mul))).expand_as(a4),a4), \
               torch.mul(((F.relu(y_a5)+F.relu(y_a5_add)+F.relu(y_a5_mul))).expand_as(a5),a5), \
               torch.mul(((F.relu(y_a6)+F.relu(y_a6_add)+F.relu(y_a6_mul))).expand_as(a6),a6)#2*a1*y_a1.expand_as(a1), 2*a2*y_a2.expand_as(a2),2*a3*y_a3.expand_as(a3), 2*a4*y_a4.expand_as(a4),2*a5*y_a5.expand_as(a5), 2*a6*y_a6.expand_as(a6)

def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        #SelfAttention(chann_out),  #self attention
        nn.BatchNorm2d(chann_out),
        #SELayer(chann_out, 16), #squeeze and excitation
        nn.ReLU()
    )
    return layer

def se_conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        #SelfAttention(chann_out),  #self attention
        nn.BatchNorm2d(chann_out),
        SELayer(chann_out, 8), #squeeze and excitation
        nn.ReLU()
    )
    return layer  

def attention_conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        #SelfAttention(chann_out),  #self attention
        nn.BatchNorm2d(chann_out),
        SelfAttention(chann_out), #squeeze and excitation
        nn.ReLU()
    )
    return layer  

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_se_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ se_conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer

class VggNet16(nn.Module):
    output_size = 512
    def __init__(self, n_classes):
        super(VggNet16, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        #self.layer6=SELayer(512, 16)
        self.layer6 = vgg_fc_layer(7*7*512, 256) #4096
        self.layer7 = vgg_fc_layer(256,64)

        # Final layer
        self.layer8 = nn.Linear(64, n_classes)

    def forward(self, x, get_ha=False):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        vgg16_features = self.layer5(out4)
        out4 = vgg16_features.view(out4.size(0), -1)
        out5 = self.layer6(out4)
        out6=self.layer7(out5)
        out7 = self.layer8(out6)
        #out7=F.normalize(out7, p=2, dim=1)
        #out = self.layer8(out)
        if get_ha:
            return out2,out3,out4,out5,out6,out7

        return out7

class SeVggNet16(nn.Module):
    output_size =512
    def __init__(self, n_classes):
        super(SeVggNet16, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_se_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_se_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_se_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_se_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_se_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        #self.layer6=SELayer(512, 16)
        self.layer6 = vgg_fc_layer(7*7*512, 256) #4096
        self.layer7 = vgg_fc_layer(256,64)

        # Final layer
        self.layer8 = nn.Linear(64, n_classes)

    def forward(self, x, get_ha=False):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        vgg16_features = self.layer5(out4)
        out4 = vgg16_features.view(out4.size(0), -1)
        out5 = self.layer6(out4)
        out6=self.layer7(out5)
        out7 = self.layer8(out6)
        #out = self.layer8(out)
        if get_ha:
            return out1,out2,out3,out4,out5,out6,out7

        return out7

class SeFusionVGG16(nn.Module):
    def __init__(self, n_classes=1000):
        super(SeFusionVGG16, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.s_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.s_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.s_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.s_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.s_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        self.s_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.s_layer7 = nn.Linear(300,300)

        # Final layer
        self.s_layer8 = nn.Linear(300, n_classes)

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.t_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.t_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.t_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.t_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.t_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        self.t_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.t_layer7 = nn.Linear(300,300)

        # Final layer
        self.t_layer8 = nn.Linear(300, n_classes)
        self.fusion1=SE_Fusion(64,64,4)
        self.fusion2=SE_Fusion(128,128,4)
        self.fusion3=SE_Fusion(256,256,4)
        self.fusion4=SE_Fusion(512,512,4)
        self.fusion5=SE_Fusion(512,512,4)
        self.fusion6=SE_Fusion(256,256,4)
        self.fusion7=SE_Fusion(64,64,4)
        self.fusion8=SE_Fusion(n_classes,n_classes,4)
    def forward(self, x, y, get_ha=False):
        s_out1 = self.s_layer1(x)
        t_out1 = self.t_layer1(y)
        #s_out1,t_out1=self.fusion1(s_out1,t_out1)

        #visualize 
        #feature=s_out[1,0,:,:]
        #feature=feature.view(feature.shape[0],feature.shape[1])
        #feature=feature.cpu().data.numpy()
        #feature= 1.0/(1+np.exp(-1*feature))
        #feature=np.round(feature*255)
        #cv2.imwrite('s_img.jpg',feature)

        s_out2 = self.s_layer2(s_out1)
        t_out2 = self.t_layer2(t_out1)
        #s_out2,t_out2=self.fusion2(s_out2,t_out2)

        s_out3 = self.s_layer3(s_out2)
        t_out3 = self.t_layer3(t_out2)
        #s_out3,t_out3=self.fusion3(s_out3,t_out3)

        s_out4 = self.s_layer4(s_out3)
        t_out4 = self.t_layer4(t_out3)
        #s_out4,t_out4=self.fusion4(s_out4,t_out4)

        s_out5 = self.s_layer5(s_out4)
        t_out5 = self.t_layer5(t_out4)
        #s_out5,t_out5=self.fusion5(s_out5,t_out5)


        s_out5_temp = s_out5.view(s_out5.size(0), -1)
        t_out5_temp = t_out5.view(t_out5.size(0), -1)

        s_out6 = self.s_layer6(s_out5_temp)
        t_out6 = self.t_layer6(t_out5_temp)
        #s_out6,t_out6 = self.fusion6(s_out6,t_out6)

        s_out7 = self.s_layer7(s_out6)
        t_out7 = self.t_layer7(t_out6)
        #s_out7,t_out7 = self.fusion7(s_out7,t_out7)

        s_out8 = self.s_layer8(s_out7)
        t_out8 = self.t_layer8(t_out7)
        #s_out8,t_out8 = self.fusion8(s_out8,t_out8)     
        if get_ha:

            return s_out1,t_out1,s_out2,t_out2,s_out3,t_out3,s_out4,t_out4,s_out5,t_out5,s_out6,t_out6,s_out7,t_out7,s_out8,t_out8

        return s_out8,t_out8    

class SeFusionVGG16_MMAct(nn.Module):
    def __init__(self, n_classes=1000):
        super(SeFusionVGG16_MMAct, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.ap_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.ap_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.ap_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.ap_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.ap_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.ap_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.ap_layer7 = nn.Linear(300,300)
        # Final layer
        self.ap_layer8 = nn.Linear(300, n_classes)

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.aw_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.aw_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.aw_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.aw_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.aw_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.aw_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.aw_layer7 = nn.Linear(300,300)
        # Final layer
        self.aw_layer8 = nn.Linear(300, n_classes)

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.gy_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.gy_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.gy_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.gy_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.gy_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.gy_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.gy_layer7 = nn.Linear(300,300)
        # Final layer
        self.gy_layer8 = nn.Linear(300, n_classes)

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.ori_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.ori_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.ori_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.ori_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.ori_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.ori_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.ori_layer7 = nn.Linear(300,300)
        # Final layer
        self.ori_layer8 = nn.Linear(300, n_classes)

        self.fusion1=SE_semantic_Fusion_MMAct(64,64,64,64,8)
        self.fusion2=SE_semantic_Fusion_MMAct(128,128,128,128,8)
        self.fusion3=SE_semantic_Fusion_MMAct(256,256,256,256,8)
        self.fusion4=SE_semantic_Fusion_MMAct(512,512,512,512,8)
        self.fusion5=SE_semantic_Fusion_MMAct(512,512,512,512,8)
        self.fusion6=SE_semantic_Fusion_MMAct(300,300,300,300,8)
        self.fusion7=SE_semantic_Fusion_MMAct(300,300,300,300,8)
        #self.fusion8=SE_semantic_Fusion_MMAct(n_classes,n_classes,n_classes,4)
    def forward(self, ap, aw, gy, ori, get_ha=False):
        ap_out1 = self.ap_layer1(ap)
        aw_out1 = self.aw_layer1(aw)
        gy_out1 = self.gy_layer1(gy)
        ori_out1 = self.ori_layer1(ori)
        #ap_out1, aw_out1,gy_out1, ori_out1=self.fusion1(ap_out1,aw_out1,gy_out1,ori_out1)

        #visualize 
        #feature=s_out[1,0,:,:]
        #feature=feature.view(feature.shape[0],feature.shape[1])
        #feature=feature.cpu().data.numpy()
        #feature= 1.0/(1+np.exp(-1*feature))
        #feature=np.round(feature*255)
        #cv2.imwrite('s_img.jpg',feature)

        ap_out2 = self.ap_layer2(ap_out1)
        aw_out2 = self.aw_layer2(aw_out1)
        gy_out2 = self.gy_layer2(gy_out1)
        ori_out2 = self.ori_layer2(ori_out1)
        #ap_out2,aw_out2,gy_out2,ori_out2=self.fusion2(ap_out2,aw_out2,gy_out2,ori_out2)

        ap_out3 = self.ap_layer3(ap_out2)
        aw_out3 = self.aw_layer3(aw_out2)
        gy_out3 = self.gy_layer3(gy_out2)
        ori_out3 = self.ori_layer3(ori_out2)
        #ap_out3,aw_out3,gy_out3,ori_out3=self.fusion3(ap_out3,aw_out3,gy_out3,ori_out3)

        ap_out4 = self.ap_layer4(ap_out3)
        aw_out4 = self.aw_layer4(aw_out3)
        gy_out4 = self.gy_layer4(gy_out3)
        ori_out4 = self.ori_layer4(ori_out3)
        #ap_out4,aw_out4,gy_out4,ori_out4=self.fusion4(ap_out4,aw_out4,gy_out4,ori_out4)

        ap_out5 = self.ap_layer5(ap_out4)
        aw_out5 = self.aw_layer5(aw_out4)
        gy_out5 = self.gy_layer5(gy_out4)
        ori_out5 = self.ori_layer5(ori_out4)
        #ap_out5,aw_out5,gy_out5,ori_out5=self.fusion5(ap_out5,aw_out5,gy_out5,ori_out5)


        ap_out5_temp = ap_out5.view(ap_out5.size(0), -1)
        aw_out5_temp = aw_out5.view(aw_out5.size(0), -1)
        gy_out5_temp = gy_out5.view(gy_out5.size(0), -1)
        ori_out5_temp = ori_out5.view(ori_out5.size(0), -1)

        ap_out6 = self.ap_layer6(ap_out5_temp)
        aw_out6 = self.aw_layer6(aw_out5_temp)
        gy_out6 = self.gy_layer6(gy_out5_temp)
        ori_out6 = self.ori_layer6(ori_out5_temp)
        #ap_out6,aw_out6,gy_out6,ori_out6 = self.fusion6(ap_out6,aw_out6,gy_out6,ori_out6)


        ap_out7 = self.ap_layer7(ap_out6)
        aw_out7 = self.aw_layer7(aw_out6)
        gy_out7 = self.gy_layer7(gy_out6)
        ori_out7 = self.ori_layer7(ori_out6)
        #ap_out7,aw_out7,gy_out7,ori_out7 = self.fusion7(ap_out7,aw_out7,gy_out7,ori_out7)


        ap_out8 = self.ap_layer8(ap_out7)
        aw_out8 = self.aw_layer8(aw_out7)
        gy_out8 = self.gy_layer8(gy_out7)
        ori_out8 = self.ori_layer8(ori_out7)
        #s_out7,t_out7 = self.fusion8(s_out7,t_out7)

        if get_ha:
            return ap_out1, aw_out1, gy_out1, ori_out1,\
            ap_out2, aw_out2, gy_out2, ori_out2,\
            ap_out3, aw_out3, gy_out3, ori_out3,\
            ap_out4, aw_out4, gy_out4, ori_out4,\
            ap_out5, aw_out5, gy_out5, ori_out5,\
            ap_out6, aw_out6, gy_out6,ori_out6,\
            ap_out7, aw_out7, gy_out7, ori_out7,\
            ap_out8, aw_out8, gy_out8, ori_out8

        return ap_out8, aw_out8, gy_out8, ori_out8    

class SeFusionVGG16_Berkeley(nn.Module):
    def __init__(self, n_classes=1000):
        super(SeFusionVGG16_Berkeley, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.a1_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a1_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a1_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a1_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a1_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.a1_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.a1_layer7 = nn.Linear(300,300)
        # Final layer
        self.a1_layer8 = nn.Linear(300, n_classes)

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.a2_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a2_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a2_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a2_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a2_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.a2_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.a2_layer7 = nn.Linear(300,300)
        # Final layer
        self.a2_layer8 = nn.Linear(300, n_classes)

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.a3_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a3_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a3_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a3_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a3_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.a3_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.a3_layer7 = nn.Linear(300,300)
        # Final layer
        self.a3_layer8 = nn.Linear(300, n_classes)

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.a4_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a4_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a4_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a4_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a4_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.a4_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.a4_layer7 = nn.Linear(300,300)
        # Final layer
        self.a4_layer8 = nn.Linear(300, n_classes)

        self.a5_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a5_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a5_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a5_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a5_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.a5_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.a5_layer7 = nn.Linear(300,300)
        # Final layer
        self.a5_layer8 = nn.Linear(300, n_classes)   

        self.a6_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a6_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a6_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a6_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a6_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.a6_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.a6_layer7 = nn.Linear(300,300)
        # Final layer
        self.a6_layer8 = nn.Linear(300, n_classes)       

        self.fusion1=SE_semantic_Fusion_Berkeley(64,64,64,64,64,64,12)
        self.fusion2=SE_semantic_Fusion_Berkeley(128,128,128,128,128,128,12)
        self.fusion3=SE_semantic_Fusion_Berkeley(256,256,256,256,256,256,12)
        self.fusion4=SE_semantic_Fusion_Berkeley(512,512,512,512,512,512,12)
        self.fusion5=SE_semantic_Fusion_Berkeley(512,512,512,512,512,512,12)
        self.fusion6=SE_semantic_Fusion_Berkeley(300,300,300,300,300,300,12)
        self.fusion7=SE_semantic_Fusion_Berkeley(300,300,300,300,300,300,12)
        #self.fusion8=SE_semantic_Fusion_MMAct(n_classes,n_classes,n_classes,4)
    def forward(self, a1, a2, a3, a4, a5, a6, get_ha=False):
        a1_out1 = self.a1_layer1(a1)
        a2_out1 = self.a2_layer1(a2)
        a3_out1 = self.a3_layer1(a3)
        a4_out1 = self.a4_layer1(a4)
        a5_out1 = self.a3_layer1(a5)
        a6_out1 = self.a4_layer1(a6)
        #a1_out1, a2_out1, a3_out1, a4_out1, a5_out1, a6_out1=self.fusion1(a1_out1,a2_out1,a3_out1,a4_out1,a5_out1,a6_out1)

        #visualize 
        #feature=s_out[1,0,:,:]
        #feature=feature.view(feature.shape[0],feature.shape[1])
        #feature=feature.cpu().data.numpy()
        #feature= 1.0/(1+np.exp(-1*feature))
        #feature=np.round(feature*255)
        #cv2.imwrite('s_img.jpg',feature)

        a1_out2 = self.a1_layer2(a1_out1)
        a2_out2 = self.a2_layer2(a2_out1)
        a3_out2 = self.a3_layer2(a3_out1)
        a4_out2 = self.a4_layer2(a4_out1)
        a5_out2 = self.a5_layer2(a5_out1)
        a6_out2 = self.a6_layer2(a6_out1)
        #a1_out2, a2_out2, a3_out2, a4_out2, a5_out2, a6_out2=self.fusion2(a1_out2,a2_out2,a3_out2,a4_out2,a5_out2,a6_out2)

        a1_out3 = self.a1_layer3(a1_out2)
        a2_out3 = self.a2_layer3(a2_out2)
        a3_out3 = self.a3_layer3(a3_out2)
        a4_out3 = self.a4_layer3(a4_out2)
        a5_out3 = self.a5_layer3(a5_out2)
        a6_out3 = self.a6_layer3(a6_out2)
        #a1_out3, a2_out3, a3_out3, a4_out3, a5_out3, a6_out3=self.fusion3(a1_out3,a2_out3,a3_out3,a4_out3,a5_out3,a6_out3)

        a1_out4 = self.a1_layer4(a1_out3)
        a2_out4 = self.a2_layer4(a2_out3)
        a3_out4 = self.a3_layer4(a3_out3)
        a4_out4 = self.a4_layer4(a4_out3)
        a5_out4 = self.a5_layer4(a5_out3)
        a6_out4 = self.a6_layer4(a6_out3)
        #a1_out4, a2_out4, a3_out4, a4_out4, a5_out4, a6_out4=self.fusion4(a1_out4,a2_out4,a3_out4,a4_out4,a5_out4,a6_out4)

        a1_out5 = self.a1_layer5(a1_out4)
        a2_out5 = self.a2_layer5(a2_out4)
        a3_out5 = self.a3_layer5(a3_out4)
        a4_out5 = self.a4_layer5(a4_out4)
        a5_out5 = self.a5_layer5(a5_out4)
        a6_out5 = self.a6_layer5(a6_out4)
        #a1_out5, a2_out5, a3_out5, a4_out5, a5_out5, a6_out5=self.fusion5(a1_out5,a2_out5,a3_out5,a4_out5,a5_out5,a6_out5)


        a1_out5_temp = a1_out5.view(a1_out5.size(0), -1)
        a2_out5_temp = a2_out5.view(a2_out5.size(0), -1)
        a3_out5_temp = a3_out5.view(a3_out5.size(0), -1)
        a4_out5_temp = a4_out5.view(a4_out5.size(0), -1)
        a5_out5_temp = a5_out5.view(a5_out5.size(0), -1)
        a6_out5_temp = a6_out5.view(a6_out5.size(0), -1)

        a1_out6 = self.a1_layer6(a1_out5_temp)
        a2_out6 = self.a2_layer6(a2_out5_temp)
        a3_out6 = self.a3_layer6(a3_out5_temp)
        a4_out6 = self.a4_layer6(a4_out5_temp)
        a5_out6 = self.a5_layer6(a5_out5_temp)
        a6_out6 = self.a6_layer6(a6_out5_temp)
        #a1_out6,a2_out6,a3_out6,a4_out6,a5_out6,a6_out6 = self.fusion6(a1_out6,a2_out6,a3_out6,a4_out6,a5_out6,a6_out6)


        a1_out7 = self.a1_layer7(a1_out6)
        a2_out7 = self.a2_layer7(a2_out6)
        a3_out7 = self.a3_layer7(a3_out6)
        a4_out7 = self.a4_layer7(a4_out6)
        a5_out7 = self.a5_layer7(a5_out6)
        a6_out7 = self.a6_layer7(a6_out6)
        #a1_out7,a2_out7,a3_out7,a4_out7,a5_out7,a5_out7 = self.fusion7(a1_out7,a2_out7,a3_out7,a4_out7,a5_out7,a6_out7)


        a1_out8 = self.a1_layer8(a1_out7)
        a2_out8 = self.a2_layer8(a2_out7)
        a3_out8 = self.a3_layer8(a3_out7)
        a4_out8 = self.a4_layer8(a4_out7)
        a5_out8 = self.a5_layer8(a5_out7)
        a6_out8 = self.a6_layer8(a6_out7)
        #s_out7,t_out7 = self.fusion8(s_out7,t_out7)

        if get_ha:
            return a1_out1, a2_out1, a3_out1, a4_out1,a5_out1, a6_out1,\
            a1_out2, a2_out2, a3_out2, a4_out2, a5_out2, a6_out2,\
            a1_out3, a2_out3, a3_out3, a4_out3, a5_out3, a6_out3,\
            a1_out4, a2_out4, a3_out4, a4_out4, a5_out4, a6_out4,\
            a1_out5, a2_out5, a3_out5, a4_out5, a5_out5, a6_out5,\
            a1_out6, a2_out6, a3_out6, a4_out6, a5_out6, a6_out6,\
            a1_out7, a2_out7, a3_out7, a4_out7, a5_out7, a6_out7,\
            a1_out8, a2_out8, a3_out8, a4_out8, a5_out8, a6_out8

        return a1_out8, a2_out8, a3_out8, a4_out8, a5_out8, a6_out8   

class SemanticFusionVGG16(nn.Module):
    def __init__(self, n_classes=1000):
        super(SemanticFusionVGG16, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.s_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.s_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.s_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.s_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.s_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        self.s_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.s_layer7 = nn.Linear(300,300)

        # Final layer
        self.s_layer8 = nn.Linear(300, n_classes)

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.t_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.t_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.t_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.t_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.t_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        self.t_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.t_layer7 = nn.Linear(300,300)

        # Final layer
        self.t_layer8 = nn.Linear(300, n_classes)
        self.fusion1=SE_semantic_Fusion(64,64,4)
        self.fusion2=SE_semantic_Fusion(128,128,4)
        self.fusion3=SE_semantic_Fusion(256,256,4)
        self.fusion4=SE_semantic_Fusion(512,512,4)
        self.fusion5=SE_semantic_Fusion(512,512,4)
        self.fusion6=SE_semantic_Fusion(300,300,4)
        self.fusion7=SE_semantic_Fusion(300,300,4)
        self.fusion8=SE_semantic_Fusion(n_classes,n_classes,4)

        #self.fusion1=SE_semantic_Fusion_batch(8,8,2)
        #self.fusion2=SE_semantic_Fusion_batch(8,8,2)
        #self.fusion3=SE_semantic_Fusion_batch(8,8,2)
        #self.fusion4=SE_semantic_Fusion_batch(8,8,2)
        #self.fusion5=SE_semantic_Fusion_batch(8,8,2)
        #self.fusion6=SE_semantic_Fusion_batch(8,8,2)
        #self.fusion7=SE_semantic_Fusion_batch(8,8,2)
        #self.fusion8=SE_semantic_Fusion_batch(8,8,2)
    def forward(self, x, y,get_ha=False):
        s_out1 = self.s_layer1(x)
        t_out1 = self.t_layer1(y)
        s_out1,t_out1=self.fusion1(s_out1,t_out1)

        #visualize 
        #feature=s_out[1,0,:,:]
        #feature=feature.view(feature.shape[0],feature.shape[1])
        #feature=feature.cpu().data.numpy()
        #feature= 1.0/(1+np.exp(-1*feature))
        #feature=np.round(feature*255)
        #cv2.imwrite('s_img.jpg',feature)

        s_out2 = self.s_layer2(s_out1)
        t_out2 = self.t_layer2(t_out1)
        s_out2,t_out2=self.fusion2(s_out2,t_out2)

        s_out3 = self.s_layer3(s_out2)
        t_out3 = self.t_layer3(t_out2)
        s_out3,t_out3=self.fusion3(s_out3,t_out3)

        s_out4 = self.s_layer4(s_out3)
        t_out4 = self.t_layer4(t_out3)
        s_out4,t_out4=self.fusion4(s_out4,t_out4)

        s_out5 = self.s_layer5(s_out4)
        t_out5 = self.t_layer5(t_out4)
        s_out5,t_out5=self.fusion5(s_out5,t_out5)


        s_out5_temp = s_out5.view(s_out5.size(0), -1)
        t_out5_temp = t_out5.view(t_out5.size(0), -1)

        s_out6 = self.s_layer6(s_out5_temp)
        t_out6 = self.t_layer6(t_out5_temp)
        s_out6,t_out6 = self.fusion6(s_out6,t_out6)

        s_out7 = self.s_layer7(s_out6)
        t_out7 = self.t_layer7(t_out6)
        s_out7,t_out7 = self.fusion7(s_out7,t_out7)

        s_out8 = self.s_layer8(s_out7)
        t_out8 = self.t_layer8(t_out7)
        #s_out8,t_out8 = self.fusion8(s_out8,t_out8)

        if get_ha:

            return s_out1,t_out1,s_out2,t_out2,s_out3,t_out3,s_out4,t_out4,s_out5,t_out5,s_out6,t_out6,s_out7,t_out7,s_out8,t_out8

        return s_out8,t_out8       

class SemanticFusionVGG16_MMAct(nn.Module):
    def __init__(self, n_classes=1000):
        super(SemanticFusionVGG16_MMAct, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.ap_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.ap_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.ap_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.ap_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.ap_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.ap_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.ap_layer7 = nn.Linear(300,300)
        # Final layer
        self.ap_layer8 = nn.Linear(300, n_classes)

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.aw_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.aw_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.aw_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.aw_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.aw_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.aw_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.aw_layer7 = nn.Linear(300,300)
        # Final layer
        self.aw_layer8 = nn.Linear(300, n_classes)

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.gy_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.gy_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.gy_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.gy_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.gy_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.gy_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.gy_layer7 = nn.Linear(300,300)
        # Final layer
        self.gy_layer8 = nn.Linear(300, n_classes)

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.ori_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.ori_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.ori_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.ori_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.ori_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.ori_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.ori_layer7 = nn.Linear(300,300)
        # Final layer
        self.ori_layer8 = nn.Linear(300, n_classes)

        self.fusion1=SE_semantic_Fusion_MMAct(64,64,64,64,8)
        self.fusion2=SE_semantic_Fusion_MMAct(128,128,128,128,8)
        self.fusion3=SE_semantic_Fusion_MMAct(256,256,256,256,8)
        self.fusion4=SE_semantic_Fusion_MMAct(512,512,512,512,8)
        self.fusion5=SE_semantic_Fusion_MMAct(512,512,512,512,8)
        self.fusion6=SE_semantic_Fusion_MMAct(300,300,300,300,8)
        self.fusion7=SE_semantic_Fusion_MMAct(300,300,300,300,8)
        #self.fusion8=SE_semantic_Fusion_MMAct(n_classes,n_classes,n_classes,4)
    def forward(self, ap, aw, gy, ori, get_ha=False):
        ap_out1 = self.ap_layer1(ap)
        aw_out1 = self.aw_layer1(aw)
        gy_out1 = self.gy_layer1(gy)
        ori_out1 = self.ori_layer1(ori)
        ap_out1, aw_out1,gy_out1, ori_out1=self.fusion1(ap_out1,aw_out1,gy_out1,ori_out1)

        #visualize 
        #feature=s_out[1,0,:,:]
        #feature=feature.view(feature.shape[0],feature.shape[1])
        #feature=feature.cpu().data.numpy()
        #feature= 1.0/(1+np.exp(-1*feature))
        #feature=np.round(feature*255)
        #cv2.imwrite('s_img.jpg',feature)

        ap_out2 = self.ap_layer2(ap_out1)
        aw_out2 = self.aw_layer2(aw_out1)
        gy_out2 = self.gy_layer2(gy_out1)
        ori_out2 = self.ori_layer2(ori_out1)
        ap_out2,aw_out2,gy_out2,ori_out2=self.fusion2(ap_out2,aw_out2,gy_out2,ori_out2)

        ap_out3 = self.ap_layer3(ap_out2)
        aw_out3 = self.aw_layer3(aw_out2)
        gy_out3 = self.gy_layer3(gy_out2)
        ori_out3 = self.ori_layer3(ori_out2)
        ap_out3,aw_out3,gy_out3,ori_out3=self.fusion3(ap_out3,aw_out3,gy_out3,ori_out3)

        ap_out4 = self.ap_layer4(ap_out3)
        aw_out4 = self.aw_layer4(aw_out3)
        gy_out4 = self.gy_layer4(gy_out3)
        ori_out4 = self.ori_layer4(ori_out3)
        ap_out4,aw_out4,gy_out4,ori_out4=self.fusion4(ap_out4,aw_out4,gy_out4,ori_out4)

        ap_out5 = self.ap_layer5(ap_out4)
        aw_out5 = self.aw_layer5(aw_out4)
        gy_out5 = self.gy_layer5(gy_out4)
        ori_out5 = self.ori_layer5(ori_out4)
        ap_out5,aw_out5,gy_out5,ori_out5=self.fusion5(ap_out5,aw_out5,gy_out5,ori_out5)


        ap_out5_temp = ap_out5.view(ap_out5.size(0), -1)
        aw_out5_temp = aw_out5.view(aw_out5.size(0), -1)
        gy_out5_temp = gy_out5.view(gy_out5.size(0), -1)
        ori_out5_temp = ori_out5.view(ori_out5.size(0), -1)

        ap_out6 = self.ap_layer6(ap_out5_temp)
        aw_out6 = self.aw_layer6(aw_out5_temp)
        gy_out6 = self.gy_layer6(gy_out5_temp)
        ori_out6 = self.ori_layer6(ori_out5_temp)
        ap_out6,aw_out6,gy_out6,ori_out6 = self.fusion6(ap_out6,aw_out6,gy_out6,ori_out6)


        ap_out7 = self.ap_layer7(ap_out6)
        aw_out7 = self.aw_layer7(aw_out6)
        gy_out7 = self.gy_layer7(gy_out6)
        ori_out7 = self.ori_layer7(ori_out6)
        ap_out7,aw_out7,gy_out7,ori_out7 = self.fusion7(ap_out7,aw_out7,gy_out7,ori_out7)


        ap_out8 = self.ap_layer8(ap_out7)
        aw_out8 = self.aw_layer8(aw_out7)
        gy_out8 = self.gy_layer8(gy_out7)
        ori_out8 = self.ori_layer8(ori_out7)
        #s_out7,t_out7 = self.fusion8(s_out7,t_out7)

        if get_ha:
            return ap_out1, aw_out1, gy_out1, ori_out1,\
            ap_out2, aw_out2, gy_out2, ori_out2,\
            ap_out3, aw_out3, gy_out3, ori_out3,\
            ap_out4, aw_out4, gy_out4, ori_out4,\
            ap_out5, aw_out5, gy_out5, ori_out5,\
            ap_out6, aw_out6, gy_out6,ori_out6,\
            ap_out7, aw_out7, gy_out7, ori_out7,\
            ap_out8, aw_out8, gy_out8, ori_out8

        return ap_out8, aw_out8, gy_out8, ori_out8                 


class SemanticFusionVGG16_Berkeley(nn.Module):
    def __init__(self, n_classes=1000):
        super(SemanticFusionVGG16_Berkeley, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.a1_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a1_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a1_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a1_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a1_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.a1_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.a1_layer7 = nn.Linear(300,300)
        # Final layer
        self.a1_layer8 = nn.Linear(300, n_classes)

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.a2_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a2_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a2_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a2_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a2_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.a2_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.a2_layer7 = nn.Linear(300,300)
        # Final layer
        self.a2_layer8 = nn.Linear(300, n_classes)

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.a3_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a3_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a3_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a3_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a3_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.a3_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.a3_layer7 = nn.Linear(300,300)
        # Final layer
        self.a3_layer8 = nn.Linear(300, n_classes)

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.a4_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a4_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a4_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a4_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a4_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.a4_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.a4_layer7 = nn.Linear(300,300)
        # Final layer
        self.a4_layer8 = nn.Linear(300, n_classes)

        self.a5_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a5_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a5_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a5_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a5_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.a5_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.a5_layer7 = nn.Linear(300,300)
        # Final layer
        self.a5_layer8 = nn.Linear(300, n_classes)   

        self.a6_layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.a6_layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.a6_layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.a6_layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.a6_layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # FC layers
        self.a6_layer6 = vgg_fc_layer(7*7*512, 300) #4096
        self.a6_layer7 = nn.Linear(300,300)
        # Final layer
        self.a6_layer8 = nn.Linear(300, n_classes)       

        self.fusion1=SE_semantic_Fusion_Berkeley(64,64,64,64,64,64,12)
        self.fusion2=SE_semantic_Fusion_Berkeley(128,128,128,128,128,128,12)
        self.fusion3=SE_semantic_Fusion_Berkeley(256,256,256,256,256,256,12)
        self.fusion4=SE_semantic_Fusion_Berkeley(512,512,512,512,512,512,12)
        self.fusion5=SE_semantic_Fusion_Berkeley(512,512,512,512,512,512,12)
        self.fusion6=SE_semantic_Fusion_Berkeley(300,300,300,300,300,300,12)
        self.fusion7=SE_semantic_Fusion_Berkeley(300,300,300,300,300,300,12)
        #self.fusion8=SE_semantic_Fusion_MMAct(n_classes,n_classes,n_classes,4)
    def forward(self, a1, a2, a3, a4, a5, a6, get_ha=False):
        a1_out1 = self.a1_layer1(a1)
        a2_out1 = self.a2_layer1(a2)
        a3_out1 = self.a3_layer1(a3)
        a4_out1 = self.a4_layer1(a4)
        a5_out1 = self.a3_layer1(a5)
        a6_out1 = self.a4_layer1(a6)
        a1_out1, a2_out1, a3_out1, a4_out1, a5_out1, a6_out1=self.fusion1(a1_out1,a2_out1,a3_out1,a4_out1,a5_out1,a6_out1)

        #visualize 
        #feature=s_out[1,0,:,:]
        #feature=feature.view(feature.shape[0],feature.shape[1])
        #feature=feature.cpu().data.numpy()
        #feature= 1.0/(1+np.exp(-1*feature))
        #feature=np.round(feature*255)
        #cv2.imwrite('s_img.jpg',feature)

        a1_out2 = self.a1_layer2(a1_out1)
        a2_out2 = self.a2_layer2(a2_out1)
        a3_out2 = self.a3_layer2(a3_out1)
        a4_out2 = self.a4_layer2(a4_out1)
        a5_out2 = self.a5_layer2(a5_out1)
        a6_out2 = self.a6_layer2(a6_out1)
        a1_out2, a2_out2, a3_out2, a4_out2, a5_out2, a6_out2=self.fusion2(a1_out2,a2_out2,a3_out2,a4_out2,a5_out2,a6_out2)

        a1_out3 = self.a1_layer3(a1_out2)
        a2_out3 = self.a2_layer3(a2_out2)
        a3_out3 = self.a3_layer3(a3_out2)
        a4_out3 = self.a4_layer3(a4_out2)
        a5_out3 = self.a5_layer3(a5_out2)
        a6_out3 = self.a6_layer3(a6_out2)
        a1_out3, a2_out3, a3_out3, a4_out3, a5_out3, a6_out3=self.fusion3(a1_out3,a2_out3,a3_out3,a4_out3,a5_out3,a6_out3)

        a1_out4 = self.a1_layer4(a1_out3)
        a2_out4 = self.a2_layer4(a2_out3)
        a3_out4 = self.a3_layer4(a3_out3)
        a4_out4 = self.a4_layer4(a4_out3)
        a5_out4 = self.a5_layer4(a5_out3)
        a6_out4 = self.a6_layer4(a6_out3)
        a1_out4, a2_out4, a3_out4, a4_out4, a5_out4, a6_out4=self.fusion4(a1_out4,a2_out4,a3_out4,a4_out4,a5_out4,a6_out4)

        a1_out5 = self.a1_layer5(a1_out4)
        a2_out5 = self.a2_layer5(a2_out4)
        a3_out5 = self.a3_layer5(a3_out4)
        a4_out5 = self.a4_layer5(a4_out4)
        a5_out5 = self.a5_layer5(a5_out4)
        a6_out5 = self.a6_layer5(a6_out4)
        a1_out5, a2_out5, a3_out5, a4_out5, a5_out5, a6_out5=self.fusion5(a1_out5,a2_out5,a3_out5,a4_out5,a5_out5,a6_out5)


        a1_out5_temp = a1_out5.view(a1_out5.size(0), -1)
        a2_out5_temp = a2_out5.view(a2_out5.size(0), -1)
        a3_out5_temp = a3_out5.view(a3_out5.size(0), -1)
        a4_out5_temp = a4_out5.view(a4_out5.size(0), -1)
        a5_out5_temp = a5_out5.view(a5_out5.size(0), -1)
        a6_out5_temp = a6_out5.view(a6_out5.size(0), -1)

        a1_out6 = self.a1_layer6(a1_out5_temp)
        a2_out6 = self.a2_layer6(a2_out5_temp)
        a3_out6 = self.a3_layer6(a3_out5_temp)
        a4_out6 = self.a4_layer6(a4_out5_temp)
        a5_out6 = self.a5_layer6(a5_out5_temp)
        a6_out6 = self.a6_layer6(a6_out5_temp)
        a1_out6,a2_out6,a3_out6,a4_out6,a5_out6,a6_out6 = self.fusion6(a1_out6,a2_out6,a3_out6,a4_out6,a5_out6,a6_out6)


        a1_out7 = self.a1_layer7(a1_out6)
        a2_out7 = self.a2_layer7(a2_out6)
        a3_out7 = self.a3_layer7(a3_out6)
        a4_out7 = self.a4_layer7(a4_out6)
        a5_out7 = self.a5_layer7(a5_out6)
        a6_out7 = self.a6_layer7(a6_out6)
        a1_out7,a2_out7,a3_out7,a4_out7,a5_out7,a5_out7 = self.fusion7(a1_out7,a2_out7,a3_out7,a4_out7,a5_out7,a6_out7)


        a1_out8 = self.a1_layer8(a1_out7)
        a2_out8 = self.a2_layer8(a2_out7)
        a3_out8 = self.a3_layer8(a3_out7)
        a4_out8 = self.a4_layer8(a4_out7)
        a5_out8 = self.a5_layer8(a5_out7)
        a6_out8 = self.a6_layer8(a6_out7)
        #s_out7,t_out7 = self.fusion8(s_out7,t_out7)

        if get_ha:
            return a1_out1, a2_out1, a3_out1, a4_out1,a5_out1, a6_out1,\
            a1_out2, a2_out2, a3_out2, a4_out2, a5_out2, a6_out2,\
            a1_out3, a2_out3, a3_out3, a4_out3, a5_out3, a6_out3,\
            a1_out4, a2_out4, a3_out4, a4_out4, a5_out4, a6_out4,\
            a1_out5, a2_out5, a3_out5, a4_out5, a5_out5, a6_out5,\
            a1_out6, a2_out6, a3_out6, a4_out6, a5_out6, a6_out6,\
            a1_out7, a2_out7, a3_out7, a4_out7, a5_out7, a6_out7,\
            a1_out8, a2_out8, a3_out8, a4_out8, a5_out8, a6_out8

        return a1_out8, a2_out8, a3_out8, a4_out8, a5_out8, a6_out8                                                             