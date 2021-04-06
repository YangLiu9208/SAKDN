import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from metric.utils import pdist

__all__ = ['L1Triplet', 'L2Triplet', 'ContrastiveLoss', 'RkdDistance', 'RKdAngle', 'HardDarkRank','SP', 'SE_Fusion', 'SoftTarget','Gram_loss','Hint', 'AT', 'RKD', 'CC', 'IRG', 'OFD', 'AFD', 'Logits']


class _Triplet(nn.Module):
    def __init__(self, p=2, margin=0.2, sampler=None, reduce=True, size_average=True):
        super().__init__()
        self.p = p
        self.margin = margin

        # update distance function accordingly
        self.sampler = sampler
        self.sampler.dist_func = lambda e: pdist(e, squared=(p==2))

        self.reduce = reduce
        self.size_average = size_average

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        loss = F.triplet_margin_loss(anchor_embed, positive_embed, negative_embed,
                                     margin=self.margin, p=self.p, reduction='none')

        if not self.reduce:
            return loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class L2Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=2, margin=margin, sampler=sampler)


class L1Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=1, margin=margin, sampler=sampler)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__()
        self.margin = margin
        self.sampler = sampler

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        pos_loss = (F.pairwise_distance(anchor_embed, positive_embed, p=2)).pow(2)
        neg_loss = (self.margin - F.pairwise_distance(anchor_embed, negative_embed, p=2)).clamp(min=0).pow(2)

        loss = torch.cat((pos_loss, neg_loss))
        return loss.mean()


class HardDarkRank(nn.Module):
    def __init__(self, alpha=3, beta=3, permute_len=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len

    def forward(self, student, teacher):
        score_teacher = -1 * self.alpha * pdist(teacher, squared=False).pow(self.beta)
        score_student = -1 * self.alpha * pdist(student, squared=False).pow(self.beta)

        permute_idx = score_teacher.sort(dim=1, descending=True)[1][:, 1:(self.permute_len+1)]
        ordered_student = torch.gather(score_student, 1, permute_idx)

        log_prob = (ordered_student - torch.stack([torch.logsumexp(ordered_student[:, i:], dim=1) for i in range(permute_idx.size(1))], dim=1)).sum(dim=1)
        loss = (-1 * log_prob).mean()

        return loss


class FitNet(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.transform = nn.Conv2d(in_feature, out_feature, 1, bias=False)
        self.transform.weight.data.uniform_(-0.005, 0.005)

    def forward(self, student, teacher):
        if student.dim() == 2:
            student = student.unsqueeze(2).unsqueeze(3)
            teacher = teacher.unsqueeze(2).unsqueeze(3)

        return (self.transform(student) - teacher).pow(2).mean()


class AttentionTransfer(nn.Module):
    def forward(self, student, teacher):
        s_attention = F.normalize(student.pow(2).mean(1).view(student.size(0), -1))

        with torch.no_grad():
            t_attention = F.normalize(teacher.pow(2).mean(1).view(teacher.size(0), -1))

        return (s_attention - t_attention).pow(2).mean()


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss
#sp
class SP(nn.Module):

	#Similarity-Preserving Knowledge Distillation

    def __init__(self):

        super(SP, self).__init__()

    def forward(self, fm_s, fm_t):

        with torch.no_grad(): 
            fm_t = fm_t.view(fm_t.size(0), -1)
            G_t  = torch.mm(fm_t, fm_t.t())
            norm_G_t = F.normalize(G_t, p=2, dim=1)
        fm_s = fm_s.view(fm_s.size(0), -1)
        G_s  = torch.mm(fm_s, fm_s.t())
        norm_G_s = F.normalize(G_s, p=2, dim=1)
        loss = F.mse_loss(norm_G_s, norm_G_t)

        return loss

class SE_Fusion(nn.Module):
    def __init__(self,channel_s=128,channel_t=256,reduction=4):
        super(SE_Fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_s = nn.Sequential(
            nn.Linear((channel_s+channel_t), (channel_s+channel_t) // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_s+channel_t) // reduction, channel_s, bias=False),
            nn.Sigmoid()
        )
        self.fc_t = nn.Sequential(
            nn.Linear((channel_s+channel_t), (channel_s+channel_t) // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_s+channel_t) // reduction, channel_t, bias=False),
            nn.Sigmoid()
        )

    def forward(self, student, teacher):

        #squeeze
        b_s, c_s, _, _ = student.size()
        y_s = self.avg_pool(student).view(b_s, c_s)
        b_t, c_t, _, _ = teacher.size()
        y_t = self.avg_pool(teacher).view(b_t, c_t)
        #joint
        z=torch.cat((y_s,y_t),1)
        #excitation
        y_s = self.fc_s(z).view(b_s, c_s, 1, 1)
        y_t = self.fc_t(z).view(b_t, c_t, 1, 1)        

        
        return student * y_s.expand_as(student), teacher * y_t.expand_as(teacher)

class SoftTarget(nn.Module):
	'''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T
	def forward(self, out_s, out_t):
            loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                            F.softmax(out_t/self.T, dim=1),
                            reduction='batchmean') * self.T * self.T

            return loss        
class Gram_loss(nn.Module):

	#Similarity-Preserving Knowledge Distillation

    def __init__(self):

        super(Gram_loss, self).__init__()

    def forward(self, student, teacher):
        student = torch.tensor(student, dtype=torch.float32)
        teacher = torch.tensor(teacher, dtype=torch.float32)
        with torch.no_grad():
            norm_teacher = F.normalize(teacher, p=2, dim=1)
        norm_student = F.normalize(student, p=2, dim=1)
        loss = F.mse_loss(norm_teacher, norm_student)
        return loss          

#fitnet
class Hint(nn.Module):
	'''
	FitNets: Hints for Thin Deep Nets
	https://arxiv.org/pdf/1412.6550.pdf
	'''
	def __init__(self):
		super(Hint, self).__init__()

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(fm_s, fm_t)

		return loss

#at
class AT(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self, p):
		super(AT, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am        
#rkd
class RKD(nn.Module):
	'''
	Relational Knowledge Distillation
	https://arxiv.org/pdf/1904.05068.pdf
	'''
	def __init__(self, w_dist, w_angle):
		super(RKD, self).__init__()

		self.w_dist  = w_dist
		self.w_angle = w_angle

	def forward(self, feat_s, feat_t):
		loss = self.w_dist * self.rkd_dist(feat_s, feat_t) + \
			   self.w_angle * self.rkd_angle(feat_s, feat_t)

		return loss

	def rkd_dist(self, feat_s, feat_t):
		feat_t_dist = self.pdist(feat_t, squared=False)
		mean_feat_t_dist = feat_t_dist[feat_t_dist>0].mean()
		feat_t_dist = feat_t_dist / mean_feat_t_dist

		feat_s_dist = self.pdist(feat_s, squared=False)
		mean_feat_s_dist = feat_s_dist[feat_s_dist>0].mean()
		feat_s_dist = feat_s_dist / mean_feat_s_dist

		loss = F.smooth_l1_loss(feat_s_dist, feat_t_dist)

		return loss

	def rkd_angle(self, feat_s, feat_t):
		# N x C --> N x N x C
		feat_t_vd = (feat_t.unsqueeze(0) - feat_t.unsqueeze(1))
		norm_feat_t_vd = F.normalize(feat_t_vd, p=2, dim=2)
		feat_t_angle = torch.bmm(norm_feat_t_vd, norm_feat_t_vd.transpose(1, 2)).view(-1)

		feat_s_vd = (feat_s.unsqueeze(0) - feat_s.unsqueeze(1))
		norm_feat_s_vd = F.normalize(feat_s_vd, p=2, dim=2)
		feat_s_angle = torch.bmm(norm_feat_s_vd, norm_feat_s_vd.transpose(1, 2)).view(-1)

		loss = F.smooth_l1_loss(feat_s_angle, feat_t_angle)

		return loss

	def pdist(self, feat, squared=False, eps=1e-12):
		feat_square = feat.pow(2).sum(dim=1)
		feat_prod   = torch.mm(feat, feat.t())
		feat_dist   = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=eps)

		if not squared:
			feat_dist = feat_dist.sqrt()

		feat_dist = feat_dist.clone()
		feat_dist[range(len(feat)), range(len(feat))] = 0

		return feat_dist
##cc
class CC(nn.Module):
	'''
	Correlation Congruence for Knowledge Distillation
	http://openaccess.thecvf.com/content_ICCV_2019/papers/
	Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.pdf
	'''
	def __init__(self, gamma, P_order):
		super(CC, self).__init__()
		self.gamma = gamma
		self.P_order = P_order

	def forward(self, feat_s, feat_t):
		corr_mat_s = self.get_correlation_matrix(feat_s)
		corr_mat_t = self.get_correlation_matrix(feat_t)

		loss = F.mse_loss(corr_mat_s, corr_mat_t)

		return loss

	def get_correlation_matrix(self, feat):
		feat = F.normalize(feat, p=2, dim=-1)
		sim_mat  = torch.matmul(feat, feat.t())
		corr_mat = torch.zeros_like(sim_mat)

		for p in range(self.P_order+1):
			corr_mat += math.exp(-2*self.gamma) * (2*self.gamma)**p / \
						math.factorial(p) * torch.pow(sim_mat, p)

		return corr_mat

class IRG(nn.Module):
	'''
	Knowledge Distillation via Instance Relationship Graph
	http://openaccess.thecvf.com/content_CVPR_2019/papers/
	Liu_Knowledge_Distillation_via_Instance_Relationship_Graph_CVPR_2019_paper.pdf

	The official code is written by Caffe
	https://github.com/yufanLIU/IRG
	'''
	def __init__(self, w_irg_vert, w_irg_edge, w_irg_tran):
		super(IRG, self).__init__()

		self.w_irg_vert = w_irg_vert
		self.w_irg_edge = w_irg_edge
		self.w_irg_tran = w_irg_tran

	def forward(self, irg_s, irg_t):
		fm_s1, fm_s2, feat_s, out_s = irg_s
		fm_t1, fm_t2, feat_t, out_t = irg_t

		loss_irg_vert = F.mse_loss(out_s, out_t)

		irg_edge_feat_s = self.euclidean_dist_feat(feat_s, squared=True)
		irg_edge_feat_t = self.euclidean_dist_feat(feat_t, squared=True)
		irg_edge_fm_s1  = self.euclidean_dist_fm(fm_s1, squared=True)
		irg_edge_fm_t1  = self.euclidean_dist_fm(fm_t1, squared=True)
		irg_edge_fm_s2  = self.euclidean_dist_fm(fm_s2, squared=True)
		irg_edge_fm_t2  = self.euclidean_dist_fm(fm_t2, squared=True)
		loss_irg_edge = (F.mse_loss(irg_edge_feat_s, irg_edge_feat_t) +
						 F.mse_loss(irg_edge_fm_s1,  irg_edge_fm_t1 ) +
						 F.mse_loss(irg_edge_fm_s2,  irg_edge_fm_t2 )) / 3.0

		irg_tran_s = self.euclidean_dist_fms(fm_s1, fm_s2, squared=True)
		irg_tran_t = self.euclidean_dist_fms(fm_t1, fm_t2, squared=True)
		loss_irg_tran = F.mse_loss(irg_tran_s, irg_tran_t)

		# print(self.w_irg_vert * loss_irg_vert)
		# print(self.w_irg_edge * loss_irg_edge)
		# print(self.w_irg_tran * loss_irg_tran)
		# print()

		loss = (self.w_irg_vert * loss_irg_vert +
				self.w_irg_edge * loss_irg_edge +
				self.w_irg_tran * loss_irg_tran)

		return loss

	def euclidean_dist_fms(self, fm1, fm2, squared=False, eps=1e-12):
		'''
		Calculating the IRG Transformation, where fm1 precedes fm2 in the network.
		'''
		if fm1.size(2) > fm2.size(2):
			fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))
		if fm1.size(1) < fm2.size(1):
			fm2 = (fm2[:,0::2,:,:] + fm2[:,1::2,:,:]) / 2.0

		fm1 = fm1.view(fm1.size(0), -1)
		fm2 = fm2.view(fm2.size(0), -1)
		fms_dist = torch.sum(torch.pow(fm1-fm2, 2), dim=-1).clamp(min=eps)

		if not squared:
			fms_dist = fms_dist.sqrt()

		fms_dist = fms_dist / fms_dist.max()

		return fms_dist

	def euclidean_dist_fm(self, fm, squared=False, eps=1e-12): 
		'''
		Calculating the IRG edge of feature map. 
		'''
		fm = fm.view(fm.size(0), -1)
		fm_square = fm.pow(2).sum(dim=1)
		fm_prod   = torch.mm(fm, fm.t())
		fm_dist   = (fm_square.unsqueeze(0) + fm_square.unsqueeze(1) - 2 * fm_prod).clamp(min=eps)

		if not squared:
			fm_dist = fm_dist.sqrt()

		fm_dist = fm_dist.clone()
		fm_dist[range(len(fm)), range(len(fm))] = 0
		fm_dist = fm_dist / fm_dist.max()

		return fm_dist

	def euclidean_dist_feat(self, feat, squared=False, eps=1e-12):
		'''
		Calculating the IRG edge of feat.
		'''
		feat_square = feat.pow(2).sum(dim=1)
		feat_prod   = torch.mm(feat, feat.t())
		feat_dist   = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=eps)

		if not squared:
			feat_dist = feat_dist.sqrt()

		feat_dist = feat_dist.clone()
		feat_dist[range(len(feat)), range(len(feat))] = 0
		feat_dist = feat_dist / feat_dist.max()

		return feat_dist        

class OFD(nn.Module):
	'''
	A Comprehensive Overhaul of Feature Distillation
	http://openaccess.thecvf.com/content_ICCV_2019/papers/
	Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.pdf
	'''
	def __init__(self, in_channels, out_channels):
		super(OFD, self).__init__()
		self.connector = nn.Sequential(*[
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(out_channels)
			])

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, fm_s, fm_t):
		margin = self.get_margin(fm_t)
		fm_t = torch.max(fm_t, margin)
		fm_s = self.connector(fm_s)

		mask = 1.0 - ((fm_s <= fm_t) & (fm_t <= 0.0)).float()
		loss = torch.mean((fm_s - fm_t)**2 * mask)

		return loss

	def get_margin(self, fm, eps=1e-6):
		mask = (fm < 0.0).float()
		masked_fm = fm * mask

		margin = masked_fm.sum(dim=(0,2,3), keepdim=True) / (mask.sum(dim=(0,2,3), keepdim=True)+eps)

		return margin        

class AFD(nn.Module):
	'''
	Pay Attention to Features, Transfer Learn Faster CNNs
	https://openreview.net/pdf?id=ryxyCeHtPB
	'''
	def __init__(self, in_channels, att_f):
		super(AFD, self).__init__()
		mid_channels = int(in_channels * att_f)

		self.attention = nn.Sequential(*[
				nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=True),
				nn.ReLU(inplace=True),
				nn.Conv2d(mid_channels, in_channels, 1, 1, 0, bias=True)
			])

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
		
	def forward(self, fm_s, fm_t, eps=1e-6):
		fm_t_pooled = F.adaptive_avg_pool2d(fm_t, 1)
		rho = self.attention(fm_t_pooled)
		# rho = F.softmax(rho.squeeze(), dim=-1)
		rho = torch.sigmoid(rho.squeeze())
		rho = rho / torch.sum(rho, dim=1, keepdim=True)

		fm_s_norm = torch.norm(fm_s, dim=(2,3), keepdim=True)
		fm_s      = torch.div(fm_s, fm_s_norm+eps)
		fm_t_norm = torch.norm(fm_t, dim=(2,3), keepdim=True)
		fm_t      = torch.div(fm_t, fm_t_norm+eps)

		loss = rho * torch.pow(fm_s-fm_t, 2).mean(dim=(2,3))
		loss = loss.sum(1).mean(0)

		return loss        

class Logits(nn.Module):
	'''
	Do Deep Nets Really Need to be Deep?
	http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf
	'''
	def __init__(self):
		super(Logits, self).__init__()

	def forward(self, out_s, out_t):
		loss = F.mse_loss(out_s, out_t)

		return loss
