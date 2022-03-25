import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GHMC_loss(torch.nn.Module):
    def __init__(self, bins=10, momentum=0, use_sigmiod=True, loss_weight=1.0, batchsize=64, clsnum=2):
        super(GHMC_loss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.use_sigmoid = use_sigmiod
        self.loss_weight = loss_weight
        self.label_weight = torch.zeros(batchsize, clsnum).cuda()

    def forward(self, pred, target, label_weight):
        '''

        :param pred:[batch_num, class_num]:
        :param target:[batch_num]:Binary class target for each sample.
        :param label_weight:[batch_num, class_num]: the value is 1 if the sample is valid and 0 if ignored.
        :return: GHMC_Loss
        '''
        #label_weight = self.label_weight#样本权重先不改
        if not self.use_sigmoid:
            raise NotImplementedError
        target = target.view(-1,1)
        target_onehot_t = torch.zeros(len(target), 2).cuda()
        target_onehot = target_onehot_t.scatter_(1, target, 1)
        pred, target, label_weight = pred.float(), target_onehot.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred).cuda()
        # gradient length
        #g = torch.abs(pred.sigmoid().detach() - target)
        g = torch.abs(F.softmax(pred,dim=1).detach() - target).cuda()
        valid = label_weight > 0
        total = max(valid.float().sum().item(), 1.0)
        n = 0  # the number of valid bins

        for i in range(self.bins):
            inds = (g >= edges[i]) & (g <= edges[i + 1]) & valid
            num_in_bins = inds.sum().item()
            if num_in_bins > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bins
                    weights[inds] = total / self.acc_sum[i]
                else:
                    weights[inds] = total / num_in_bins
                n += 1

        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(pred, target, weights, reduction='sum') / total

        return loss * self.loss_weight	



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, size_average=False):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        # 如果模型最后没有 nn.Sigmoid()，那么这里就需要对预测结果计算一次 Sigmoid 操作
        pred = F.sigmoid(pred)#应该是softmax先算一下分类比对后的分类概率

        # 展开 pred 和 target,此时 pred.size = target.size = (BatchSize,1) 
        #pred = pred.view(-1,1)
        target = target.view(-1,1)

        # 此处将预测样本为正负的概率都计算出来，此时 pred.size = (BatchSize,2)
        #pred = torch.cat((1-pred,pred),dim=1)

        # 根据 target 生成 mask，即根据 ground truth 选择所需概率
        # 用大白话讲就是：
        # 当标签为 1 时，我们就将模型预测该样本为正类的概率代入公式中进行计算
        # 当标签为 0 时，我们就将模型预测该样本为负类的概率代入公式中进行计算
        class_mask = torch.zeros(pred.shape[0],pred.shape[1]).cuda()
        # 这里的 scatter_ 操作不常用，其函数原型为:
        # scatter_(dim,index,src)->Tensor
        # Writes all values from the tensor src into self at the indices specified in the index tensor. 
        # For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        # 利用 mask 将所需概率值挑选出来
        probs = (pred * class_mask).sum(dim=1).view(-1,1)
        probs = probs.clamp(min=0.0001,max=1.0)

        # 计算概率的 log 值
        log_p = probs.log()

        # 根据论文中所述，对 alpha　进行设置（该参数用于调整正负样本数量不均衡带来的问题）
        alpha = torch.ones(pred.shape[0],pred.shape[1]).cuda()
        alpha[:,0] = alpha[:,0] * (1-self.alpha)
        alpha[:,1] = alpha[:,1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1,1)

        # 根据 Focal Loss 的公式计算 Loss
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

         # Loss Function的常规操作，mean 与 sum 的区别不大，相当于学习率设置不一样而已
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss
		
class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=2, feat_dim=48, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask_classcentre = torch.tensor([1.0,1.0]).expand(batch_size, self.num_classes).cuda()
        mask = labels.eq(classes.expand(batch_size, self.num_classes))*mask_classcentre.float()#只要

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class Negative_MLS_loss(torch.nn.Module):
    def __init__(self):
        super(Negative_MLS_loss, self).__init__()

    def negative_MLS_torch(self,X, Y, sigma_sq_X, sigma_sq_Y, mean=False):
        if mean:
            D = X.size(1)
            Y = torch.transpose(Y,0,1)
            XX = torch.sum(torch.square(X), axis=1,keepdim=True)
            YY = torch.sum(torch.square(Y), axis=0,keepdim=True)
            XY = torch.matmul(X,Y)
            diffs = XX + YY - 2*XY

            sigma_sq_Y = torch.transpose(sigma_sq_Y,0,1)
            sigma_sq_X = torch.mean(sigma_sq_X, axis=1,keepdim=True)  # [batch_size, 1]
            sigma_sq_Y = torch.mean(sigma_sq_Y, axis=0,keepdim=True)  # [1, batch_size]
            sigma_sq_fuse = sigma_sq_X + sigma_sq_Y  # [batch_size, batch_size]

            diffs = diffs / (1e-8 + sigma_sq_fuse) + D * torch.log(sigma_sq_fuse)
            return diffs
        else:
            D = X.size(1) # embedding_size
            X = X.view(-1, 1, D)  # [batch_size, 1, embedding_size]
            Y = Y.view(1, -1, D)  # [1, batch_size, embedding_size]
            sigma_sq_X = sigma_sq_X.view(-1, 1, D)  # [batch_size, 1, embedding_size]
            sigma_sq_Y = sigma_sq_Y.view(1, -1, D)  # [1, batch_size, embedding_size]
            sigma_sq_fuse = sigma_sq_X + sigma_sq_Y  # [batch_size, batch_size, embedding_size],就能够得到所有i、j交叉求和的结果
            diffs = torch.square(X - Y) / (1e-10 + sigma_sq_fuse) + torch.log(sigma_sq_fuse)  # 等式(3)
            return torch.sum(diffs, axis=2)  # 返回[batch_size, batch_size]

    def forward(self, mu, log_sigma_sq,labels):
        batch_size = mu.size(0)
        mask = torch.zeros((batch_size,batch_size), dtype=bool)

        diag_mask = torch.eye(batch_size, dtype=bool)
        non_diag_mask = torch.logical_not(diag_mask)

        sigma_sq = torch.exp(log_sigma_sq)
        loss_mat = self.negative_MLS_torch(mu, mu, sigma_sq, sigma_sq,mean=True)  # 返回的size为[batch_size, batch_size]，每个位置ij的值表示batch_i对batch_j的结果

        x1,y1 = torch.where((labels[:, None]-labels[None,:])==0)
        mask[x1,y1] = True
        label_mask_pos = torch.logical_and(non_diag_mask, mask)  # 得到类相同的mask
        loss_pos = loss_mat[label_mask_pos]  # 得到类相同的两个输入计算得到的loss

        return loss_pos.mean()  # 然后计算他们的均值得到最后的损失结果
        
def aggregate_PFE(x, sigma_sq=None, normalize=True, concatenate=False):

    mu = x
    attention = 1. / sigma_sq  # 等式(7)
    attention = attention / np.sum(attention, axis=0, keepdims=True)  # 等式(7)

    mu_new = np.sum(mu * attention, axis=0)  # 等式(6)
    sigma_sq_new = np.max(sigma_sq, axis=0)  # 取最大值做新的不确定度
    #y = x.shape[0]
    #sigma_sq_new = y/np.sum(1. / sigma_sq, axis=0) # 取调和平均值作为衡量
    if normalize:
        mu_new = np.linalg.norm(mu_new)

    if concatenate:
        return np.concatenate([mu_new, sigma_sq_new])
    else:
        return sigma_sq_new
		
class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3, global_feat=None, labels=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap,dist_an = [],[]
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        # Compute ranking hinge loss
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        #loss = dist_ap
        return self.ranking_loss(dist_an, dist_ap, y)#max(0,margin+y(x2-x1))

class TripletLoss_withanchor(nn.Module):

    def __init__(self, margin=0.3, global_feat=None, labels=None):
        super(TripletLoss_withanchor, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets, anchor):  # anchor是手指样本的centre
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        #n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        a = (inputs - anchor)
        dist = torch.linalg.norm(a, ord=2, dim=1)
        Mask = (targets == 0)
        loss_ap = dist[Mask].max()  # 难分的正样本
        loss_an = dist[Mask == 0].min()  # 难分的负样本
        # loss_ap = torch.topk(dist[Mask],3,largest=True) #难分的正样本
        # loss_an = torch.topk(dist[Mask==0],3,largest=False) #难分的负样本
        Ap = loss_ap
        An = loss_an
        loss_triplet = torch.maximum(torch.tensor([0]).cuda(), self.margin + Ap - An)
        return loss_triplet
		
class AMSoftmax(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes=10,
                 m=0.3,
                 s=15):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        lb_view = lb.view(-1, 1)
        if lb_view.is_cuda: lb_view = lb_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss
'''		
device = torch.device('cuda')
criterion_GHMCloss = GHMC_loss().to(device)
#pred = torch.tensor([[4,-6], [-5,5],[-4,6]]).cuda()
# target = torch.tensor([[0,1],[1,0]])
target = torch.tensor([1,0,1]).cuda()
label_weight = torch.ones((3,1)).cuda()
a = criterion_GHMCloss(pred,target,label_weight)
print(a,)
'''




		
