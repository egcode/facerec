from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pdb import set_trace as bp
  
class ArcFaceLossMargin(nn.Module):
    # def __init__(self, num_classes, feat_dim, device, s=64.0, m=0.5):
    def __init__(self, num_classes, feat_dim, device, s=32.0, m=0.5):
        super(ArcFaceLossMargin, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weights = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.device = device

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = math.sin(math.pi-m)*m
        self.threshold = math.cos(math.pi-m)

    def forward(self, feat, label):

        eps = 1e-4
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        feat_l2norm = torch.div(feat, norms)
        feat_l2norm = feat_l2norm * self.s

        norms_w = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        weights_l2norm = torch.div(self.weights, norms_w)

        fc7 = torch.matmul(feat_l2norm, torch.transpose(weights_l2norm, 0, 1))

        if torch.cuda.is_available():
            label = label.cuda()
            fc7 = fc7.cuda()
        else:
            label = label.cpu()
            fc7 = fc7.cpu()

        target_one_hot = torch.zeros(len(label), self.num_classes).to(self.device)
        target_one_hot = target_one_hot.scatter_(1, label.unsqueeze(1), 1.)        
        zy = torch.addcmul(torch.zeros(fc7.size()).to(self.device), 1., fc7, target_one_hot)
        zy = zy.sum(-1)

        cos_theta = zy/self.s
        cos_theta = cos_theta.clamp(min=-1+eps, max=1-eps) # for numerical stability

        theta = torch.acos(cos_theta)
        theta = theta+self.m

        body = torch.cos(theta)
        new_zy = body*self.s

        diff = new_zy - zy
        diff = diff.unsqueeze(1)

        body = torch.addcmul(torch.zeros(diff.size()).to(self.device), 1., diff, target_one_hot)
        output = fc7+body

        return output.to(self.device)


# class ArcFaceLossMargin2(nn.Module):
#     # def __init__(self, num_classes, feat_dim, device, s=64.0, m=0.50, easy_margin = False):
#     def __init__(self, num_classes, feat_dim, device, s=32.0, m=0.50, easy_margin = False):
#         super(ArcFaceLossMargin2, self).__init__()
#         self.in_features = feat_dim
#         self.out_features = num_classes

#         self.device = device

#         self.s = s
#         self.m = m
        
#         self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
#         nn.init.xavier_uniform_(self.weight)
      
#         self.easy_margin = easy_margin
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m

#     def forward(self, input, label):
#         cosine = F.linear(F.normalize(input), F.normalize(self.weight))
#         sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
#         phi = cosine * self.cos_m - sine * self.sin_m
#         if self.easy_margin:
#             phi = torch.where(cosine > 0, phi, cosine)
#         else:
#             phi = torch.where(cosine > self.th, phi, cosine - self.mm)

#         one_hot = torch.zeros(cosine.size())
#         one_hot = one_hot.to(self.device)
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
#         output *= self.s

#         # return output
#         return output.to(self.device)







