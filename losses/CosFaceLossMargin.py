from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable
from pdb import set_trace as bp

class CosFaceLossMargin(nn.Module):
    # def __init__(self, num_classes, feat_dim, device, s=7.00, m=0.2):
    def __init__(self, num_classes, feat_dim, device, s=64.00, m=0.35):

        super(CosFaceLossMargin, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.device = device

        # For Softmax Feeding after model features
        self.prelu = nn.PReLU().to(self.device)

    def forward(self, feat, label):
        # For Softmax Feeding after model features    
        feat_prelu = self.prelu(feat)

        batch_size = feat_prelu.shape[0]
        norms = torch.norm(feat_prelu, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat_prelu, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, self.num_classes).to(self.device)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot)
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (logits - y_onehot)

        return logits, margin_logits
