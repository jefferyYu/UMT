import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class myResnet(nn.Module):
    def __init__(self, resnet, if_fine_tune, device):
        super(myResnet, self).__init__()
        self.resnet = resnet
        self.if_fine_tune = if_fine_tune
        self.device = device

    def forward(self, x, att_size=7):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2)
        att = F.adaptive_avg_pool2d(x,[att_size,att_size])

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)

        if not self.if_fine_tune:
            
            x= Variable(x.data)
            fc = Variable(fc.data)
            att = Variable(att.data)

        return x, fc, att


