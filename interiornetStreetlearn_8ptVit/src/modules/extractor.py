import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1, kernel_size=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        if kernel_size > 1:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if stride > 1 or kernel_size > 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if stride > 1 or kernel_size > 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if stride > 1 or kernel_size > 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if stride > 1 or kernel_size > 1:
                self.norm3 = nn.Sequential()

        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)
        elif kernel_size > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride), self.norm3)
        else:    
            self.downsample = None

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        if y.shape[1] < x.shape[1]:
            x = x[:,:y.shape[1]]
        elif y.shape[1] > x.shape[1]:
            remaining_zeros = torch.zeros([x.shape[0],y.shape[1] - x.shape[1],x.shape[2],x.shape[3]]).cuda()
            x = torch.cat([x, remaining_zeros],dim=1)

        return self.relu(x+y)
