import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import ast

class RGBDAugmentor:
    """ perform augmentation on RGB-D video """

    def __init__(self, reshape_size, datapath=None):
        self.reshape_size = reshape_size
        if type(self.reshape_size) == str:
            self.reshape_size = ast.literal_eval(self.reshape_size)
        p_gray = 0.1
        self.augcolor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.4/3.14),
            transforms.RandomGrayscale(p=p_gray),
            transforms.ToTensor()])

    def color_transform(self, images):
        """ color jittering """
        num, ch, ht, wd = images.shape
        images = images.permute(1, 2, 3, 0).reshape(ch, ht, wd*num)
        images = 255 * self.augcolor(images[[2,1,0]] / 255.0)
        return images[[2,1,0]].reshape(ch, ht, wd, num).permute(3,0,1,2).contiguous()

    def __call__(self, images, poses, intrinsics):
        images = self.color_transform(images)

        sizey, sizex = self.reshape_size
        scalex = sizex / images.shape[-1]
        scaley = sizey / images.shape[-2]
        xidx = np.array([0,2])
        yidx = np.array([1,3])
        intrinsics[:,xidx] = scalex * intrinsics[:,xidx]
        intrinsics[:,yidx] = scaley * intrinsics[:,yidx]
            
        images = F.interpolate(images, size=self.reshape_size)
        return images, poses, intrinsics