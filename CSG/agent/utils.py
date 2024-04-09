

from typing import Tuple
import torch as th
import torch.nn.functional as F
import torch.nn as nn

class add_coord(nn.Module):
    def __init__(self):
        super(add_coord, self).__init__()
        self.bs = None
        self.ch = None
        self.h_coord = None
        self.w_coord = None

    def forward(self, x):
        if self.bs == None:
            bs, ch, h, w = x.size()
            self.h_coord = th.range(
                start=0, end=h-1).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat([bs, 1, 1, w])/(h/2)-1
            self.w_coord = th.range(
                start=0, end=w-1).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat([bs, 1, h, 1])/(w/2)-1
            self.h_coord = self.h_coord.cuda()
            self.w_coord = self.w_coord.cuda()
        h_coord = self.h_coord# .clone()
        w_coord = self.w_coord# .clone()
        return th.cat([x, h_coord, w_coord], dim=1)


def conv_block(in_channels, out_channels, p=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2))

def conv_coord_block(in_channels, out_channels, p=True):
    return nn.Sequential(
        add_coord(),
        nn.Conv2d(in_channels + 2, out_channels, 3, padding=1),
        nn.ReLU(),
        add_coord(),
        nn.Conv2d(out_channels + 2, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2))
        
class ResLikeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, maxpool):
        super(ResLikeBlock, self).__init__()
        self.encoder = nn.Sequential(
            add_coord(),
            nn.Conv2d(in_channels + 2, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU())
        self.maxpool = maxpool
        if self.maxpool:
            self.maxpool = nn.MaxPool2d(2)
        
    def forward(self, x_in):
        encoding = self.encoder(x_in)
        encoding = encoding + x_in
        if self.maxpool:
            output = self.maxpool(encoding)
        else:
            output = encoding
        return output
        
def res_like_block(in_channels, out_channels, p=True, maxpool=True):
    return ResLikeBlock(in_channels, out_channels, maxpool)

def res_combo_block(in_channels, out_channels, p=True):
    
    return nn.Sequential(
        res_like_block(in_channels, out_channels, maxpool=False),
        res_like_block(out_channels, out_channels, maxpool=True)
        )

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


