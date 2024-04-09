### Create vision extractors:
"""
Defines Neural Networks
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd.variable import Variable
from typing import List, Tuple, Callable, Optional, Union, Dict, Type
import gym
from .utils import add_coord, conv_block, conv_coord_block, res_like_block

# we'll use three operations
from einops import rearrange, reduce, repeat
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import CombinedExtractor
from torchvision import models


class WrapperRes18Extractor(CombinedExtractor):

    def __init__(self, observation_space, features_dim):

        super(WrapperRes18Extractor, self).__init__(
            observation_space, features_dim)
        
        extractor = models.resnet18
        obs = observation_space['obs']
        last_index = -2
        
        # features_dim 
        features_dim = 2048
        self.create_model(extractor, last_index, obs, features_dim)
    
    def create_model(self, extractor, last_index, obs, features_dim):
        temp = extractor()
        layers = list(temp.children())[:last_index]
        first_conv = layers[0]
        in_channels = obs.shape[0]
        out_channels = first_conv.out_channels
        layers[0] = nn.Conv2d(in_channels, out_channels, first_conv.kernel_size, first_conv.stride, first_conv.padding, bias=first_conv.bias)
        self.extractor = torch.nn.Sequential(*(layers))
        self.extractor.output_dim = features_dim
        self.flatten = nn.Flatten()
        # self.extractor = BaseExtractor(obs, features_dim)
        self.apply(self.initialize_weights)

    def forward(self, obs_in):

        obs = obs_in['obs']
        features = self.extractor(obs)
        features = self.flatten(features)
        return features

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

class WrapperGoogleNetExtractor(WrapperRes18Extractor):
    
    def __init__(self, observation_space, features_dim):

        super(WrapperRes18Extractor, self).__init__(
            observation_space, features_dim)
        
        extractor = models.googlenet
        obs = observation_space['obs']
        last_index = -5
        self.pool = torch.nn.AvgPool2d(2)
        features_dim = 1024
        self.create_model(extractor, last_index, obs, features_dim)

    def forward(self, obs_in):

        obs = obs_in['obs']
        features = self.extractor(obs)
        features = self.pool(features)
        features = self.flatten(features)
        return features
    

class WrapperVGG16Extractor(WrapperGoogleNetExtractor):
    
    def __init__(self, observation_space, features_dim):

        super(WrapperRes18Extractor, self).__init__(
            observation_space, features_dim)
        
        extractor = models.vgg16
        obs = observation_space['obs']
        last_index = -1
        self.pool = torch.nn.AdaptiveAvgPool2d((2, 2))
        # feature dim
        features_dim = 2048
        self.create_model(extractor, last_index, obs, features_dim)

class WrapperRegNetExtractor(WrapperRes18Extractor):
    
    def __init__(self, observation_space, features_dim):

        super(WrapperRes18Extractor, self).__init__(
            observation_space, features_dim)
        
        extractor = models.regnet_x_400mf
        obs = observation_space['obs']
        last_index = -2
        
        #  features_dim
        features_dim = 1600
        self.create_model(extractor, last_index, obs, features_dim)
