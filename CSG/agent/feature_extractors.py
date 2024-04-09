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
from .utils import add_coord, conv_block, conv_coord_block, res_like_block, res_combo_block

# we'll use three operations
from einops import rearrange, reduce, repeat
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import CombinedExtractor


class BaseExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, dropout=0.0):
        """
        Encoder for 2D CSGNet.
        :param dropout: dropout
        """
        super(BaseExtractor, self).__init__(
            observation_space, features_dim)
        self.p = dropout
        self.output_dim = features_dim
        in_channels = observation_space.shape[0]
        
        
        modules = []
        modules.extend(self.get_conv_pack(in_channels, 8, dropout))
        modules.extend(self.get_conv_pack(8, 16, dropout))
        modules.extend(self.get_conv_pack(16, 32, dropout))
        modules.extend(self.get_conv_pack(32, 64, dropout))
        modules.extend([nn.Flatten()])
        self.extractor = nn.Sequential(*modules)
        
        # self.initialize_weights()
        self.apply(self.initialize_weights)

    def get_conv_pack(self, in_channels, out_channels, dropout):
        output = [
            nn.Conv2d(in_channels, out_channels, 3, padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2)
        ]
        return output
    
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

    def forward(self, x_in):
        # (X, X, 64, 64) INPUT
        x = self.extractor(x_in)
        return x

class ConvCoordExtractor(BaseExtractor):

    def get_conv_pack(self, in_channels, out_channels, dropout):
        output = [
            add_coord(),
            nn.Conv2d(in_channels + 2, out_channels, 3, padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2)
        ]
        return output
    
class NoStackConvCoordExtractor(ConvCoordExtractor):
    def __init__(self, observation_space, features_dim, dropout=0.0):
        """
        Encoder for 2D CSGNet. = Change: only 1 input.
        :param dropout: dropout
        """

        super(BaseExtractor, self).__init__(
            observation_space, features_dim)
        
        self.output_dim = features_dim
        self.p = dropout
        
        in_channels = 1# observation_space.shape[0]
        
        self.extractor = nn.Sequential([
            *self.get_conv_pack(in_channels, 8, dropout),
            *self.get_conv_pack(8+2, 16, dropout),
            *self.get_conv_pack(16+2, 32, dropout),
            *self.get_conv_pack(32+2, 64, dropout),
            nn.Flatten()
        ])
        # self.initialize_weights()
        self.apply(self.initialize_weights)

    def forward(self, x_in):
        x_in = x_in[:,:1]
        x = self.extractor(x_in)
        return x

## TODO: Handle Variables converted into 1-hot. 
class RNNNoStackConvCoordExtractor(NoStackConvCoordExtractor):
    
    def __init__(self, observation_space, features_dim, dropout=0.0):
        
        super(RNNNoStackConvCoordExtractor, self).__init__(observation_space, features_dim, dropout)
        
        self.output_dim = features_dim
        self.embeddings = nn.Embedding(401, 256)
        self.recurrent = nn.GRU(256, 1024, 1, batch_first=True, dropout=dropout)
        self.start_token = None
        self.apply(self.initialize_weights)
        
    def forward(self, x_in, y_in, y_length):
        cnn_features = super(RNNNoStackConvCoordExtractor, self).forward(x_in)
        # Append start token
        # Get embeddings:
        extracted_embeddings = self.embeddings(y_in)
        if len(y_length.shape) == 2:
            y_length = y_length[:,0]
        rnn_input = torch.nn.utils.rnn.pack_padded_sequence(extracted_embeddings, (y_length+1).cpu(), batch_first=True,enforce_sorted=False)
        
        rnn_features, h_ = self.recurrent(rnn_input)
        rnn_features, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_features, batch_first=True)
        # TODO: Change clumsy index select:
        rnn_features = torch.stack([rnn_features[i, j] for i, j in enumerate(y_length)], 0)
        # rnn_features_ = torch.index_select(rnn_features, 1, y_length-1)
        cat_features = torch.cat([cnn_features, rnn_features], 1)
        return cat_features 
    

class RNNConvCoordExtractor(ConvCoordExtractor):
    
    def __init__(self, observation_space, features_dim, dropout=0.0):
        
        super(RNNConvCoordExtractor, self).__init__(observation_space, features_dim, dropout)
        
        self.output_dim = features_dim
        self.embeddings = nn.Embedding(401, 256)
        self.recurrent = nn.GRU(256, 1024, 1, batch_first=True, dropout=dropout)
        self.start_token = None
        self.apply(self.initialize_weights)
        
    def forward(self, x_in, y_in, y_length):
        cnn_features = super(RNNConvCoordExtractor, self).forward(x_in)
        # Append start token
        # Get embeddings:
        extracted_embeddings = self.embeddings(y_in)
        if len(y_length.shape) == 2:
            y_length = y_length[:,0]
        rnn_input = torch.nn.utils.rnn.pack_padded_sequence(extracted_embeddings, (y_length+1).cpu(), batch_first=True,enforce_sorted=False)
        
        rnn_features, h_ = self.recurrent(rnn_input)
        rnn_features, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_features, batch_first=True)
        # TODO: Change clumsy index select:
        rnn_features = torch.stack([rnn_features[i, j] for i, j in enumerate(y_length)], 0)
        # rnn_features_ = torch.index_select(rnn_features, 1, y_length-1)
        cat_features = torch.cat([cnn_features, rnn_features], 1)
        return cat_features 
    

    
class MultiRNNConvCoordExtractor(RNNConvCoordExtractor):
    
    def __init__(self, observation_space, features_dim, dropout=0.0):
        
        super(MultiRNNConvCoordExtractor, self).__init__(observation_space, features_dim, dropout)
        self.output_dim = features_dim
        self.cnn_to_rnn = nn.Linear(1024, 512)
        self.start_nn = nn.Linear(512 + 256, 256)
        
    def forward(self, x_in, y_in, y_length):
        cnn_features = super(RNNConvCoordExtractor, self).forward(x_in)
        # Append start token
        # Get embeddings:
        
        extracted_embeddings = self.embeddings(y_in)
        # Replace 1st token with processed cnn feature
        x = self.cnn_to_rnn(cnn_features)
        x = torch.cat([x, extracted_embeddings[:,0]], 1)
        x = self.start_nn(x)
        # should be B * 256
        extracted_embeddings[:,0] = x
        
        if len(y_length.shape) == 2:
            y_length = y_length[:,0]
        
        rnn_input = torch.nn.utils.rnn.pack_padded_sequence(extracted_embeddings, (y_length+1).cpu(), batch_first=True,enforce_sorted=False)
        
        rnn_features, h_ = self.recurrent(rnn_input)
        rnn_features, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_features, batch_first=True)
        # TODO: Change clumsy index select:
        rnn_features = torch.stack([rnn_features[i, j] for i, j in enumerate(y_length)], 0)
        # rnn_features_ = torch.index_select(rnn_features, 1, y_length-1)
        cat_features = torch.cat([cnn_features, rnn_features], 1)
        return cat_features 
    
    
class LargeConvCoordExtractor(ConvCoordExtractor):
    def __init__(self, observation_space, features_dim, dropout=0.0):
        """
        Encoder for 2D CSGNet.
        :param dropout: dropout
        """
        # DOUBLE THE FEATURES: features_dim * 2
        super(BaseExtractor, self).__init__(
            observation_space, features_dim)
        in_channels = observation_space.shape[0]
        
        self.output_dim = features_dim
        modules = []
        modules.extend(self.get_conv_pack(in_channels, 8 * 2, dropout))
        modules.extend(self.get_conv_pack(8 * 2, 16 * 2, dropout))
        modules.extend(self.get_conv_pack(16 * 2, 32 * 2, dropout))
        modules.extend(self.get_conv_pack(32 * 2, 64 * 2, dropout))
        modules.extend([nn.Flatten()])
        self.extractor = nn.Sequential(*modules)
        
        # self.initialize_weights()
        self.apply(self.initialize_weights)



class LargeNoStackConvCoordExtractor(LargeConvCoordExtractor):
    def __init__(self, observation_space, features_dim, dropout=0.0):
        """
        Encoder for 2D CSGNet.
        :param dropout: dropout
        """
        super(BaseExtractor, self).__init__(
            observation_space, features_dim)
        # DOUBLE THE FEATURES: features_dim * 2
        in_channels = 1 # observation_space.shape[0]
        self.output_dim = features_dim
        modules = []
        modules.extend(self.get_conv_pack(in_channels, 8 * 2, dropout))
        modules.extend(self.get_conv_pack(8*2, 16 * 2, dropout))
        modules.extend(self.get_conv_pack(16*2, 32 * 2, dropout))
        modules.extend(self.get_conv_pack(32 * 2, 64 * 4, dropout))
        # modules.extend([nn.Flatten()])
        self.extractor = nn.Sequential(*modules)
        
        # self.initialize_weights()
        self.apply(self.initialize_weights)


    def forward(self, x_in):
        x_in = x_in[:,:1]
        x = self.extractor(x_in)
        return x

class ReplCNN(BaseExtractor):
    def __init__(self, observation_space, features_dim, dropout=0.0):
        super(BaseExtractor, self).__init__(observation_space, features_dim)
        
        layers = 4
        hiddenChannels = 64
        outputChannels = 64 * 2
                 
        def conv_block(in_channels, out_channels, p=True):
            return nn.Sequential(
                add_coord(),
                nn.Conv2d(in_channels + 2, out_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2))

        self.p = dropout
        self.output_dim = features_dim

        # channels for hidden
        hid_dim = hiddenChannels
        z_dim = outputChannels
        in_channels = observation_space.shape[0]

        self.extractor = nn.Sequential(conv_block(in_channels, 32),
                                       res_like_block(32, 32, maxpool=False),
                                       res_like_block(32, 32, maxpool=False),
                                       *self.get_conv_pack(32, 64, dropout=dropout),
                                       res_combo_block(64, 64),
                                       conv_block(64, 128),
                                       nn.Flatten())
        
        self.apply(self.initialize_weights)
        