
from .transformers import PLADTransformerExtractor
import torch.nn as nn
from .default_tf import (MultiHeadedAttention, PositionwiseFeedForward, FixedPositionalEncoding, LearnablePositionalEncoding,
                         Encoder, EncoderLayer, Decoder, DecoderLayer, Embeddings,Generator, subsequent_mask)
from .plad_tf import DMLP, SDMLP, AttnLayer
from .rl_tf import GatedAttnLayer, GatedDecoderLayer
import torch

from .extractors_3d import Vox3DCNN
from .extractors_2d import Vox2DCNN


class Sampler(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(Sampler, self).__init__()        
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, hidden_size)
        self.mlp2var = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        encode = torch.relu(self.mlp1(x))
        
        mu = self.mlp2mu(encode)
        logvar = self.mlp2var(encode)

        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        # kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        # kld = -kld.sum(1).mean(0)
        kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1), 0)
        
        return eps.mul(std).add_(mu), kld


class Vox3DVAE(Vox3DCNN):
    """ Attach a VAE head to the input code.
    """
    def __init__(self, observation_space, features_dim, dropout=0.0, first_stride=2, out_len=64, latent_dim=128):

        super(Vox3DVAE, self).__init__(
            observation_space, features_dim, dropout, first_stride, out_len)

        self.latent_dim = latent_dim
        self.sampler = Sampler(out_len * features_dim, latent_dim)
        self.up_ll = nn.Linear(latent_dim, out_len * features_dim)
        self.mode = "encode"
    
    def set_mode(self, mode):
        self.mode = mode
    
    def forward(self, input):
        batch_size = input.shape[0]
        device = input.device
        if self.mode == "encode":
            features = super(Vox3DVAE, self).forward(input)   
            features= features.view(batch_size, -1) 
            gaussian_code, kld_loss = self.sampler(features)
            self.kld_loss = kld_loss 
        elif self.mode == "sample":
            gaussian_code = torch.randn(batch_size, self.latent_dim).to(device)
        features = self.up_ll(gaussian_code)
        features = features.view(batch_size, self.out_len, -1)
        return features


        
class Vox2DVAE(Vox2DCNN):
    """ Attach a VAE head to the input code.
    """
    def __init__(self, observation_space, features_dim, dropout=0.0, first_stride=2, out_len=64, latent_dim=128):

        super(Vox2DVAE, self).__init__(
            observation_space, features_dim, dropout, first_stride, out_len)

        self.latent_dim = latent_dim
        self.sampler = Sampler(out_len * features_dim, latent_dim)
        self.up_ll = nn.Linear(latent_dim, out_len * features_dim)
        self.mode = "encode"
    
    def set_mode(self, mode):
        self.mode = mode
    
    def forward(self, input):
        batch_size = input.shape[0]
        device = input.device
        if self.mode == "encode":
            features = super(Vox2DVAE, self).forward(input)   
            features= features.view(batch_size, -1) 
            gaussian_code, kld_loss = self.sampler(features)
            self.kld_loss = kld_loss 
        elif self.mode == "sample":
            gaussian_code = torch.randn(batch_size, self.latent_dim).to(device)
        features = self.up_ll(gaussian_code)
        features = features.view(batch_size, self.out_len, -1)
        return features



class PLADTransVAE(PLADTransformerExtractor):

    def __init__(self, observation_space, features_dim, config):

        super(PLADTransformerExtractor, self).__init__(observation_space, features_dim)
        # Need access to other specs
        dropout = config.DROPOUT
        self.latent_dimesion = config.LATENT_DIM
        ## Parameters:
        self.set_settings(config, dropout)
        max_length = self.inp_seq_len + self.out_seq_len
        
        if config.INPUT_DOMAIN == "3D":
        
            self.cnn_extractor = Vox3DVAE(observation_space, self.hidden_dim, dropout, first_stride=self.cnn_first_stride,
                                        out_len=self.inp_seq_len)
        elif config.INPUT_DOMAIN == "2D":
            self.cnn_extractor = Vox2DVAE(observation_space, self.hidden_dim, dropout, first_stride=self.cnn_first_stride,
                                        out_len=self.inp_seq_len)
        
        if self.pos_encoding_type == "FIXED":
            self.pos_encoding = FixedPositionalEncoding(self.hidden_dim, dropout)
        else:
            self.pos_encoding = LearnablePositionalEncoding(self.hidden_dim, dropout, max_len=max_length)
        self.token_embedding = nn.Embedding(self.out_token_count, self.hidden_dim) 
        if self.attention_type == "DEFAULT":
            attention_class = AttnLayer
        elif self.attention_type == "FAST":
            raise ValueError("Use FastTransformerExtractor!")
        else:
            attention_class = GatedAttnLayer
            
        self.attn_layers = nn.ModuleList([attention_class(self.num_heads, self.hidden_dim, self.dropout) for _ in range(self.num_dec_layers)])
        if config.OLD_ARCH:
            self.attn_to_output = nn.Linear(self.hidden_dim, self.output_dim)
        else:
            self.attn_to_output = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim), nn.LeakyReLU(0.2), nn.Dropout(self.dropout))
        
        ## For Transformer:
        attn_mask = self.generate_attn_mask()
        self.key_mask = None
        # self.attn_mask = self.attn_mask.to(self.init_device)
        start_token = torch.LongTensor([[self.out_token_count - 1]])# .to(self.init_device)
        self.register_buffer("start_token", start_token)
        self.register_buffer("attn_mask", attn_mask)
        
    
        self.apply(self.initialize_weights)