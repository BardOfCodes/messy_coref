
import copy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import feature_alpha_dropout
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import einops
from CSG.env.action_spaces import MULTI_ACTION_SPACE
from .default_tf import (MultiHeadedAttention, PositionwiseFeedForward, FixedPositionalEncoding, LearnablePositionalEncoding,
                         Encoder, EncoderLayer, Decoder, DecoderLayer, Embeddings,Generator, subsequent_mask)
from .extractors_3d import Vox3DCNN
from .extractors_2d import Vox2DCNN
from .rl_tf import GatedAttnLayer, GatedDecoderLayer
from .plad_tf import DMLP, SDMLP, AttnLayer
from .linear_tf import FastAttenLayer

class PLADTransformerExtractor(BaseFeaturesExtractor):
    
    def __init__(self, observation_space, features_dim, config):
        
        super(PLADTransformerExtractor, self).__init__(observation_space, features_dim)
        # Need access to other specs
        dropout = config.DROPOUT
        ## Parameters:
        self.set_settings(config, dropout)
        max_length = self.inp_seq_len + self.out_seq_len

        if config.INPUT_DOMAIN == "3D":
        
            self.cnn_extractor = Vox3DCNN(observation_space, self.hidden_dim, dropout, first_stride=self.cnn_first_stride,
                                        out_len=self.inp_seq_len)
        elif config.INPUT_DOMAIN == "2D":
            self.cnn_extractor = Vox2DCNN(observation_space, self.hidden_dim, dropout, first_stride=self.cnn_first_stride,
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
    
    def set_settings(self, config, dropout):
        
        ## Parameters:
        self.cnn_first_stride = config.CNN_FIRST_STRIDE
        self.output_dim = config.OUTPUT_DIM # 256
        self.inp_seq_len = config.INPUT_SEQ_LENGTH # 8
        self.out_seq_len = config.OUTPUT_SEQ_LENGTH + 1 #128  + 1# seq_len
        self.num_enc_layers = config.NUM_ENC_LAYERS# 8 # num_layers
        self.num_dec_layers = config.NUM_DEC_LAYERS# 8 # num_layers
        self.num_heads = config.NUM_HEADS# 16# num_heads
        self.input_token_count = config.INPUT_SEQ_LENGTH # 75 # len(ex.TOKENS)
        self.out_token_count = config.OUTPUT_TOKEN_COUNT + 1# 75 # len(ex.TOKENS) + start symbol
        self.hidden_dim = config.HIDDEN_DIM # 128 # hidden_dim 
        self.init_device = config.INIT_DEVICE 
        self.dropout = dropout
        self.return_all = config.RETURN_ALL #False
        self.pos_encoding_type = config.POS_ENCODING_TYPE
        self.attention_type = config.ATTENTION_TYPE
        self.zero_value = torch.FloatTensor([0])
        self.beam_mode = False
        #
        self.beam_partial_init = False
        self.x_count = None
    
    def enable_beam_mode(self):
        self.beam_mode = True
        self.beam_partial_init = True
    def disable_beam_mode(self):
        self.beam_mode = False
        self.beam_partial_init = False

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
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
                
    def generate_attn_mask(self):
        sz = self.inp_seq_len + self.out_seq_len
        mask = (torch.triu(torch.ones(sz, sz)) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).T
        mask[:self.inp_seq_len, :self.inp_seq_len] = 0.
        return mask

    def generate_key_mask(self, num, device):
        if num == self.key_mask.shape[0]:
            return self.key_mask
        else:
            sz = self.inp_seq_len + self.out_seq_len
            self.key_mask = torch.zeros(num, sz).bool().to(device)
        
    def generate_start_token(self, num, device):
        if not num == self.start_token.shape[0]:
            self.start_token = torch.LongTensor([[self.out_token_count - 1]]).to(device).repeat(num, 1)
            
    def forward(self, x_in, y_in, y_length):
        
        if self.beam_mode:
            return self.partial_beam_forward(x_in, y_in, y_length)

        batch_size = x_in.shape[0]

        self.generate_start_token(batch_size, x_in.device)
        y_in = torch.cat([self.start_token, y_in], 1)
        
        cnn_features = self.cnn_extractor.forward(x_in)
        token_embeddings = self.token_embedding(y_in)

        out = self.pos_encoding(torch.cat((cnn_features, token_embeddings), dim = 1))
        
        # self.generate_key_mask(batch_size, x_in.device)
        
        for attn_layer in self.attn_layers:        
            out = attn_layer(out, self.attn_mask, self.key_mask)
        seq_out = out[:,self.inp_seq_len:,:]
    
        if self.return_all:
            output = self.stack_vectors(seq_out, y_length)
        else:
            output = self.vector_gather(seq_out, y_length)
            
        if len(output.shape) == 3:
            output = output.squeeze(1)
        output = self.attn_to_output(output)
        return output
        
    def partial_beam_forward(self, x_in, y_in, y_length):
        assert not self.x_count is None, "Need to set x_count"
        if self.beam_partial_init:
            self.beam_partial_init = False
            self.cnn_features = self.cnn_extractor.forward(x_in).detach()
        
        # Replicate the cnn_features to get
        cnn_features = []
        for ind, count in enumerate(self.x_count):
            cnn_features.append(self.cnn_features[ind:ind+1].detach().expand(count, -1, -1))
        
        cnn_features = torch.cat(cnn_features, 0)
        
        batch_size = y_in.shape[0]
        ## Cut size:
        current_seq_len = y_length[0]
        
        self.generate_start_token(batch_size, y_in.device)
        y_in = torch.cat([self.start_token, y_in], 1)
        y_in = y_in[:,:current_seq_len + 1]
        
        token_embeddings = self.token_embedding(y_in)

        out = self.pos_encoding(torch.cat((cnn_features, token_embeddings), dim = 1))
        
        # self.generate_key_mask(batch_size, y_in.device)
        ## Cut size:
        total_len = self.inp_seq_len + (current_seq_len) + 1
        attn_mask = self.attn_mask[:total_len, :total_len]
        # key_mask = self.key_mask[:, :total_len].detach()
        for attn_layer in self.attn_layers:        
            out = attn_layer(out, attn_mask, None)
        seq_out = out[:,self.inp_seq_len:,:]
    
        if self.return_all:
            raise ValueError("Cant use Return all with this mode")
        else:
            output = seq_out[:,-1]
            
        if len(output.shape) == 3:
            output = output.squeeze(1)
        output = self.attn_to_output(output)
        return output
    
    def vector_gather(self, vectors, indices):
        """
        Gathers (batched) vectors according to indices.
        Arguments:
            vectors: Tensor[N, L, D]
            indices: Tensor[N, K] or Tensor[N]
        Returns:
            Tensor[N, K, D] or Tensor[N, D]
        """
        N, L, D = vectors.shape
        squeeze = False
        if indices.ndim == 1:
            squeeze = True
            indices = indices.unsqueeze(-1)
        N2, K = indices.shape
        assert N == N2
        indices = einops.repeat(indices, "N K -> N K D", D=D)
        out = torch.gather(vectors, dim=1, index=indices)
        if squeeze:
            out = out.squeeze(1)
        return out
        
    def stack_vectors(self, vectors, indices):
        output_list = []
        for ind in range(vectors.shape[0]):
            selected = vectors[ind, :indices[ind], :]
            output_list.append(selected)
        output = torch.cat(output_list, 0)
        return output
        
class DefaultTransformerExtractor(PLADTransformerExtractor):
    
    def __init__(self, observation_space, features_dim, config, dropout=0.0):
        
        super(PLADTransformerExtractor, self).__init__(observation_space, features_dim)
        
        self.set_settings(config, dropout)
        max_length = max(self.inp_seq_len, self.out_seq_len)
        # First a CNN for feature extraction
        self.cnn_extractor = Vox3DCNN(observation_space, self.hidden_dim, dropout, first_stride=self.cnn_first_stride, 
                                       out_len=self.inp_seq_len)
        
        attn = MultiHeadedAttention(self.num_heads, self.hidden_dim)
        ff = PositionwiseFeedForward(self.hidden_dim, self.output_dim, dropout)
        if self.pos_encoding_type == "FIXED":
            pos_encoding = FixedPositionalEncoding(self.hidden_dim, dropout)
        else:
            pos_encoding = LearnablePositionalEncoding(self.hidden_dim, dropout, max_len=max_length)
        
        # A TF Encoder for the n*n*n tokens
        self.encoder = Encoder(EncoderLayer(self.hidden_dim, copy.deepcopy(attn), 
                                            copy.deepcopy(ff), dropout), 
                               self.num_enc_layers)
        
        if self.attention_type == "DEFAULT":
            decoder_layer = DecoderLayer
        else:
            decoder_layer = GatedDecoderLayer
        # A TF Decoder for each prediction.
        self.decoder = Decoder(decoder_layer(self.hidden_dim, copy.deepcopy(attn), 
                                            copy.deepcopy(attn), copy.deepcopy(ff), dropout), 
                               self.num_dec_layers)
        
        # A output positional encodings part:
        self.source_token_embedding_pos = copy.deepcopy(pos_encoding)
        self.target_token_embedding = nn.Sequential(Embeddings(self.hidden_dim, self.out_token_count), copy.deepcopy(pos_encoding))
        
        # A Final Head:
        self.head = Generator(self.hidden_dim, self.output_dim)


        # PREVIOUSLY USED
        # For learnt positional embeddings
        # self.pos_arange = torch.arange(self.out_seq_len+self.inp_seq_len).unsqueeze(0)
        # self.attn_mask = self.generate_attn_mask()
        self.src_mask = torch.FloatTensor([[1],[1]])
        self.generate_source_mask(1, self.init_device)
        self.tgt_mask = subsequent_mask(self.out_seq_len)
        self.generate_target_mask(1, self.init_device)
        
        self.start_token = torch.LongTensor([[self.out_token_count - 1]]).to(self.init_device)
        # self.initialize_weights()
        self.apply(self.initialize_weights)
        # self.set_device(self.init_device)
        
    
    def forward(self, x_in, y_in, y_length):

        batch_size = x_in.shape[0]
        
        # pad y_in with the start token:
        self.generate_start_token(batch_size, x_in.device)
        y_in = torch.cat([self.start_token, y_in], 1)
        
        cnn_features = self.cnn_extractor.forward(x_in)
        cnn_f_pos = self.source_token_embedding_pos(cnn_features)
        
        self.generate_source_mask(batch_size, x_in.device)
        source_encoding = self.encoder(cnn_f_pos, self.src_mask)
        
        target_embeddings = self.target_token_embedding(y_in)
        
        self.generate_target_mask(batch_size, x_in.device)
        decoded_output = self.decoder(target_embeddings, source_encoding, self.src_mask, self.tgt_mask)
        if self.return_all:
            output = self.stack_vectors(decoded_output, y_length)
        else:
            output = self.vector_gather(decoded_output, y_length)
            if len(output.shape) == 3:
                output = output.squeeze(1)
        final_embedding = self.head(output)
        
        return final_embedding        

    def generate_source_mask(self, num, device):
        if not num == self.src_mask.shape[0]:
            sz = self.inp_seq_len
            self.src_mask = torch.ones(num, 1, sz).bool().to(device)
            
    def generate_target_mask(self, num, device):
        if not num == self.tgt_mask.shape[0]:
            mask = subsequent_mask(self.out_seq_len)
            self.tgt_mask = mask.repeat(num, 1, 1).to(device)
        
class FastTransformerExtractor(PLADTransformerExtractor):
    def __init__(self, observation_space, features_dim, config, dropout=0.0):
        
        super(PLADTransformerExtractor, self).__init__(observation_space, features_dim)
        # Need access to other specs
        
        ## Parameters:
        self.set_settings(config, dropout)
        max_length = self.inp_seq_len + self.out_seq_len
        
        self.cnn_extractor = Vox3DCNN(observation_space, self.hidden_dim, dropout, first_stride=self.cnn_first_stride)
        
        if self.pos_encoding_type == "FIXED":
            self.pos_encoding = FixedPositionalEncoding(self.hidden_dim, dropout)
        else:
            self.pos_encoding = LearnablePositionalEncoding(self.hidden_dim, dropout, max_len=max_length)
        self.token_embedding = nn.Embedding(self.out_token_count, self.hidden_dim) 

        self.attn_layers = nn.ModuleList([FastAttenLayer(self.num_heads, self.hidden_dim, self.dropout) for _ in range(self.num_dec_layers)])
        
        self.attn_to_output = nn.Linear(self.hidden_dim, self.output_dim)
        
        ## For Transformer:
        self.attn_mask = self.generate_attn_mask()
        self.attn_mask = self.attn_mask.to(self.init_device)
        
        self.key_mask = torch.FloatTensor([[1],[1]])
        self.generate_key_mask(1, self.init_device)
        self.key_mask = self.key_mask.to(self.init_device)
        
        self.start_token = torch.LongTensor([[self.out_token_count - 1]]).to(self.init_device)
    
        self.apply(self.initialize_weights)
    
    def forward(self, x_in, y_in, y_length):
        
        batch_size = x_in.shape[0]
        device = x_in.device
        
        self.generate_start_token(batch_size, x_in.device)
        y_in = torch.cat([self.start_token, y_in], 1)
        
        cnn_features = self.cnn_extractor.forward(x_in)
        token_embeddings = self.token_embedding(y_in)

        out = self.pos_encoding(torch.cat((cnn_features, token_embeddings), dim = 1))
        
        self.generate_key_mask(batch_size, x_in.device)
        
        # repeat at this point
        
        if self.return_all:
            all_outs = []
            all_masks = []
            for i in range(batch_size):
                # out_tensor = out[i:i+1].repeat(y_length[i], 1, 1)
                out_tensor = out[i:i+1].expand(y_length[i],-1, -1)
                out_mask = self.attn_mask[self.inp_seq_len:self.inp_seq_len + y_length[i]]
                all_outs.append(out_tensor)
                all_masks.append(out_mask)
            out = torch.cat(all_outs, 0)
            attn_mask = torch.cat(all_masks, 0)
        else:
            all_masks = []
            for i in range(batch_size):
                out_mask = self.attn_mask[self.inp_seq_len + y_length[i]:self.inp_seq_len + y_length[i]+1]
                all_masks.append(out_mask)
            attn_mask = torch.cat(all_masks, 0)
        
        for attn_layer in self.attn_layers:        
            out = attn_layer(out, attn_mask)
        seq_out = out[:,self.inp_seq_len:,:]
    
        if self.return_all:
            y_len_list = []
            for i in range(batch_size):
                ids = torch.arange(y_length[i]).to(device)
                y_len_list.append(ids)
            y_len_list = torch.cat(y_len_list)
            output = self.vector_gather(seq_out, y_len_list)
        else:
            output = self.vector_gather(seq_out, y_length)
            
        if len(output.shape) == 3:
            output = output.squeeze(1)
        output = self.attn_to_output(output)
        return output
    