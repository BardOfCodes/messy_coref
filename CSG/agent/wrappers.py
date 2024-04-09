
from stable_baselines3.common.torch_layers import CombinedExtractor
from .feature_extractors import (BaseExtractor, ConvCoordExtractor, 
                                 LargeConvCoordExtractor,
                                 RNNConvCoordExtractor,
                                 NoStackConvCoordExtractor,
                                 LargeNoStackConvCoordExtractor,
                                 RNNNoStackConvCoordExtractor,
                                 ReplCNN,
                                 MultiRNNConvCoordExtractor)
from .transformers import transformer_dict
class WrapperBaseExtractor(CombinedExtractor):
    
    def __init__(self, observation_space, features_dim, *args, **kwargs):

        super(WrapperBaseExtractor, self).__init__(
            observation_space, features_dim)
        obs = observation_space['obs']

        self.extractor = BaseExtractor(obs, features_dim)

    def forward(self, obs_in):

        obs = obs_in['obs']
        features = self.extractor(obs)
        return features


class WrapperConvCoordExtractor(WrapperBaseExtractor):

    def __init__(self, observation_space, features_dim, *args, **kwargs):

        super(WrapperBaseExtractor, self).__init__(
            observation_space, features_dim)
        obs = observation_space['obs']

        # self.extractor = ConvCoordExtractor(obs, features_dim)
        self.extractor = LargeConvCoordExtractor(obs, features_dim)


class WrapperReplCNNExtractor(WrapperBaseExtractor):

    def __init__(self, observation_space, features_dim, *args, **kwargs):

        super(WrapperBaseExtractor, self).__init__(
            observation_space, features_dim)
        obs = observation_space['obs']

        self.extractor = ReplCNN(obs, features_dim)

class WrapperNoStackConvCoordExtractor(WrapperBaseExtractor):

    def __init__(self, observation_space, features_dim, *args, **kwargs):

        super(WrapperBaseExtractor, self).__init__(
            observation_space, features_dim)
        obs = observation_space['obs']

        self.extractor = LargeNoStackConvCoordExtractor(obs, features_dim)
        
        
class WrapperRNNConvCoordExtractor(WrapperBaseExtractor):

    def __init__(self, observation_space, features_dim, *args, **kwargs):

        super(WrapperBaseExtractor, self).__init__(
            observation_space, features_dim)
        obs = observation_space['obs']
        
        self.extractor = RNNNoStackConvCoordExtractor(obs, features_dim)

    def forward(self, obs_in):

        obs = obs_in['obs']
        y_in = obs_in['previous_steps']
        y_length = obs_in['cur_step']
        features = self.extractor(obs, y_in, y_length)
        return features
    
    
    
class WrapperTransformerExtractor(WrapperRNNConvCoordExtractor):

    def __init__(self, observation_space, features_dim, config, dropout, *args, **kwargs):

        super(WrapperBaseExtractor, self).__init__(observation_space, features_dim)
        obs = observation_space['obs']
        transformer_class = transformer_dict[config.TYPE]
        self.extractor = transformer_class(obs, features_dim, config)
