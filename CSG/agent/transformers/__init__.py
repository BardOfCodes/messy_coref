from .transformers import PLADTransformerExtractor, DefaultTransformerExtractor, FastTransformerExtractor
from .trans_vae import PLADTransVAE
transformer_dict = dict(PLADTransformerExtractor=PLADTransformerExtractor,
                        PLADTransVAE=PLADTransVAE,
                        DefaultTransformerExtractor=DefaultTransformerExtractor,
                        FastTransformerExtractor=FastTransformerExtractor)