# Load data from json.
# Get a score function
# Get a pacing function
from .restricted_env import RestrictedCSG
# from .csg2d.curriculum_generator import CurriculumGenerateData


class CurriculumCSG(RestrictedCSG):
    
    def __init__(self, config, phase_config, seed=0,
                 n_proc=1, proc_id=0):
        # load differently 
        super(CurriculumCSG, self).__init__(config, phase_config, 
                                            seed, n_proc, proc_id)
    
    def get_program_generator(self, config, phase_config):
        return CurriculumGenerateData(data_dir=phase_config.ENV.DATA_PATH, mode=phase_config.ENV.MODE, 
                                      n_proc=self.n_proc, proc_id=self.proc_id, train_proportion=config.TRAIN_PROPORTION, 
                                      unique_draw=self.unique_draw, program_lengths=self.program_lengths, 
                                      proportions=self.program_proportions, canvas_shape=self.canvas_shape, 
                                      sampling=phase_config.ENV.SAMPLING, pacing_function=phase_config.ENV.PACING_FUNCTION)
        
    