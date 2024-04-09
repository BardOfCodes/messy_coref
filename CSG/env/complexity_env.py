from .restricted_env import RestrictedCSG
# from .csg2d.cad_generator import CADGenerateData
# from .csg2d.instant_generator import InstantMixedGenerateData, RefactoredMixedGenerateData
from gym import spaces
import numpy as np
from .action_spaces import get_action_space

class ComplexityCSG(RestrictedCSG):

    def __init__(self, *args, **kwargs):
        super(ComplexityCSG, self).__init__(*args, **kwargs)
        image_space = spaces.Box(low=0, high=255,
                                    shape=(self.init_channels + self.observable_stack_size, self.canvas_shape[0],
                                            self.canvas_shape[0]), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "obs": image_space,
            "draw_allowed": spaces.Discrete(2),
            "op_allowed": spaces.Discrete(2),
            "stop_allowed" : spaces.Discrete(2),
            "complexity" : spaces.Discrete(10)
        })
        
        
        self.complexity_list = self.program_lengths# [x for x in range(max_complexity+1) if x>3 and x % 2 == 1]
        
        
        self.len_to_complexity_map = {
            x:i for i,x in enumerate(self.complexity_list)
        }
        
        self.complexity_to_len_map = {
            i:x for i,x in enumerate(self.complexity_list)
        }
        self.len_to_bin = {
            x:i//3 for i,x in enumerate(self.complexity_list)
            
        }
        
    def reset(self):
        obs_dict = super(ComplexityCSG,self).reset()
        self.target_len = len(self.target_expression)-1
        self.complexity = self.len_to_complexity_map[self.target_len] 
        obs_dict['draw_allowed'] = 1
        obs_dict['op_allowed'] = 1
        obs_dict['stop_allowed'] = 1
        obs_dict['complexity'] = self.complexity
        return obs_dict
    
    def step(self, action):
        # convert the output into action:
        done = True
        pred_len = self.complexity_to_len_map[action]
        if pred_len == self.target_len:
            reward = 1.0
        else:
            reward = 0.0
        
        obs_dict = {
            "obs": self.obs,
            "draw_allowed": 1,
            "op_allowed": 1,
            "stop_allowed": 1,
            "complexity" : self.complexity
            }
        
        
        info = {}
        info['target_expression'] = [self.target_len]
        info['predicted_expression'] = [pred_len]
        info['target_canvas'] = self.obs[0:1].copy()
        info['predicted_canvas'] = self.obs[1:2].copy()
        info['original_sequence'] = self.target_expression.copy()
        #  Get Larger Bin
        info['target_bin'] = [self.len_to_bin[self.target_len]]
        info['predicted_bin'] = [self.len_to_bin[pred_len]]
            
        return obs_dict, reward, done, info
    
class InstantComplexityCSG(ComplexityCSG):
    """
    Create Example on the fly with random size selected from size list.
    """
    def __init__(self, *args, **kwargs):
        
        self.generator_action_space = get_action_space("OpRestrictedAction")
        super(InstantComplexityCSG, self).__init__(*args, **kwargs)
        # TODO: Remove hardcode
        # Create CSG Dicts:
    def get_program_generator(self, config, phase_config):
        
        return InstantMixedGenerateData(action_space=self.generator_action_space, data_dir=config.MACHINE_SPEC.DATA_PATH, mode=phase_config.ENV.MODE, n_proc=self.n_proc, proc_id=self.proc_id,
                                                   train_proportion=config.TRAIN_PROPORTION, unique_draw=self.unique_draw,
                                                   program_lengths=self.program_lengths, proportions=self.program_proportions, 
                                                   canvas_shape=self.canvas_shape, sampling=phase_config.ENV.SAMPLING)      

class RefactoredComplexityCSG(ComplexityCSG):
    """
    Create Example on the fly with random size selected from size list.
    """
    def __init__(self, *args, **kwargs):
        
        self.generator_action_space = get_action_space("OpRestrictedAction")
        super(RefactoredComplexityCSG, self).__init__(*args, **kwargs)
        # TODO: Remove hardcode
        # Create CSG Dicts:
    def get_program_generator(self, config, phase_config):
        return RefactoredMixedGenerateData(action_space=self.generator_action_space, config=config.REFACTOR_CONFIG, data_dir=config.MACHINE_SPEC.DATA_PATH, mode=phase_config.ENV.MODE, n_proc=self.n_proc, proc_id=self.proc_id,
                                                   train_proportion=config.TRAIN_PROPORTION, unique_draw=self.unique_draw,
                                                   program_lengths=self.program_lengths, proportions=self.program_proportions, 
                                                   canvas_shape=self.canvas_shape, sampling=phase_config.ENV.SAMPLING)   
    
class CADComplexityCSG(ComplexityCSG):
    
    
    def __init__(self, *args, **kwargs):
        
        super(CADComplexityCSG, self).__init__(*args, **kwargs)
        # Load the program generator:
        self.gt_program = False

    def get_program_generator(self, config, phase_config):
        return CADGenerateData(data_dir=config.MACHINE_SPEC.DATA_PATH, mode=phase_config.ENV.MODE, unique_draw=self.unique_draw, n_proc=self.n_proc,
                               proc_id=self.proc_id, canvas_shape=self.canvas_shape, max_length=phase_config.ENV.CAD_MAX_LENGTH)
