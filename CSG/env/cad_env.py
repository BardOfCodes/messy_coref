from .restricted_env import RestrictedCSG
# from .csg2d.cad_generator import CADGenerateData
from .action_spaces import MULTI_ACTION_SPACE
from gym import spaces
import numpy as np

class CADCSG(RestrictedCSG):
    
    def __init__(self, *args, **kwargs):
        
        super(CADCSG, self).__init__(*args, **kwargs)
        # Load the program generator:
        self.gt_program = False

    def get_program_generator(self, config, phase_config):
        return CADGenerateData(data_dir=config.MACHINE_SPEC.DATA_PATH, mode=phase_config.ENV.MODE, unique_draw=self.unique_draw, n_proc=self.n_proc,
                               proc_id=self.proc_id, canvas_shape=self.canvas_shape, max_length=phase_config.ENV.CAD_MAX_LENGTH)

class RNNCADCSG(CADCSG):
    
    def __init__(self, *args, **kwargs):
        
        super(RNNCADCSG, self).__init__(*args, **kwargs)
        # Load the program generator:
        self.perm_max_len = self.max_len
        
        image_space = spaces.Box(low=0, high=255,
                                    shape=(self.init_channels + self.observable_stack_size, self.canvas_shape[0],
                                            self.canvas_shape[0]), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "obs": image_space,
            "draw_allowed": spaces.Discrete(2),
            "op_allowed": spaces.Discrete(2),
            "stop_allowed" : spaces.Discrete(2),
            "stop_allowed" : spaces.Discrete(2),
            "previous_steps" : spaces.MultiDiscrete(MULTI_ACTION_SPACE * (self.perm_max_len+1)),
            "cur_step" : spaces.Discrete(self.perm_max_len + 1),
        })
        self.previous_steps = self.observation_space['previous_steps'].sample() * 0
        self.action_step_size = len(MULTI_ACTION_SPACE)

    def reset(self):
        obs_dict = super(RNNCADCSG, self).reset()
        # label_list = [400]
        self.previous_steps = self.observation_space['previous_steps'].sample() * 0
        # label_list += [0,] * (self.perm_max_len - len(self.active_program))
        obs_dict['previous_steps'] = self.previous_steps
        obs_dict['cur_step'] = 0
        return obs_dict
    
    def step(self, action):
        obs_dict, reward, done, info = super(RNNCADCSG, self).step(action)
        cur_step = len(self.active_program)
        obs_dict['cur_step'] = cur_step
        
        self.previous_steps[(cur_step-1) * self.action_step_size : cur_step * self.action_step_size] = action
        
        obs_dict['previous_steps'] = self.previous_steps.copy()
        
        return obs_dict, reward, done, info
        