# from .restricted_env import RestrictedCSG
from .csg3d_env import CSG3DBase
from .csg3d.shapenet_generator import ShapeNetGenerateData
from gym import spaces
import numpy as np
import torch as th

class CSG3DShapeNet(CSG3DBase):
    
    def __init__(self, *args, **kwargs):
        
        super(CSG3DShapeNet, self).__init__(*args, **kwargs)
        # Load the program generator:
        self.gt_program = False

    def get_program_generator(self, config, phase_config):
        return ShapeNetGenerateData(data_dir=config.MACHINE_SPEC.DATA_PATH, mode=phase_config.ENV.MODE, n_proc=self.n_proc, proc_id=self.proc_id,
                                                   proportion=config.TRAIN_PROPORTION, program_lengths=self.program_lengths, csg_config=phase_config.ENV.CSG_CONF,
                                                   proportions=self.program_proportions, sampling=phase_config.ENV.SAMPLING, 
                                                   project_root=config.MACHINE_SPEC.PROJECT_ROOT)

class CSG3DShapeNetBC(CSG3DShapeNet):    
    
    def __init__(self, *args, **kwargs):
        super(CSG3DShapeNetBC, self).__init__(*args, **kwargs)
        
        self.observation_space = spaces.Dict({
            "obs": self.observation_space['obs'],
            "numbers_allowed": spaces.Discrete(2),
            "ptype_allowed": spaces.Discrete(2),
            "transform_allowed": spaces.Discrete(2),
            "bool_allowed": spaces.Discrete(2),
            "stop_allowed": spaces.Discrete(2),
            "previous_steps" : spaces.MultiDiscrete([self.action_space.n]* (self.perm_max_len)),
            "cur_step" : spaces.Discrete(self.perm_max_len),
            "target" : spaces.MultiDiscrete([self.action_space.n]* (self.perm_max_len)),
        })
        self.default_target = self.observation_space['target'].sample() * 0
        
        

    def reset(self):
        obs_dict = super(CSG3DShapeNetBC, self).reset()
        
        
        # Do all moves: 
        num_steps = len(self.target_actions)
        obs_dict['cur_step'] = num_steps
        target = self.observation_space['target'].sample() * 0
        target[:num_steps] = self.target_actions
        obs_dict['target'] = target.copy()
        obs_dict['previous_steps'] = target.copy()

        return obs_dict 
    
    def minimal_reset(self):

        data, expression, slot_id, target_id = self.program_generator.get_next_sample(0, return_numpy=False)
        self.target_id = target_id
        self.slot_id = slot_id
        self.log_prob = 0
        target_actions = self.action_space.expression_to_action(expression)
        num_steps = len(target_actions)
        target = np.copy(self.default_target)
        target[:num_steps] = target_actions
        target = th.from_numpy(target).to("cuda", non_blocking=True)
        num_steps = th.tensor(num_steps).to("cuda", non_blocking=True)
        obs_dict = {
            'obs': data,
            'previous_steps': target,
            'target': target,
            'cur_step': num_steps,
        }
        return obs_dict