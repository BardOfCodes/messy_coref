from ast import Expression
import gym
from gym.spaces.discrete import Discrete
import numpy as np
import gym
import os
from gym import spaces
from numpy.lib.arraysetops import unique
from yacs.config import CfgNode
from .csg2d.data_generators import MixedGenerateData2D
from .csg2d.data_generators import ShapeNetGenerateData2D
import torch as th
from .csg3d import languages
from .restricted_env import RestrictedCSG
from .action_spaces import get_action_space
from .reward_function import Reward
from .csg3d_env import CSG3DBase, CSG3DBaseBC
from .csg3d_shapenet_env import CSG3DShapeNet, CSG3DShapeNetBC
import copy

class CSG2DBase(CSG3DBase):

    
    
        
    def get_program_generator(self, config, phase_config):
        return MixedGenerateData2D(data_dir=config.MACHINE_SPEC.DATA_PATH, mode=phase_config.ENV.MODE, n_proc=self.n_proc, proc_id=self.proc_id,
                                                   proportion=config.TRAIN_PROPORTION, program_lengths=self.program_lengths, csg_config=phase_config.ENV.CSG_CONF,
                                                   proportions=self.program_proportions, sampling=phase_config.ENV.SAMPLING,
                                                   set_loader_limit=phase_config.ENV.SET_LOADER_LIMIT, loader_limit=phase_config.ENV.LOADER_LIMIT,
                                                   project_root=config.MACHINE_SPEC.PROJECT_ROOT, action_space=self.action_space)
    @property
    def slot_target_id(self):
        return str(self.slot_id) + "_" + str(self.target_id)
    
        
class CSG2DBaseBC(CSG3DBaseBC):   
    def __init__(self, *args, **kwargs):
        
        super(CSG2DBaseBC, self).__init__(*args, **kwargs)
        # Load the program generator:
        self.gt_program = True
        
    def get_program_generator(self, config, phase_config):
        return MixedGenerateData2D(data_dir=config.MACHINE_SPEC.DATA_PATH, mode=phase_config.ENV.MODE, n_proc=self.n_proc, proc_id=self.proc_id,
                                                   proportion=config.TRAIN_PROPORTION, program_lengths=self.program_lengths, csg_config=phase_config.ENV.CSG_CONF,
                                                   proportions=self.program_proportions, sampling=phase_config.ENV.SAMPLING,
                                                   set_loader_limit=phase_config.ENV.SET_LOADER_LIMIT, loader_limit=phase_config.ENV.LOADER_LIMIT,
                                                   project_root=config.MACHINE_SPEC.PROJECT_ROOT, action_space=self.action_space)



class CSG2DShapeNet(CSG3DShapeNet):
    
    def __init__(self, *args, **kwargs):
        
        super(CSG2DShapeNet, self).__init__(*args, **kwargs)
        # Load the program generator:
        self.gt_program = False

    def get_program_generator(self, config, phase_config):
        return ShapeNetGenerateData2D(data_dir=config.MACHINE_SPEC.DATA_PATH, mode=phase_config.ENV.MODE, n_proc=self.n_proc, proc_id=self.proc_id,
                                                   proportion=config.TRAIN_PROPORTION, program_lengths=self.program_lengths, csg_config=phase_config.ENV.CSG_CONF,
                                                   proportions=self.program_proportions, sampling=phase_config.ENV.SAMPLING, 
                                                   project_root=config.MACHINE_SPEC.PROJECT_ROOT)

class CSG2DShapeNetBC(CSG2DShapeNet):    
    
    def __init__(self, *args, **kwargs):
        super(CSG2DShapeNetBC, self).__init__(*args, **kwargs)
        
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
        obs_dict = super(CSG2DShapeNetBC, self).reset()
        
        
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