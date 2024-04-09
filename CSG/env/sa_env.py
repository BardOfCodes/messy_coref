from ast import Expression
import time
import gym
from gym.spaces.discrete import Discrete
import numpy as np
import torch as th
import gym
import os
from gym import spaces
from numpy.lib.arraysetops import unique
from yacs.config import CfgNode
from .shape_assembly.data_generators import SAMixedGenerateData, SAShapeNetGenerateData
from .shape_assembly.state_machine import SAStateMachine
from .csg3d_env import CSG3DBase
from .restricted_env import RestrictedCSG
from .action_spaces import get_action_space
from .reward_function import Reward
import copy

class SA3DBase(CSG3DBase):


    # Check using 'human'
    metadata = {'render.modes': ['console']}

    def __init__(self, config: CfgNode, phase_config: CfgNode, seed=0, n_proc=1, proc_id=0):
        super(RestrictedCSG, self).__init__()
        self.seed(seed)
        self.program_lengths = phase_config.ENV.PROGRAM_LENGTHS
        self.program_proportions = phase_config.ENV.PROGRAM_PROPORTIONS
        
        self.action_space_type = config.ACTION_SPACE_TYPE
        self.canvas_shape = config.CANVAS_SHAPE
        self.reward = Reward(phase_config.ENV.REWARD)
        self.dynamic_max_len = phase_config.ENV.DYNAMIC_MAX_LEN
        sa_conf = phase_config.ENV.SA_CONF
        # RANDOM
        self.iter_counter = 0
        ## Fixed Temp.
        self.perm_max_len = sa_conf.PERM_MAX_LEN
        self.gt_program = True
        self.proc_id = proc_id
        self.n_proc = n_proc
        self.total_iter = config.TRAIN.NUM_STEPS / self.n_proc
        self.reset_on_done = False
        self.language_name = sa_conf.LANGUAGE_NAME
        self.max_expression_complexity = sa_conf.MAX_EXPRESSION_COMPLEXITY
        if self.language_name == "HSA3D":
            hierarchy_allowed = True
        else:
            hierarchy_allowed = False
        
        self.n_executions = 0

    # TODO: Create a reward class which takes these as input.
        
        action_specs = dict(resolution=config.ACTION_RESOLUTION, n_cuboid_ind_states=sa_conf.N_CUBOID_IND_STATES)
        self.action_space = get_action_space(self.action_space_type, **action_specs)
        # Load the program generator:
        self.program_generator = self.sa_program_generator(config, phase_config, self.action_space)

        self.action_sim = None
        # Generator for the first data
        # final output, draw board, stack
        state_machine_args = dict(master_min_prim=sa_conf.MASTER_MIN_PRIM, master_max_prim=sa_conf.MASTER_MAX_PRIM, 
                                  sub_min_prim=sa_conf.SUB_MIN_PRIM, sub_max_prim=sa_conf.SUB_MAX_PRIM, 
                                  hierarchy_allowed=hierarchy_allowed, max_sub_progs=sa_conf.MAX_SUB_PROGS,
                                  action_space=self.action_space, n_cuboid_ind_states=sa_conf.N_CUBOID_IND_STATES)
        self.state_machine = SAStateMachine(**state_machine_args)
        
        image_space = spaces.Box(low=0, high=255,
                                            shape=tuple(self.canvas_shape), dtype=np.float32)
        # TBD When figuring out StateMachine
        self.state_size = 11 + sa_conf.N_CUBOID_IND_STATES + 2
        self.observation_space = spaces.Dict({
            "obs": image_space,
            "state": spaces.MultiDiscrete([2] * self.state_size),
            "previous_steps" : spaces.MultiDiscrete([self.action_space.n]* (self.perm_max_len)),
            "cur_step" : spaces.Discrete(self.perm_max_len),
        })
        
        obs_size = self.canvas_shape
        self.obs = np.zeros(obs_size, dtype=np.float32)
        self.empty_info_dict = {
            'target_expression' : self.program_generator.parser.trivial_expression.copy(),
            'predicted_expression' : self.program_generator.parser.trivial_expression.copy(),
            'target_canvas': image_space.sample() * 0,
            'predicted_canvas' : image_space.sample() * 0,
            'log_prob' : 0,
            'target_id': 0,
            'slot_id': 0
            
        }

    def sa_program_generator(self, config, phase_config, action_space):
        return SAMixedGenerateData(data_dir=config.MACHINE_SPEC.DATA_PATH, mode=phase_config.ENV.MODE, n_proc=self.n_proc, proc_id=self.proc_id,
                                                   proportion=config.TRAIN_PROPORTION, program_lengths=self.program_lengths, sa_config=phase_config.ENV.SA_CONF,
                                                   proportions=self.program_proportions, sampling=phase_config.ENV.SAMPLING,
                                                   set_loader_limit=phase_config.ENV.SET_LOADER_LIMIT, loader_limit=phase_config.ENV.LOADER_LIMIT,
                                                   project_root=config.MACHINE_SPEC.PROJECT_ROOT, action_space=action_space)

    def reset_from_info(self, info):
        
        self.target_expression = info['target_expression']
        self.target_actions = info['target_actions']
        target_canvas = info['target_canvas']
        
        self.pred_actions = []
        self.active_program = []
        self.pred_expression = []
        
        self.target_id = info['target_id']
        self.slot_id = info['slot_id']
        # if self.dynamic_max_len:
        #     self.boolean_count = self.program_generator.boolean_count(self.target_expression)
        # reinit observation
        # Initially empty stack
        self.obs *= 0
        self.obs_tensor = th.tensor(self.obs, device=self.program_generator.compiler.device, dtype=self.program_generator.compiler.tensor_type)
        self.obs = target_canvas
        self.time_step = 0
        self.state_machine.reset()
        state = self.state_machine.get_state()
        self.previous_steps = np.zeros(self.observation_space['previous_steps'].shape, dtype=np.int64)

        obs_dict = {
            "obs": self.obs.copy(),
            "state": state.copy(),
            "previous_steps" : self.previous_steps.copy(),
            "cur_step" : 0,
        }
        self.action_space.set_state(state)


        # Temporary
        # self.test_state_machine()
        return obs_dict


    def step(self, action):
        done = False
        self.iter_counter += 1
        
        self.pred_actions.append(action)
        # self.action_space.validate_action(action)
        cur_step = len(self.pred_actions)
        if self.action_space.is_stop_action(action):
            done = True

        # after action
        self.time_step += 1
        if self.time_step == (self.perm_max_len + 1):
            done = True
            # self.pred_actions = [64, 0, 74]
            # print("Reached limit!!")
        reward = 0.0
        if done:
            # Get pred from compiling the program:
            # print(self.pred_actions)
            # print(self.pred_expression)
            self.state_machine.reset()
            if self.time_step == (self.perm_max_len + 1):
                print("Expression rejected due to being too long. Lengh = %d" % len(self.pred_actions))
                # print(self.pred_actions)
                # pred_actions = self.program_generator.parser.trivial_actions
                self.pred_expression = self.program_generator.parser.trivial_expression.copy()
                self.pred_actions = self.action_space.expression_to_action(self.program_generator.parser.trivial_expression)
                # self.action_space.action_to_expression(np.array(pred_actions))
                self.pred_canvas = self.obs_tensor.clone()# self.obs.copy() * 0
                reward = -0.1
                self.state_machine.reset()
                action = 0
            else:
                # if self.reset_on_done:
                #     self.obs_tensor, expression = self.program_generator.get_executed_program(self.slot_id, self.target_id, return_numpy=False, return_bool=True)
                    # self.obs_tensor = th.tensor(self.obs, device=self.program_generator.compiler.device, dtype=self.program_generator.compiler.tensor_type)
                self.pred_expression = self.action_space.action_to_expression(np.array(self.pred_actions))
                

                command_list = self.program_generator.parser.parse(self.pred_expression)
                complexity = self.program_generator.compiler._get_complexity(command_list)
                acceptable_complexity = complexity <= self.max_expression_complexity

                if acceptable_complexity:
                    # self.n_executions += 1
                    # self.pred_canvas = self.program_generator.execute(self.pred_expression, return_numpy=False, return_bool=True)
                    # print("num_executions", self.n_executions)
                    # self.pred_canvas = self.obs.copy() * 0
                    # reward = self.reward.iou_3d(self.pred_canvas, self.obs_tensor, done, self.pred_expression)
                    R = 0.1# th.logical_and(self.pred_canvas, self.obs_tensor).sum()/th.logical_or(self.pred_canvas, self.obs_tensor).sum()
                    reward = R# .item()
                else:
                    print("Expression Rejected becuase too complex. Complexity = %d" % complexity)
                    self.pred_expression = self.program_generator.parser.trivial_expression.copy()
                    self.pred_actions = self.action_space.expression_to_action(self.program_generator.parser.trivial_expression)
                    self.pred_canvas = self.obs_tensor.clone()# self.obs.copy() * 0
                    reward = -0.1
                
        # Check what actions are possible: 
        self.state_machine.update_state(action)
        state = self.state_machine.get_state()
        self.previous_steps[(cur_step-1) : cur_step] = action
        obs_dict = {
            "obs": self.obs,
            "state": state,
            "previous_steps":self.previous_steps.copy(),
            "cur_step" : cur_step,
        }
        self.action_space.set_state(state)        
        # print(self.slot_id, self.target_id)
        info = {}
        if done:
            
            info['target_expression'] = self.target_expression.copy()
            info['predicted_expression'] = self.pred_expression.copy()
            info['target_canvas'] = self.obs.copy()
            info['predicted_canvas'] = self.obs.copy()
            info['target_id'] = self.target_id 
            info['slot_id'] = self.slot_id 
            info['log_prob'] = self.log_prob
            info['reward'] = reward
        return obs_dict, reward, done, info

    def get_internal_state(self):
        internal_state = super(CSG3DBase, self).get_internal_state()
        internal_state['previous_steps'] = np.copy(self.previous_steps)
        internal_state['target_actions'] = np.copy(self.target_actions)
        internal_state['pred_actions'] = self.pred_actions.copy()
        action_state = self.state_machine.get_internal_state()
        internal_state['action_state'] = copy.deepcopy(action_state)
        return internal_state

    
    def get_minimal_internal_state(self):
        internal_state = dict(target_expression=self.target_expression.copy(),
                              time_step=self.time_step,
                              slot_id=self.slot_id,
                              target_id=self.target_id)

        internal_state['previous_steps'] = np.copy(self.previous_steps)
        internal_state['target_actions'] = np.copy(self.target_actions)
        internal_state['pred_actions'] = self.pred_actions.copy()
        action_state = self.state_machine.get_internal_state()
        internal_state['action_state'] = action_state# copy.deepcopy(action_state)
        return internal_state

    def set_minimal_internal_state(self, internal_state):
        self.target_expression = internal_state['target_expression'].copy()
        self.time_step = internal_state['time_step']
        self.slot_id = internal_state['slot_id']
        self.target_id = internal_state['target_id']
        
        self.previous_steps =  np.copy(internal_state['previous_steps'])
        self.target_actions =  np.copy(internal_state['target_actions'])
        self.pred_actions = internal_state['pred_actions'].copy()
        action_state = internal_state['action_state']# copy.deepcopy(internal_state['action_state'])
        self.state_machine.set_internal_state(action_state)

        # self.action_space.set_state(self.state_machine.state)

        
    def set_internal_state(self, internal_state):
        super(CSG3DBase, self).set_internal_state(internal_state)

        self.previous_steps = internal_state['previous_steps']
        self.target_actions = internal_state['target_actions']
        self.pred_actions = internal_state['pred_actions']
        action_state = copy.deepcopy(internal_state['action_state'])
        self.state_machine.set_internal_state(action_state)
        # self.action_space.set_state(self.state_machine.state)
        

    def test_state_machine(self):
        
        self.program_generator.set_execution_mode(th.device("cuda"), th.float16)
        self.reset()
        state = self.state_machine.get_state()
        self.action_space.set_state(state)
        self.pred_actions = []
        while(True):
            done = False
            # print(state)
            action = self.action_space.sample()
            self.action_space.validate_action(action)
            # print(action)
            self.pred_actions.append(action)   
            self.state_machine.update_state(action)
            state = self.state_machine.get_state()
            self.action_space.set_state(state) 
            action_len = len(self.pred_actions)
            if action_len == (self.perm_max_len + 1):
                done = True
                self.pred_actions = self.action_space.expression_to_action(self.program_generator.parser.trivial_expression)

            if action == self.action_space.stop_action:
                done = True
            if done:
                print(self.pred_actions)
                expression = self.action_space.action_to_expression(np.array(self.pred_actions))
                for expr in expression:
                    print(expr)
                n_cuboids = len([x for x in expression if "cuboid(" in x])
                print("%d actions and %d cuboids"% (len(self.pred_actions), n_cuboids))
                self.state_machine.reset()
                state = self.state_machine.get_state()
                self.action_space.set_state(state) 
                self.reset()

class SA3DShapeNet(SA3DBase):
    def __init__(self, *args, **kwargs):
        
        super(SA3DShapeNet, self).__init__(*args, **kwargs)
        # Load the program generator:
        self.gt_program = False

    def sa_program_generator(self, config, phase_config, *args, **kwargs):
        return SAShapeNetGenerateData(data_dir=config.MACHINE_SPEC.DATA_PATH, mode=phase_config.ENV.MODE, n_proc=self.n_proc, proc_id=self.proc_id,
                                                   proportion=config.TRAIN_PROPORTION, program_lengths=self.program_lengths, sa_config=phase_config.ENV.SA_CONF,
                                                   proportions=self.program_proportions, sampling=phase_config.ENV.SAMPLING,
                                                   project_root=config.MACHINE_SPEC.PROJECT_ROOT)

class SA3DBaseBC(SA3DBase):
    
    def __init__(self, *args, **kwargs):
        super(SA3DBaseBC, self).__init__(*args, **kwargs)
        
        self.observation_space = spaces.Dict({
            "obs": self.observation_space['obs'],
            "state": spaces.MultiDiscrete([2] * self.state_size),
            "previous_steps" : spaces.MultiDiscrete([self.action_space.n]* (self.perm_max_len)),
            "cur_step" : spaces.Discrete(self.perm_max_len),
            "target" : spaces.MultiDiscrete([self.action_space.n]* (self.perm_max_len)),
        })
        self.default_target = self.observation_space['target'].sample() * 0
        

    def reset(self):
        obs_dict = super(SA3DBaseBC, self).reset()
        # Do all moves: 
        num_steps = len(self.target_actions)
        obs_dict['cur_step'] = num_steps
        # target = self.target_actions# [num_steps]
        target = self.observation_space['target'].sample() * 0
        target[:num_steps] = self.target_actions
        obs_dict['target'] = target.copy()
        obs_dict['previous_steps'] = target.copy()
        
        return obs_dict 
    
    
    def minimal_reset(self):

        data, expression, slot_id, target_id = self.program_generator.get_next_sample(return_numpy=False)
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

class SA3DShapeNetBC(SA3DShapeNet):    
    
    def __init__(self, *args, **kwargs):
        super(SA3DShapeNetBC, self).__init__(*args, **kwargs)
        
        self.observation_space = spaces.Dict({
            "obs": self.observation_space['obs'],
            "state": spaces.MultiDiscrete([2] * self.state_size),
            "previous_steps" : spaces.MultiDiscrete([self.action_space.n]* (self.perm_max_len)),
            "cur_step" : spaces.Discrete(self.perm_max_len),
            "target" : spaces.MultiDiscrete([self.action_space.n]* (self.perm_max_len)),
        })
        self.default_target = self.observation_space['target'].sample() * 0
        
        

    def reset(self):
        obs_dict = super(SA3DShapeNetBC, self).reset()        
        # Do all moves: 
        num_steps = len(self.target_actions)
        obs_dict['cur_step'] = num_steps
        target = self.observation_space['target'].sample() * 0
        target[:num_steps] = self.target_actions
        obs_dict['target'] = target.copy()
        obs_dict['previous_steps'] = target.copy()

        return obs_dict 
    
    def minimal_reset(self):
        data, expression, slot_id, target_id = self.program_generator.get_next_sample(return_numpy=False)
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
