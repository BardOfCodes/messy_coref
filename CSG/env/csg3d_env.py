from ast import Expression
import gym
from gym.spaces.discrete import Discrete
import numpy as np
import gym
import os
from gym import spaces
from numpy.lib.arraysetops import unique
from yacs.config import CfgNode
from .csg3d.mixed_len_generator import MixedGenerateData
import torch as th
from .csg3d import languages
from .restricted_env import RestrictedCSG
from .action_spaces import get_action_space
from .reward_function import Reward
import copy

class CSG3DBase(RestrictedCSG):
    
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
        # RANDOM
        self.iter_counter = 0
        ## Fixed Temp.
        self.boolean_count = phase_config.ENV.CSG_CONF.BOOLEAN_COUNT
        self.perm_max_len = phase_config.ENV.CSG_CONF.PERM_MAX_LEN
        self.gt_program = True
        self.proc_id = proc_id
        self.n_proc = n_proc
        self.total_iter = config.TRAIN.NUM_STEPS / self.n_proc
        self.reset_on_done = False
        self.language_name = phase_config.ENV.CSG_CONF.LANG_TYPE
        self.execute_on_done = True
        self.max_expression_complexity = phase_config.ENV.CSG_CONF.MAX_EXPRESSION_COMPLEXITY

    # TODO: Create a reward class which takes these as input.
        

        self.action_sim = None
        # Generator for the first data
        # final output, draw board, stack
        action_specs = dict(resolution=config.ACTION_RESOLUTION,
                            valid_draws=phase_config.ENV.CSG_CONF.VALID_DRAWS,
                            valid_transforms=phase_config.ENV.CSG_CONF.VALID_TRANFORMS,
                            valid_booleans=phase_config.ENV.CSG_CONF.VALID_BOOL)
        self.action_space = get_action_space(self.action_space_type, **action_specs)
        state_machine_class = languages.language_map[phase_config.ENV.CSG_CONF.LANG_TYPE]['state_machine']
        # Load the program generator:
        self.program_generator = self.get_program_generator(config, phase_config)
        self.state_machine = state_machine_class(self.boolean_count, self.action_space)
        
        image_space = spaces.Box(low=0, high=255,
                                            shape=tuple(self.canvas_shape), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "obs": image_space,
            "numbers_allowed": spaces.Discrete(2),
            "ptype_allowed": spaces.Discrete(2),
            "transform_allowed": spaces.Discrete(2),
            "bool_allowed": spaces.Discrete(2), 
            "stop_allowed": spaces.Discrete(2),
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
        
        
    def get_program_generator(self, config, phase_config):
        return MixedGenerateData(data_dir=config.MACHINE_SPEC.DATA_PATH, mode=phase_config.ENV.MODE, n_proc=self.n_proc, proc_id=self.proc_id,
                                                   proportion=config.TRAIN_PROPORTION, program_lengths=self.program_lengths, csg_config=phase_config.ENV.CSG_CONF,
                                                   proportions=self.program_proportions, sampling=phase_config.ENV.SAMPLING,
                                                   set_loader_limit=phase_config.ENV.SET_LOADER_LIMIT, loader_limit=phase_config.ENV.LOADER_LIMIT,
                                                   project_root=config.MACHINE_SPEC.PROJECT_ROOT, action_space=self.action_space)
    @property
    def slot_target_id(self):
        return str(self.slot_id) + "_" + str(self.target_id)
    
    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        # Initialize the agent at the right of the grid
        # self.iter_counter +=  1 #Since each will be run twice
        cur_frac = float(self.iter_counter/self.total_iter)
        data, expression, slot_id, target_id = self.program_generator.get_next_sample(cur_frac)
        self.target_id = target_id
        self.slot_id = slot_id
        self.log_prob = 0
        
        target_actions = self.action_space.expression_to_action(expression)
        self.target_expression = self.action_space.action_to_expression(target_actions)
        info = {
            'target_canvas': data,
            'target_actions': target_actions,
            'target_expression': expression,
            'target_id': self.target_id,
            'slot_id': self.slot_id
        }
        obs_dict = self.reset_from_info(info)

        return obs_dict
    
    def reset_to_target(self, slot_id, target_id):
        data, expression = self.program_generator.get_executed_program(slot_id, target_id)
        self.target_id = target_id
        self.slot_id = slot_id
        self.log_prob = 0
        
        target_actions = self.action_space.expression_to_action(expression)
        self.target_expression = self.action_space.action_to_expression(target_actions)
        info = {
            'target_canvas': data,
            'target_actions': target_actions,
            'target_expression': self.target_expression,
            'target_id': self.target_id,
            'slot_id': self.slot_id
        }
        obs_dict = self.reset_from_info(info)
        return obs_dict
    
    
    def reset_to_target_ids(self, slot_id_list=None, target_id_list=None):
        obs_list, internal_state_list = [], []
        for ind, target_id in enumerate(target_id_list): 
            if slot_id_list:
                slot = slot_id_list[ind]
            else:
                slot = self.program_generator.get_next_slot()
            data, expression = self.program_generator.get_executed_program(slot, target_id)
            self.target_id = target_id
            self.slot_id = slot
            self.log_prob = 0
            
            target_actions = self.action_space.expression_to_action(expression)
            self.target_expression = self.action_space.action_to_expression(target_actions)
            info = {
                'target_canvas': data,
                'target_actions': target_actions,
                'target_expression': self.target_expression,
                'target_id': self.target_id,
                'slot_id': self.slot_id
            }
            obs_dict = self.reset_from_info(info)
            obs_list.append(obs_dict.copy())
            # internal_state_list.append(self.get_internal_state())
            internal_state_list.append(self.get_minimal_internal_state())
        return obs_list, internal_state_list
    
    def reset_to_eval_target(self, slot_id, target_id):
        data, expression = self.program_generator.get_executed_program(slot_id, target_id)
        self.target_id = target_id
        self.slot_id = slot_id
        self.log_prob = 0
        
        target_actions = self.action_space.expression_to_action(expression)
        self.target_expression = self.action_space.action_to_expression(target_actions)
        info = {
            'target_canvas': data,
            'target_actions': target_actions,
            'target_expression': self.target_expression,
            'target_id': self.target_id,
            'slot_id': self.slot_id
        }
        obs_dict = self.reset_from_info(info)
        return obs_dict
    
    def reset_from_info(self, info):
        
        self.target_expression = info['target_expression']
        self.target_actions = info['target_actions']
        target_canvas = info['target_canvas']
        
        self.pred_actions = []
        self.active_program = []
        self.pred_expression = []
        
        self.target_id = info['target_id']
        self.slot_id = info['slot_id']
        if self.dynamic_max_len:
            self.boolean_count = self.program_generator.boolean_count(self.target_expression)
        # reinit observation
        # Initially empty stack
        self.obs *= 0
        self.obs = target_canvas
        self.time_step = 0
        self.state_machine.reset()
        n_a, p_a, t_a, b_a, s_a = self.state_machine.get_state()
        self.previous_steps = np.zeros(self.observation_space['previous_steps'].shape, dtype=np.int64)
        obs_dict = {
            "obs": self.obs.copy(),
            "numbers_allowed": n_a,
            "ptype_allowed": p_a,
            "transform_allowed": t_a,
            "bool_allowed": b_a,
            "stop_allowed": s_a,
            "previous_steps" : self.previous_steps.copy(),
            "cur_step" : 0,
        }
        self.action_space.set_state(n_a, p_a, t_a, b_a, s_a)

        # Temporary
        # self.test_state_machine()
        return obs_dict
    
    def step(self, action):
        done = False
        self.iter_counter += 1
        self.pred_actions.append(action)
          
        cur_step = len(self.pred_actions)
        if self.action_space.is_action_stop(action):
            # Called stop
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
                self.pred_canvas = self.obs.copy() * 0
                reward = -0.1
            else:
                if self.reset_on_done:
                    self.obs, expression = self.program_generator.get_executed_program(self.slot_id, self.target_id)
                self.pred_expression = self.action_space.action_to_expression(np.array(self.pred_actions))
                
                command_list = self.program_generator.parser.parse(self.pred_expression)
                complexity = self.program_generator.compiler._get_complexity(command_list)
                acceptable_complexity = complexity <= self.max_expression_complexity

                if acceptable_complexity:
                    self.pred_canvas = self.program_generator.execute(self.pred_expression, return_numpy=True)
                    # reward = self.reward(self.pred_canvas, self.obs, done, self.pred_expression)
                    reward = 0.1
                else:
                    print("Expression Rejected becuase too complex. Complexity = %d" % complexity)
                    self.pred_expression = self.program_generator.parser.trivial_expression.copy()
                    self.pred_actions = self.action_space.expression_to_action(self.program_generator.parser.trivial_expression)
                    self.pred_canvas = self.obs.copy() * 0
                    reward = -0.1
                
        # Check what actions are possible: 
        self.state_machine.update_state(action)
        n_a, p_a, t_a, b_a, s_a = self.state_machine.get_state()
        self.previous_steps[(cur_step-1) : cur_step] = action
        obs_dict = {
            "obs": self.obs,
            "numbers_allowed": n_a,
            "ptype_allowed": p_a,
            "transform_allowed": t_a,
            "bool_allowed": b_a,
            "stop_allowed": s_a,
            "previous_steps":self.previous_steps.copy(),
            "cur_step" : cur_step,
        }
        self.action_space.set_state(n_a, p_a, t_a, b_a, s_a)        
        # print(self.slot_id, self.target_id)
        info = {}
        if done:
            
            info['target_expression'] = self.target_expression.copy()
            info['predicted_expression'] = self.pred_expression.copy()
            info['target_canvas'] = self.obs.copy()
            info['predicted_canvas'] = self.pred_canvas.copy()
            info['target_id'] = self.target_id 
            info['slot_id'] = self.slot_id 
            info['log_prob'] = self.log_prob
            info['reward'] = reward
            
        return obs_dict, reward, done, info

    def test_state_machine(self):
        
        n_a, p_a, t_a, b_a, s_a = self.state_machine.get_state()
        self.action_space.set_state(n_a, p_a, t_a, b_a, s_a)
        self.pred_actions = []
        while(True):
            # print(n_a, p_a, t_a, b_a, s_a)
            action = self.action_space.sample()
            # print(action)
            self.pred_actions.append(action)   
            self.state_machine.update_state(action)
            n_a, p_a, t_a, b_a, s_a = self.state_machine.get_state()
            self.action_space.set_state(n_a, p_a, t_a, b_a, s_a) 
            action_len = len(self.pred_actions)
            done = False
            if action_len == (self.perm_max_len + 1):
                # pred_actions = self.program_generator.parser.trivial_actions
                self.pred_expression = self.program_generator.parser.trivial_expression.copy()
                self.pred_actions = self.action_space.expression_to_action(self.pred_expression)
                done = True
            if np.sum([n_a, p_a, t_a, b_a, s_a]) == 0:
                done = True
            if done:
                # print(self.pred_actions)
                expression = self.action_space.action_to_expression(np.array(self.pred_actions))
                for expr in expression:
                    print(expr)
                n_bools = len([x for x in expression if x in ["union", "difference", "intersection"]])
                print("%d actions and %d bools"% (len(self.pred_actions), n_bools))
                self.state_machine.reset()
                n_a, p_a, t_a, b_a, s_a = self.state_machine.get_state()
                self.action_space.set_state(n_a, p_a, t_a, b_a, s_a) 
                self.reset()
            
    def get_internal_state(self):
        internal_state = super(CSG3DBase, self).get_internal_state()
        internal_state['previous_steps'] = np.copy(self.previous_steps)
        internal_state['target_actions'] = np.copy(self.target_actions)
        internal_state['pred_actions'] = self.pred_actions.copy()
        action_state = self.state_machine.get_internal_state()
        internal_state['action_state'] = copy.deepcopy(action_state)
        return internal_state
    
    def get_minimal_internal_state(self):
        # internal_state = super(CSG3DBase, self).get_internal_state()
        internal_state = dict(target_expression=self.target_expression.copy(),
                              time_step=self.time_step,
                              slot_id=self.slot_id,
                              target_id=self.target_id)

        internal_state['previous_steps'] = np.copy(self.previous_steps)
        internal_state['target_actions'] = np.copy(self.target_actions)
        internal_state['pred_actions'] = self.pred_actions.copy()
        action_state = self.state_machine.get_internal_state()
        internal_state['action_state'] = copy.deepcopy(action_state)
        return internal_state
    
    
    def set_minimal_internal_state(self, internal_state):

        self.target_expression = internal_state['target_expression'].copy()
        self.time_step = internal_state['time_step']
        self.slot_id = internal_state['slot_id']
        self.target_id = internal_state['target_id']
        
        self.previous_steps =  np.copy(internal_state['previous_steps'])
        self.target_actions =  np.copy(internal_state['target_actions'])
        self.pred_actions = internal_state['pred_actions'].copy()
        action_state = copy.deepcopy(internal_state['action_state'])
        self.state_machine.set_internal_state(action_state)
        
    def set_internal_state(self, internal_state):
        super(CSG3DBase, self).set_internal_state(internal_state)

        self.previous_steps = internal_state['previous_steps'].copy()
        self.target_actions = internal_state['target_actions'].copy()
        self.pred_actions = internal_state['pred_actions'].copy()
        action_state = copy.deepcopy(internal_state['action_state'])
        self.state_machine.set_internal_state(action_state)
        
class CSG3DBaseBC(CSG3DBase):    
    
    def __init__(self, *args, **kwargs):
        super(CSG3DBaseBC, self).__init__(*args, **kwargs)
        
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
        obs_dict = super(CSG3DBaseBC, self).reset()
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

