
import gym
from gym.spaces.discrete import Discrete
import numpy as np
import gym
import os
from gym import spaces
from numpy.lib.arraysetops import unique
from yacs.config import CfgNode
from .action_spaces import MULTI_ACTION_SPACE


from collections import defaultdict
# from .csg2d.mixed_len_generator import MixedGenerateData
# from .csg2d.stacks import SimulateStack, MemorySimulateStack
# from .csg2d.parsers import labels2exps, Parser
# from .csg2d.generator_utils import program_validity
from .reward_function import Reward
from .action_spaces import get_action_space
from .reward_function import chamfer
import networkx as nx

ALLOWED_PROGRAM_LENGTHS = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21] # 10 for cad
  
    
def return_list(*args):
    return []

    
class RestrictedCSG(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left. 
    """

    # Check using 'human'
    metadata = {'render.modes': ['console']}

    def __init__(self, config: CfgNode, phase_config: CfgNode, seed=0, n_proc=1, proc_id=0):
        super(RestrictedCSG, self).__init__()
        # Set a random seed
        self.seed(seed)
        self.program_lengths = phase_config.ENV.PROGRAM_LENGTHS
        self.program_proportions = phase_config.ENV.PROGRAM_PROPORTIONS
        
        self.observable_stack_size = config.OBSERVABLE_STACK
        self.action_space_type = config.ACTION_SPACE_TYPE
        self.canvas_shape = config.CANVAS_SHAPE
        self.canvas_slate = config.CANVAS_SLATE
        self.reward = Reward(phase_config.ENV.REWARD)
        self.dynamic_max_len = phase_config.ENV.DYNAMIC_MAX_LEN
        self.max_len = np.max(self.program_lengths)
        self.max_stack_len = self.max_len//2+1
        self.iter_counter = 0
        self.gt_program = True
        self.proc_id = proc_id
        self.n_proc = n_proc
        self.total_iter = config.TRAIN.NUM_STEPS / self.n_proc

        # TODO: Create a reward class which takes these as input.

        # Basic checks
        for p_len in self.program_lengths:
            assert p_len in ALLOWED_PROGRAM_LENGTHS, \
                "Programs of size %d are not present in dataset." % p_len

        assert len(self.program_lengths) == len(self.program_proportions), \
            "Program-Proportions must be of the same size as the Program-Lengths list."

        assert len(self.canvas_shape) == 2, "Canvas shape is wrongly specified"

        # Load the terminals symbols of the grammar
        with open(config.MACHINE_SPEC.TERMINAL_FILE, "r") as file:
            self.unique_draw = file.readlines()
        for index, e in enumerate(self.unique_draw):
            self.unique_draw[index] = e[0:-1]
        
        # Load the program generator:
        self.program_generator = self.get_program_generator(config, phase_config)

        # Generator for the first data
        # final output, drawboard, stack
        self.init_channels = 1
        if self.canvas_slate:
            self.init_channels += 1
          
        self.action_space = get_action_space(self.action_space_type)
  
            
        image_space = spaces.Box(low=0, high=255,
                                            shape=(self.init_channels + self.observable_stack_size, self.canvas_shape[0],
                                                   self.canvas_shape[0]), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "obs": image_space,
            "draw_allowed": spaces.Discrete(2),
            "op_allowed": spaces.Discrete(2),
            "stop_allowed" : spaces.Discrete(2)
        })

        self.parser = Parser()
        if config.USE_MEMORY_STACK:
            self.action_sim = MemorySimulateStack(self.max_stack_len, self.canvas_shape, self.unique_draw)
        else:
            self.action_sim = SimulateStack(self.max_stack_len, self.canvas_shape, self.unique_draw)
        obs_size = [self.init_channels + self.observable_stack_size] + self.canvas_shape
        self.obs = np.zeros(obs_size, dtype=np.float32)

    def get_program_generator(self, config, phase_config, *args, **kwargs):
        return MixedGenerateData(data_dir=config.MACHINE_SPEC.DATA_PATH, mode=phase_config.ENV.MODE, n_proc=self.n_proc, proc_id=self.proc_id,
                                                   proportion=config.TRAIN_PROPORTION, unique_draw=self.unique_draw,
                                                   program_lengths=self.program_lengths, proportions=self.program_proportions, 
                                                   canvas_shape=self.canvas_shape, sampling=phase_config.ENV.SAMPLING, use_memory_stack=config.USE_MEMORY_STACK)
    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        # Initialize the agent at the right of the grid
        # self.iter_counter +=  1 #Since each will be run twice
        cur_frac = float(self.iter_counter/self.total_iter)
        data, labels, slot_id, target_id = self.program_generator.get_next_sample(cur_frac)
        self.target_id = target_id
        self.slot_id = slot_id
        self.log_prob = 0
        
        target_expression = self.action_space.label_to_expression(labels, self.unique_draw)
        info = {
            'target_canvas': data,
            'target_expression': target_expression,
            'target_id': self.target_id,
            'slot_id': self.slot_id
        }
        obs_dict = self.reset_from_info(info)
        return obs_dict

    def reset_to_target(self, slot_id, target_id):
        data, labels = self.program_generator.get_executed_program(slot_id, target_id)
        self.target_id = target_id
        self.slot_id = slot_id
        self.log_prob = 0
        
        target_expression = self.action_space.label_to_expression(labels, self.unique_draw)
        info = {
            'target_canvas': data,
            'target_expression': target_expression,
            'target_id': self.target_id,
            'slot_id': self.slot_id
        }
        obs_dict = self.reset_from_info(info)
        return obs_dict
    
    def reset_to_eval_target(self, slot_id, target_id):
        data, labels = self.program_generator.get_executed_program_by_IDS(slot_id, target_id)
        self.target_id = target_id
        self.slot_id = slot_id
        self.log_prob = 0
        
        target_expression = labels # self.action_space.label_to_expression(labels, self.unique_draw)
        info = {
            'target_canvas': data,
            'target_expression': target_expression,
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
            data, labels = self.program_generator.get_executed_program(slot, target_id)
            self.target_id = target_id
            self.slot_id = slot
            self.log_prob = 0
            
            target_expression = self.action_space.label_to_expression(labels, self.unique_draw)
            info = {
                'target_canvas': data,
                'target_expression': target_expression,
                'target_id': self.target_id,
                'slot_id': self.slot_id
            }
            obs_dict = self.reset_from_info(info)
            obs_list.append(obs_dict.copy())
            internal_state_list.append(self.get_internal_state())
        return obs_list, internal_state_list
    
    def generate_observations(self, slot_id, target_id, actions):
        obs_list = []
        obs_dict = self.reset_to_target(slot_id, target_id)
        obs_list.append(obs_dict)
        for cur_action in actions:
            next_obs, reward, done, info = self.step(cur_action[0]) 
            obs_list.append(next_obs)
        
        final_obs = defaultdict(return_list)
        # The last is just after done
        for obs in obs_list[:-1]:
            for key, value in obs.items():
                final_obs[key].append(value)
        
        for key, value in final_obs.items():
            value = np.stack(value, 0)
            if len(value.shape) ==1:
                 final_obs[key] = value[:, None]
            
        return final_obs
        
        
    def reset_from_info(self, info):
        
        self.target_expression = info['target_expression']
        target_canvas = info['target_canvas']
        self.active_program = []
        self.pred_expression = []
        self.target_id = info['target_id']
        self.slot_id = info['slot_id']
        self.action_sim.reset()
        self.action_sim.generate_stack(self.active_program)
        if self.dynamic_max_len:
            self.max_len = len(self.target_expression) - 1
        # reinit observation
        # Initially empty stack
        self.obs *= 0
        self.obs[0] = target_canvas
        self.time_step = 0
        obs_dict = {
            "obs": self.obs.copy(),
            "draw_allowed": 1,
            "op_allowed": 0,
            "stop_allowed": 0
        }
        self.action_space.set_state(1, 0, 0)
        return obs_dict
    
    def step(self, action):
        # convert the output into action:
        done = False
        self.iter_counter += 1
        
        # Check if the action is correctly converted
        expression = self.action_space.action_to_expression(action, self.unique_draw)
        # Convert action into a effect on stack
        # action = np.expand_dims(action, axis=0)
        program = self.parser.parse(expression)
        temp_program = self.active_program + program
        if not program_validity(temp_program, self.max_len, self.time_step):
            # invalid program
            print(temp_program)
            import pdb
            pdb.set_trace()
            raise Exception("Invalid Program!! Solve by agent design.")
        
        self.pred_expression.append(expression)
        
        if self.action_space.is_stop(expression):
            # Called stop
            done = True
        else:
            self.active_program.extend(program)
            self.action_sim.generate_stack(program, start_scratch=False)
            if self.canvas_slate:
                # Transfer program output to slate at channel 1.
                if self.action_space.is_operation(expression):
                    # operation - print output to active canvas
                    self.obs[1] = self.action_sim.stack.get_items()[0]
            stack_obs = self.action_sim.stack.get_items()[:self.observable_stack_size]
            self.obs[self.init_channels:self.init_channels + stack_obs.shape[0]] = stack_obs

        # after action
        self.time_step += 1
        if self.time_step == (self.max_len+1):
            done = True  
        
        target = self.obs[0:1]
        pred = self.obs[1:2]
        reward = self.reward(pred, target, done, self.pred_expression)
        
        # Check what actions are possible: 
        stack_size = self.action_sim.stack.size()
        draw_allowed, op_allowed, stop_allowed = self.get_possible_action_state(stack_size)
            
        obs_dict = {
            "obs": self.obs.copy(),
            "draw_allowed": draw_allowed,
            "op_allowed": op_allowed,
            "stop_allowed": stop_allowed
        }
        self.action_space.set_state(draw_allowed, op_allowed, stop_allowed)            
        
        info = {}
        if done:
            
            info['target_expression'] = self.target_expression.copy()
            info['predicted_expression'] = self.pred_expression.copy()
            info['target_canvas'] = self.obs[0:1].copy()
            info['predicted_canvas'] = self.obs[1:2].copy()
            info['target_id'] = self.target_id 
            info['slot_id'] = self.slot_id 
            info['log_prob'] = self.log_prob
            
        return obs_dict, reward, done, info
        
    def get_possible_action_state(self, stack_size):        
        if stack_size >= 2:
            op_allowed = 1
        else:
            op_allowed = 0
            
        min_required_ops = stack_size -1 
        if min_required_ops < (self.max_len - self.time_step):
            draw_allowed = 1
        else:
            draw_allowed = 0
            
        if self.time_step >= 3 and stack_size == 1:
            stop_allowed = 1
        else:
            stop_allowed = 0
        return draw_allowed, op_allowed, stop_allowed
    
    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        pass
        # print("." * self.agent_pos, end="")
    
    def visualize_expression(self, info, use_gt=False):
        exps = {
                'predicted_tree': info['predicted_expression']
        }
        if self.gt_program or use_gt:
            exps['target_tree'] = info['target_expression']
        progs = {x: self.parser.parse_full_expression(y) for x, y in exps.items()}
        vis = {}
        for name, progs in progs.items():
            # Generate the stack
            self.action_sim.reset()
            tree_vis = self.action_sim.generate_tree_visualization(progs)
            vis[name] = tree_vis
            self.action_sim.reset()
        return vis
            
    def close(self):
        pass

    def get_internal_state(self):
        # get the internal state of the env:
        
        internal_state = dict(target_expression=self.target_expression.copy(),
                              active_program=self.active_program.copy(),
                              pred_expression=self.pred_expression.copy(),
                              obs=self.obs.copy(), time_step=self.time_step,
                              max_len=self.max_len, slot_id=self.slot_id,
                              target_id=self.target_id)
        return internal_state

    def set_internal_state(self, internal_state):
        self.target_expression = internal_state['target_expression'].copy()
        self.active_program = internal_state['active_program'].copy()
        self.pred_expression = internal_state['pred_expression'].copy()
        self.obs = internal_state['obs'].copy()
        self.time_step = internal_state['time_step']
        self.max_len = internal_state['max_len']
        self.slot_id = internal_state['slot_id']
        self.target_id = internal_state['target_id']
        
        if self.action_sim:
            self.action_sim.reset()
            self.action_sim.generate_stack(self.active_program)


    def bulk_exec(self, internal_state, action_set):
        # first set state of env:
        # TODO: Bulk execute without resetting each time.
        new_obs = []
        new_internal_states = []
        new_rewards = []
        new_dones = []
        new_infos = []
        action_set = action_set[0]
        action_set = [x.item() for x in action_set]
        # I don't need to do it entirely - simply one step plus loss can help
        for ind, action in enumerate(action_set):
            self.set_internal_state(internal_state)
            observations, rewards, dones, infos = self.step(action)
            new_internal_state = self.get_internal_state()
            new_obs.append(observations)
            new_internal_states.append(new_internal_state)
            new_rewards.append(rewards)
            new_dones.append(dones)
            new_infos.append(infos)
        return new_obs, new_internal_states, new_rewards, new_dones, new_infos
            
class RNNRestrictedCSG(RestrictedCSG):
    
    def __init__(self, *args, **kwargs):
        super(RNNRestrictedCSG, self).__init__(*args, **kwargs)
        image_space = spaces.Box(low=0, high=255,
                                    shape=(self.init_channels + self.observable_stack_size, self.canvas_shape[0],
                                            self.canvas_shape[0]), dtype=np.float32)
        self.perm_max_len = self.max_len # The last has to be stop symbol? 
        self.observation_space = spaces.Dict({
            "obs": image_space,
            "draw_allowed": spaces.Discrete(2),
            "op_allowed": spaces.Discrete(2),
            "stop_allowed" : spaces.Discrete(2),
            "stop_allowed" : spaces.Discrete(2),
            "previous_steps" : spaces.MultiDiscrete(MULTI_ACTION_SPACE * (self.perm_max_len+1)),
            "cur_step" : spaces.Discrete(self.perm_max_len+1),
        })
        self.action_step_size = len(MULTI_ACTION_SPACE)
    
    def reset(self):
        obs_dict = super(RNNRestrictedCSG, self).reset()
        
        self.previous_steps = self.observation_space['previous_steps'].sample() * 0
        # label_list += [0,] * (self.perm_max_len - len(self.active_program))
        obs_dict['previous_steps'] = self.previous_steps
        obs_dict['cur_step'] = 0
        return obs_dict
    
    def step(self, action):
        obs_dict, reward, done, info = super(RNNRestrictedCSG, self).step(action)
        cur_step = len(self.active_program)
        obs_dict['cur_step'] = cur_step
        
        self.previous_steps[(cur_step-1) * self.action_step_size : cur_step * self.action_step_size] = action
        
        obs_dict['previous_steps'] = self.previous_steps.copy()
        
        return obs_dict, reward, done, info
        
        
    
    def reset_to_eval_target(self, slot_id, target_id):
        obs_dict = super(RNNRestrictedCSG, self).reset_to_eval_target(slot_id, target_id)
        self.previous_steps = self.observation_space['previous_steps'].sample() * 0
        obs_dict['previous_steps'] = self.previous_steps
        obs_dict['cur_step'] = 0
        return obs_dict