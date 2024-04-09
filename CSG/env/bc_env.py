
from gym import spaces
from .restricted_env import RestrictedCSG
from .action_spaces import MULTI_ACTION_SPACE
import numpy as np
from collections import defaultdict


class BCRestrictedCSG(RestrictedCSG):

    def __init__(self, *args, **kwargs):
        super(BCRestrictedCSG, self).__init__(*args, **kwargs)
        image_space = spaces.Box(low=0, high=255,
                                    shape=(self.init_channels + self.observable_stack_size, self.canvas_shape[0],
                                            self.canvas_shape[0]), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "obs": image_space,
            "draw_allowed": spaces.Discrete(2),
            "op_allowed": spaces.Discrete(2),
            "stop_allowed" : spaces.Discrete(2),
            "target" : self.action_space
        })

    def reset(self):
        self.time_step = 0
        self.active_program = []
        self.action_sim.reset()
        # reinit observation
        self.obs *= 0
        obs_dict, reward, done, info = self.get_random_state()
        # self.info = info
        # print(self.slot_id, self.target_id, self.temp, self.target_action)
        return obs_dict 
    
    def get_random_state(self):
        """
        Fetch a state and run k steps
        """
        # Initialize the agent at the right of the grid
        done = False
        self.iter_counter += 1
        cur_frac = float(self.iter_counter/self.total_iter)
        data, labels, slot_id, target_id = self.program_generator.get_next_sample(cur_frac)
        self.slot_id = slot_id
        self.target_id = target_id
        self.target_expression = self.action_space.label_to_expression(labels, self.unique_draw)
            
        self.obs[0] = data
        
        # Perform K steps:
        num_steps = int(np.random.sample() *  len(self.target_expression))
        cur_target_expression = self.target_expression[num_steps]
        # for RNN
        self.num_steps = num_steps
        done_steps = self.target_expression[:num_steps]
        
        program = [self.parser.parse(x) for x in done_steps]
        
        for ind, prog_step in enumerate(program):
            self.action_sim.generate_stack(prog_step, start_scratch=False)
            if self.canvas_slate:
                # Transfer program output to slate at channel 1.
                if self.action_space.is_operation(done_steps[ind]):
                    # operation - print output to active canvas
                    self.obs[1] = self.action_sim.stack.items[0].copy()
            # after action
            self.time_step += 1
        stack_obs = self.action_sim.stack.get_items()[:self.observable_stack_size]
        self.obs[self.init_channels:self.init_channels + stack_obs.shape[0]] = stack_obs
        # self.temp = cur_target_expression

        self.target_action = self.action_space.expression_to_action(cur_target_expression, self.unique_draw)      
        # Check if the action is correctly converted
        # re_target_expression = self.action_space.action_to_expression(target_action, self.unique_draw) 
        # if re_target_expression != re_target_expression:
        #     raise ValueError
        # target_action = target_action[num_steps, None]
        
        
        
        if self.dynamic_max_len:
            self.max_len = len(self.target_expression) -1
        # Check what actions are possible: 
        stack_size = self.action_sim.stack.size()
        draw_allowed, op_allowed, stop_allowed = self.get_possible_action_state(stack_size)
            
        obs_dict = {
            "obs": self.obs.copy(),
            "draw_allowed": np.array([draw_allowed]),
            "op_allowed": np.array([op_allowed]),
            "stop_allowed": np.array([stop_allowed]),
            'target': self.target_action.copy(),
        }
        self.action_space.set_state(draw_allowed, op_allowed, stop_allowed)

        # if self.time_step == self.max_len:
        #     done = True  
        # target = self.obs[0:1]
        # pred = self.obs[1:2]
        # reward = self.reward(target, pred, done)
        reward = 0
        done = False
        # Check if valid: Removed for speed.
        # Mainly for evaluation
        info = {}# {'target_expression': self.target_expression}
            
        return obs_dict, reward, done, info

class RNNBCRestrictedCSG(BCRestrictedCSG):
    
    def __init__(self, *args, **kwargs):
        super(BCRestrictedCSG, self).__init__(*args, **kwargs)
        self.perm_max_len = self.max_len
        image_space = spaces.Box(low=0, high=255,
                                    shape=(self.init_channels + self.observable_stack_size, self.canvas_shape[0],
                                            self.canvas_shape[0]), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "obs": image_space,
            "draw_allowed": spaces.Discrete(2),
            "op_allowed": spaces.Discrete(2),
            "stop_allowed" : spaces.Discrete(2),
            "target" : self.action_space,
            "previous_steps" : spaces.MultiDiscrete(MULTI_ACTION_SPACE * (self.perm_max_len+1)),
            "cur_step" : spaces.Discrete(self.perm_max_len+1),
        })
        self.action_step_size = len(MULTI_ACTION_SPACE)
        
    def reset(self):
        obs_dict = super(RNNBCRestrictedCSG, self).reset()
        # Now add the remaining:
        done_steps = self.target_expression[:self.num_steps]
        self.previous_steps = self.observation_space['previous_steps'].sample() * 0
        for ind, step in enumerate(done_steps):
            self.previous_steps[ind * self.action_step_size : (ind + 1) * self.action_step_size ] = self.action_space.expression_to_action(step, self.unique_draw)
        # convert it into previous_steps array:
        obs_dict['previous_steps'] = self.previous_steps.copy()
        obs_dict['cur_step'] = self.num_steps
        return obs_dict
    