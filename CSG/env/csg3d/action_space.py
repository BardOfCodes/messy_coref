from os import stat
import torch as th
import torch
import numpy as np

from gym.spaces.discrete import Discrete

from .parser import draw_commands, boolean_commands, transform_commands, macro_commands, fixed_macro_commands
from .constants import (ROTATE_MULTIPLIER, SCALE_ADDITION, TRANSLATE_MIN, TRANSLATE_MAX, 
                        SCALE_MIN, SCALE_MAX, ROTATE_MIN, ROTATE_MAX, DRAW_MIN, DRAW_MAX, CONVERSION_DELTA)

class MCSGAction3D(Discrete):

    def __init__(self, resolution=64):

        ## N = res + here stands for
        # 1) Numbers = Resolution = RES
        # 4) Primitive Types = 3
        # 2) Transforms = 3
        # 3) Booleans = 3
        # 4) Macro(MIRROR) = 1
        # 5) Fixed Macro(macro) = 3
        # 5) Stop Symbol
        self.resolution = resolution
        # count of different actions
        self.action_types = {
            "NUMERIC": resolution,
            "DRAW": 3,
            "TRANSFORM": 3,
            "BOOL": 3,
            "MACRO": 1,
            "FIXED_MACRO": 3,
            "STOP": 1 
        }
        self.command_index = {
            'sphere': self.resolution, 
            'cylinder': self.resolution + 1, 
            'cuboid': self.resolution + 2, 
            'translate': self.resolution + 3,
            'rotate': self.resolution + 4, 
            'scale': self.resolution + 5,
            'union': self.resolution + 6, 
            'intersection': self.resolution + 7, 
            'difference': self.resolution + 8,
            'mirror': self.resolution + 9,
            'macro(MIRROR_X)': self.resolution + 10,
            'macro(MIRROR_Y)': self.resolution + 11,
            'macro(MIRROR_Z)': self.resolution + 12,
            '$': self.resolution + 13,
            ## ADD MACROS HERE
        } 
        #Tuple for (Type of command, number of continuous parameters)
        self.index_to_command = {x:(0, 0) for x in range(self.resolution)} 
        self.index_to_command.update(
            {
                self.resolution: (1, 0),
                self.resolution + 1: (1, 0),
                self.resolution + 2: (1, 0),
                # self.resolution + 3: (1, 3),
                self.resolution + 3: (2, 3),
                self.resolution + 4: (2, 3),
                self.resolution + 5: (2, 3),
                self.resolution + 6: (3, 0),
                self.resolution + 7: (3, 0),
                self.resolution + 8: (3, 0),
                ## FUNCTIONALLY EQUIVALENT TO "TRANFORMS"
                self.resolution + 9: (2, 3),
                self.resolution + 10: (2, 0),
                self.resolution + 11: (2, 0),
                self.resolution + 12: (2, 0),
                self.resolution + 13: (4, 0)
            }
        )
        self.stop_action = self.resolution + 13

        self.init_state()
        self.init_action_limits()

    

    def init_state(self):

        total_dim = np.sum(list(self.action_types.values()))
        super(MCSGAction3D, self).__init__(total_dim)
        self.index_to_expr = {}
        for key, value in self.command_index.items():
            self.index_to_expr[value] = key
        for key in range(self.resolution):
            self.index_to_expr[key] = "N"

        self.numbers_allowed = 0
        self.ptype_allowed = 0
        self.transform_allowed = 0
        self.bool_allowed = 1
        self.stop_allowed = 0

        self.conversion_delta = CONVERSION_DELTA

        self.two_scale_delta = (2 - 2 * self.conversion_delta)/(self.resolution - 1)

        self.mid_point = self.resolution//2
        
        self.stop_expression = "$"
        self.reduction_value = th.FloatTensor([float('-inf')])
        self.zero_value = th.FloatTensor([0])
        
    
    def create_action_softmax_and_mask(self, numeric_sigma=1, discrete_delta=1):
        
        action_softmatrix = discrete_delta * np.eye(self.n, dtype=np.float32)
        mask_softmatrix = np.zeros([self.n, self.n], dtype=np.float32)
        
        for key, value in self.action_types.items():
            min_action, max_action = self.action_limits[key]
            mask_softmatrix[min_action:max_action, min_action:max_action] = 1
            action_softmatrix[min_action:max_action, min_action:max_action] += 1
        # For numeric values create gaussian
        numeric_softmatrix = np.arange(-self.resolution, 0) + 1
        numeric_softmatrix = np.repeat(numeric_softmatrix[None, :], self.resolution, 0)
        
        additional = np.arange(0, self.resolution)[::-1]
        additional = np.repeat(additional[:, None], self.resolution, 1)
        numeric_softmatrix += additional
        
        numeric_softmatrix = np.exp(-np.power(numeric_softmatrix, 2.) / (2 * np.power(numeric_sigma, 2.)))
        key = "NUMERIC"
        min_action, max_action = self.action_limits[key]
        
        action_softmatrix[min_action:max_action, min_action:max_action] = numeric_softmatrix
        
        action_softmatrix = np.log(action_softmatrix)
        
        return action_softmatrix, mask_softmatrix
        
        

    def init_action_limits(self):

        # Now action type and range of actions:
        count = 0
        self.action_limits = {}
        self.action_values = {}
        for key, value in self.action_types.items():
            self.action_limits[key] = (count, count + value)
            self.action_values[key] = list(range(count, count + value))
            count += value  

    
    def set_state(self, numbers_allowed, ptype_allowed, transform_allowed, bool_allowed, stop_allowed):
        self.numbers_allowed = numbers_allowed
        self.ptype_allowed = ptype_allowed
        self.transform_allowed = transform_allowed
        self.bool_allowed = bool_allowed
        self.stop_allowed = stop_allowed

    def get_state(self):
        return self.numbers_allowed, self.ptype_allowed, self.transform_allowed, self.bool_allowed, self.stop_allowed

    @staticmethod
    def is_draw(expression):
        bool_list = [x in expression for x in draw_commands]
        return any(bool_list)

    @staticmethod
    def is_bool(expression):
        bool_list = [x in expression for x in boolean_commands]
        return any(bool_list)
    
    @staticmethod
    def is_transform(expression):
        bool_list = [x in expression for x in transform_commands]
        return any(bool_list)

    @staticmethod
    def is_macro(expression):
        bool_list = [x in expression for x in macro_commands]
        return any(bool_list)
    @staticmethod
    def is_fixed_macro(expression):
        bool_list = [x in expression for x in fixed_macro_commands]
        return any(bool_list)
        
    @staticmethod
    def is_operation(expression):
        raise ValueError("This action Space should not do this")
        
    @staticmethod
    def is_stop(expression):
        return expression == "$"
    
    def is_action_ptype(self, action):
        lims = self.action_limits["DRAW"]
        return lims[0] <= action < (lims[1])
    
    def is_action_transform(self, action):
        lims = self.action_limits["TRANSFORM"]
        return lims[0] <= action < (lims[1])
    
    def is_action_bool(self, action):
        lims = self.action_limits["BOOL"]
        return lims[0] <= action < (lims[1])

    def is_action_macro(self, action):
        lims = self.action_limits["MACRO"]
        return lims[0] <= action < (lims[1])
    
    def is_action_stop(self, action):
        lims = self.action_limits["STOP"]
        return lims[0] <= action < (lims[1])
     
    @staticmethod
    def get_permissions(obs, expand_dims=False, to_bool=False):
        dummy = obs['numbers_allowed']
        items = [obs['numbers_allowed'], obs['ptype_allowed'], obs['transform_allowed'],
                 obs['bool_allowed'], obs['stop_allowed']]
        if not len(dummy.shape) == 1:
            items =  [x[:,0] for x in items]
        if expand_dims:
            items = [x[:, None] for x in items]
        if to_bool:
            items = [x.bool() for x in items]
        n_a, p_a, t_a, b_a, s_a = items
        return n_a, p_a, t_a, b_a, s_a

    def expression_to_action(self, expression_list):
        action_list = []
        for expr in expression_list:
            action_list.extend(self.single_expression_to_action(expr))
        action_list = np.array(action_list, dtype=np.int32)
        return action_list

    def single_expression_to_action(self, expr):
        command_symbol = expr.split("(")[0]
        if command_symbol in boolean_commands:
            action_list = [self.command_index[command_symbol]]
        elif command_symbol in draw_commands:
            cmd_ind = self.command_index[command_symbol]
            action_list = [self.command_index[command_symbol], ] 
        elif command_symbol in transform_commands:
            param_str = expr.split("(")[1][:-1]
            param = np.array([float(x.strip()) for x in param_str.split(",")])
            if command_symbol == "translate":
                param = np.clip(param, TRANSLATE_MIN + self.conversion_delta, TRANSLATE_MAX - self.conversion_delta)
                # param = [self.mid_point + np.round(x/self.two_scale_delta) for x in param]
            elif command_symbol == "rotate":
                param = np.clip(param, ROTATE_MIN + self.conversion_delta, ROTATE_MAX - self.conversion_delta)
                param = param / ROTATE_MULTIPLIER
                # param = [self.mid_point + np.round(x/180./self.two_scale_delta) for x in param]
            elif command_symbol == 'scale':
                param = np.clip(param, SCALE_MIN + self.conversion_delta, SCALE_MAX - self.conversion_delta)
                param -= SCALE_ADDITION
                # param = [self.mid_point + np.round( (x-1.25)/self.two_scale_delta) for x in param]
            param = (param - (-1 + self.conversion_delta)) / self.two_scale_delta
            param = np.round(param).astype(np.uint32)
            action_list = [self.command_index[command_symbol], ] + list(param)
        elif command_symbol == "$":
            action_list = [self.command_index[command_symbol]]
        elif command_symbol in macro_commands:
            param_str = expr.split("(")[1][:-1]
            param = np.array([float(x.strip()) for x in param_str.split(",")])
            param = np.clip(param, TRANSLATE_MIN + self.conversion_delta, TRANSLATE_MAX - self.conversion_delta)
            param = (param - (-1 + self.conversion_delta)) / self.two_scale_delta
            param = np.round(param).astype(np.uint32)
            action_list = [self.command_index[command_symbol], ] + list(param)
        elif command_symbol in fixed_macro_commands:
            # Just retrieve from the expression
            action_list = [self.command_index[expr], ]
        return action_list

    def action_to_expression(self, actions):
        size_ = actions.shape[0]
        pointer = 0
        expression_list = []
        while(pointer < size_):
            cur_command = actions[pointer]
            cur_expr = self.index_to_expr[cur_command]
            n_param = self.index_to_command[cur_command][1]
            if n_param > 0:
                # Has to be for transform or mirror
                param = np.array(actions[pointer+1: pointer + 1 + n_param])
                if cur_expr == "translate":
                    param = -1 + self.conversion_delta + (param * self.two_scale_delta)
                elif cur_expr == "rotate":
                    param = (-1 + self.conversion_delta + (param * self.two_scale_delta)) * ROTATE_MULTIPLIER
                elif cur_expr == "scale":
                    param = -1 + self.conversion_delta + (param * self.two_scale_delta) + SCALE_ADDITION
                elif cur_expr == "mirror":
                    param = -1 + self.conversion_delta + (param * self.two_scale_delta)
                
                param_str = ", ".join(["%f" % x for x in param])
                cur_expr = "%s(%s)" %(cur_expr, param_str)
                pointer += n_param
            expression_list.append(cur_expr)
            pointer += 1
                
        return expression_list

    def get_conditional_entropy(self, distribution):
        return distribution.entropy()
    
    def get_log_prob(self, distribution, actions, obs):
        log_p = distribution.log_prob(actions)
        return log_p

    def get_all_log_prob(self, distribution, actions):
        log_prob = distribution.log_prob(actions)
        return log_prob

    def get_max_action(self, distribution):
        _, max_actions = th.max(distribution.distribution.logits, 1)
        return max_actions
    
    def sample(self):
        allowed_keys = []
        if self.numbers_allowed:
            allowed_keys.append("NUMERIC")
        if self.ptype_allowed:
            allowed_keys.append("DRAW")
        if self.transform_allowed:
            allowed_keys.append("TRANSFORM")
            allowed_keys.append("MACRO")
            allowed_keys.append("FIXED_MACRO")
        if self.bool_allowed:
            allowed_keys.append("BOOL")
        if self.stop_allowed:
            allowed_keys.append("STOP")
        sample_list = []
        for key in allowed_keys:
            lims = self.action_limits[key]
            new_vals = list(range(lims[0], lims[1]))
            sample_list.extend(new_vals)
        if not sample_list:
            raise Exception("No action allowed!")
        output = np.random.choice(sample_list)
        return output
    
    def get_checklist(self, n_a, p_a, t_a, b_a, s_a):
        check_list =  {
            "NUMERIC": n_a,
            "DRAW": p_a,
            "TRANSFORM": t_a,
            "BOOL": b_a,
            "MACRO": t_a,
            "FIXED_MACRO": t_a,
            "STOP": s_a 
        }
        return check_list
    
    def restrict_pred_action(self, prediction, obs):
        n_a, p_a, t_a, b_a, s_a = self.get_permissions(obs, expand_dims=True, to_bool=True)
        check_list = self.get_checklist(n_a, p_a, t_a, b_a, s_a)
        reduction_value = self.reduction_value.to(n_a.get_device())
        if prediction.dtype == torch.float16:
            reduction_value = reduction_value.half()

        for key, limit in self.action_limits.items():
            prediction[:, limit[0]:limit[1]] = th.where(check_list[key], prediction[:, limit[0]:limit[1]], reduction_value)

        return prediction

    def get_restricted_entropy(self, distribution,  obs):
        # entropy = 0
        n_a, p_a, t_a, b_a, s_a = self.get_permissions(obs, expand_dims=True, to_bool=True)
        check_list = self.get_checklist(n_a, p_a, t_a, b_a, s_a)
        action_decision = distribution.distribution
        min_real = torch.finfo(action_decision.logits.dtype).min
        logits = torch.clamp(action_decision.logits, min=min_real)
        p_log_p = logits * action_decision.probs
        zero_value = self.zero_value.to(n_a.get_device())

        for key, limit in self.action_limits.items():
            p_log_p[:, limit[0]:limit[1]] = th.where(check_list[key], p_log_p[:, limit[0]:limit[1]], zero_value)
        
        sum_p_log_p = p_log_p.sum(-1)
        entropy = - sum_p_log_p
        return entropy  
    
    def get_action_accuracy(self, actions, predictions):
        match = (actions == predictions).float()
        overall_acc = th.mean(match)
        
        acc_dict = {
            'overall_acc': overall_acc
            }
        for key, limit in self.action_limits.items():
            validity = (actions>=limit[0]) * (actions<limit[1])
            acc_dict["%s_acc" % key.lower()] = th.mean(match[validity])
        
        return acc_dict

    def dif_get_topk_actions(self, distribution, obs, k, with_extra=False):
        # raise ValueError("Not yet programmed")
        
        batch_size = distribution.shape[0]
        top_k_vals, top_k_inds = torch.topk(distribution, k= k + 1, dim=1)
        # stop_contained = torch.any(top_k_inds[:, :-1] == self.stop_action, 1)

        top_k_vals = top_k_vals.cpu().data.numpy()
        top_k_inds = top_k_inds.cpu().data.numpy()
        n_a, p_a, t_a, b_a, s_a = self.get_permissions(obs, to_bool=True)
        check_list = self.get_checklist(n_a, p_a, t_a, b_a, s_a)
        final_val = 0
        for key, limit in self.action_limits.items():
            final_val += check_list[key] * (limit[1] - limit[0])
        ks = torch.ones(n_a.shape).cuda() * k
        # ks = ks + stop_contained.float()
        actual_ks = torch.minimum(ks, final_val).long()
        min_valid = distribution > float('-inf')
        min_valid = min_valid.sum(1)
        actual_ks = torch.minimum(actual_ks, min_valid).long()
        top_k_val_list , top_k_ind_list = [],[]
        actual_ks = actual_ks.cpu().data.numpy()
        
        for i in range(batch_size):
            top_k_val_list.append(top_k_vals[i, :actual_ks[i]])
            top_k_ind_list.append(top_k_inds[i, :actual_ks[i]])
    
        # Now if there are stop actions in this set, then redo with updated ks.
        return top_k_val_list, top_k_ind_list
    
    def get_topk_actions(self, distribution, obs, k):
        raise ValueError("Not yet programmed")


class HCSGAction3D(MCSGAction3D):

    def __init__(self, resolution=64):

        self.resolution = resolution

        self.action_types = {
            "NUMERIC": resolution,
            "DRAW": 3,
            "TRANSFORM": 3,
            "BOOL": 3,
            "MACRO": 0,
            "FIXED_MACRO": 0,
            "STOP": 1 
        }
        self.command_index = {
            'sphere': self.resolution, 
            'cylinder': self.resolution + 1, 
            'cuboid': self.resolution + 2, 
            'translate': self.resolution + 3,
            'rotate': self.resolution + 4, 
            'scale': self.resolution + 5,
            'union': self.resolution + 6, 
            'intersection': self.resolution + 7, 
            'difference': self.resolution + 8,
            '$': self.resolution + 9,
            ## ADD MACROS HERE
        } 
        #Tuple for (Type of command, number of continuous parameters)
        self.index_to_command = {x:(0, 0) for x in range(self.resolution)} 
        self.index_to_command.update(
            {
                self.resolution: (1, 0),
                self.resolution + 1: (1, 0),
                self.resolution + 2: (1, 0),
                # self.resolution + 3: (1, 3),
                self.resolution + 3: (2, 3),
                self.resolution + 4: (2, 3),
                self.resolution + 5: (2, 3),
                self.resolution + 6: (3, 0),
                self.resolution + 7: (3, 0),
                self.resolution + 8: (3, 0),
                self.resolution + 9: (4, 0)
            }
        )
        self.stop_action = self.resolution + 9

        self.init_state()
        self.init_action_limits()


class FCSGAction3D(MCSGAction3D):

    def __init__(self, resolution=64):

        self.resolution = resolution

        self.action_types = {
            "NUMERIC": resolution,
            "DRAW": 3,
            "TRANSFORM": 0,
            "BOOL": 3,
            "MACRO": 0,
            "FIXED_MACRO": 0,
            "STOP": 1 
        }
        self.command_index = {
            'sphere': self.resolution, 
            'cylinder': self.resolution + 1, 
            'cuboid': self.resolution + 2, 
            'union': self.resolution + 3, 
            'intersection': self.resolution + 4, 
            'difference': self.resolution + 5,
            '$': self.resolution + 6,
            ## ADD MACROS HERE
        } 
        #Tuple for (Type of command, number of continuous parameters)
        self.index_to_command = {x:(0, 0) for x in range(self.resolution)} 
        self.index_to_command.update(
            {
                self.resolution: (1, 9),
                self.resolution + 1: (1, 9),
                self.resolution + 2: (1, 9),
                self.resolution + 3: (3, 0),
                self.resolution + 4: (3, 0),
                self.resolution + 5: (3, 0),
                self.resolution + 6: (4, 0)
            }
        )
        self.stop_action = self.resolution + 6

        self.init_state()
        self.init_action_limits()

    def single_expression_to_action(self, expr):
        command_symbol = expr.split("(")[0]
        if command_symbol in boolean_commands:
            action_list = [self.command_index[command_symbol]]
        elif command_symbol in draw_commands:
            param_str = expr.split("(")[1][:-1]
            param = np.array([float(x.strip()) for x in param_str.split(",")])
            translate_param = np.clip(param[:3], TRANSLATE_MIN + self.conversion_delta, TRANSLATE_MAX - self.conversion_delta)
            scale_param = np.clip(param[3:6], SCALE_MIN + self.conversion_delta, SCALE_MAX - self.conversion_delta)
            scale_param -= SCALE_ADDITION
            rotate_param = np.clip(param[6:9], ROTATE_MIN + self.conversion_delta, ROTATE_MAX - self.conversion_delta)
            rotate_param = rotate_param / ROTATE_MULTIPLIER
                # param = [self.mid_point + np.round( (x-1.25)/self.two_scale_delta) for x in param]
            param = np.concatenate([translate_param, scale_param, rotate_param], 0)
            param = (param - (-1 + self.conversion_delta)) / self.two_scale_delta
            param = np.round(param).astype(np.uint32)
            action_list = [self.command_index[command_symbol], ] + list(param)
        elif command_symbol == "$":
            action_list = [self.command_index[command_symbol]]
        return action_list

    def action_to_expression(self, actions):
        size_ = actions.shape[0]
        pointer = 0
        expression_list = []
        while(pointer < size_):
            cur_command = actions[pointer]
            cur_expr = self.index_to_expr[cur_command]
            n_param = self.index_to_command[cur_command][1]
            if n_param > 0:
                # Has to be for transform or mirror
                param = np.array(actions[pointer+1: pointer + 1 + n_param])
                translate_param = -1 + self.conversion_delta + param[:3] * self.two_scale_delta
                scale_param = -1 + self.conversion_delta + param[3:6] * self.two_scale_delta + SCALE_ADDITION
                rotate_param = (-1 + self.conversion_delta + param[6:9] * self.two_scale_delta) * ROTATE_MULTIPLIER
                
                param = np.concatenate([translate_param, scale_param, rotate_param], 0)
                param_str = ", ".join(["%f" % x for x in param])
                cur_expr = "%s(%s)" %(cur_expr, param_str)
                pointer += n_param
            expression_list.append(cur_expr)
            pointer += 1
                
        return expression_list


class PCSGAction3D(FCSGAction3D):

    def __init__(self, resolution=64):

        self.resolution = resolution

        self.action_types = {
            "NUMERIC": resolution,
            "DRAW": 2,
            "TRANSFORM": 0,
            "BOOL": 3,
            "MACRO": 0,
            "FIXED_MACRO": 0,
            "STOP": 1 
        }
        self.command_index = {
            'sphere': self.resolution, 
            'cuboid': self.resolution + 1, 
            'union': self.resolution + 2, 
            'intersection': self.resolution + 3, 
            'difference': self.resolution + 4,
            '$': self.resolution + 5,
            ## ADD MACROS HERE
        } 
        #Tuple for (Type of command, number of continuous parameters)
        self.index_to_command = {x:(0, 0) for x in range(self.resolution)} 
        self.index_to_command.update(
            {
                self.resolution: (1, 6),
                self.resolution + 1: (1, 6),
                self.resolution + 2: (3, 0),
                self.resolution + 3: (3, 0),
                self.resolution + 4: (3, 0),
                self.resolution + 5: (4, 0)
            }
        )
        self.stop_action = self.resolution + 5

        self.init_state()
        self.init_action_limits()


    def single_expression_to_action(self, expr):
        command_symbol = expr.split("(")[0]
        if command_symbol in boolean_commands:
            action_list = [self.command_index[command_symbol]]
        elif command_symbol in draw_commands:
            param_str = expr.split("(")[1][:-1]
            param = np.array([float(x.strip()) for x in param_str.split(",")])
            translate_param = np.clip(param[:3], TRANSLATE_MIN + self.conversion_delta, TRANSLATE_MAX - self.conversion_delta)
            scale_param = np.clip(param[3:6], SCALE_MIN + self.conversion_delta, SCALE_MAX - self.conversion_delta)
            scale_param -= SCALE_ADDITION
                # param = [self.mid_point + np.round( (x-1.25)/self.two_scale_delta) for x in param]
            param = np.concatenate([translate_param, scale_param], 0)
            param = (param - (-1 + self.conversion_delta)) / self.two_scale_delta
            param = np.round(param).astype(np.uint32)
            action_list = [self.command_index[command_symbol], ] + list(param)
        elif command_symbol == "$":
            action_list = [self.command_index[command_symbol]]
        return action_list

    def action_to_expression(self, actions):
        size_ = actions.shape[0]
        pointer = 0
        expression_list = []
        while(pointer < size_):
            cur_command = actions[pointer]
            cur_expr = self.index_to_expr[cur_command]
            n_param = self.index_to_command[cur_command][1]
            if n_param > 0:
                # Has to be for transform or mirror
                param = np.array(actions[pointer+1: pointer + 1 + n_param])
                translate_param = -1 + self.conversion_delta + param[:3] * self.two_scale_delta
                scale_param = -1 + self.conversion_delta + param[3:6] * self.two_scale_delta + SCALE_ADDITION
                param = np.concatenate([translate_param, scale_param], 0)
                param_str = ", ".join(["%f" % x for x in param])
                cur_expr = "%s(%s)" %(cur_expr, param_str)
                pointer += n_param
            expression_list.append(cur_expr)
            pointer += 1
                
        return expression_list
    

class MCSGActionExtended3D(MCSGAction3D):
    
    def __init__(self, resolution=64):
        super(MCSGActionExtended3D, self).__init__(resolution=resolution)
        
        
        self.action_types = {
            "NUMERIC": resolution,
            "DRAW": 5,
            "TRANSFORM": 3,
            "BOOL": 3,
            "MACRO": 1,
            "FIXED_MACRO": 3,
            "STOP": 1 
        }
        self.command_index = {
            'sphere': self.resolution, 
            'cylinder': self.resolution + 1, 
            'cuboid': self.resolution + 2, 
            'infinite_cylinder': self.resolution + 3, 
            'infinite_cone': self.resolution + 4,
            'translate': self.resolution + 5,
            'rotate': self.resolution + 6, 
            'scale': self.resolution + 7,
            'union': self.resolution + 8, 
            'intersection': self.resolution + 9, 
            'difference': self.resolution + 10,
            'mirror': self.resolution +11,
            'macro(MIRROR_X)': self.resolution + 12,
            'macro(MIRROR_Y)': self.resolution + 13,
            'macro(MIRROR_Z)': self.resolution + 14,
            '$': self.resolution + 15,
            ## ADD MACROS HERE
        } 
        #Tuple for (Type of command, number of continuous parameters)
        self.index_to_command = {x:(0, 0) for x in range(self.resolution)} 
        self.index_to_command.update(
            {
                self.resolution: (1, 0),
                self.resolution + 1: (1, 0),
                self.resolution + 2: (1, 0),
                self.resolution + 3: (2, 0),
                self.resolution + 4: (2, 0),
                # self.resolution + 3: (1, 3),
                self.resolution + 5: (2, 3),
                self.resolution + 6: (2, 3),
                self.resolution + 7: (2, 3),
                self.resolution + 8: (3, 0),
                self.resolution + 9: (3, 0),
                self.resolution + 10: (3, 0),
                ## FUNCTIONALLY EQUIVALENT TO "TRANFORMS"
                self.resolution + 11: (2, 3),
                self.resolution + 12: (2, 0),
                self.resolution + 13: (2, 0),
                self.resolution + 14: (2, 0),
                self.resolution + 15: (4, 0)
            }
        )
        self.stop_action = self.resolution + 15
        
        self.init_state()
        self.init_action_limits()