"""
NN to SA program
"""

from os import stat
import torch as th
import torch
import numpy as np

from gym.spaces.discrete import Discrete

from CSG.env.csg3d.constants import (SA_FLOAT_MIN, SA_FLOAT_MAX, CONVERSION_DELTA)
_AXIS = ["X", "Y", "Z"]
_FACES = ["left", "right", "bot", "top", "back", "front"]
_CUBOID_NAME_LIST = ["bbox", "cube0", "cube1", "cube2", "cube3", "cube4", 
                     "cube5", "cube6", "cube7", "cube8", "cube9", "cube10", "cube11"]
PRE_CUBOID_IND = 11

class HSA3DAction(Discrete):


    def __init__(self, resolution=64, n_cuboid_ind_states=5):

        ## N = res + here stands for
        self.resolution = resolution
        self.n_cuboid_ind_states = n_cuboid_ind_states

        self.action_types = dict(_FLOAT         = resolution,
                                _CUBOID         = 1, # (3f) or (3f, 1i) 
                                _ATTACH         = 1, # (c1, 6f)
                                _SQUEEZE        = 1, # (c_1, c_2, face, 2f)
                                _TRANSLATE      = 1, # (Axis, n_count, f)
                                _REFLECT        = 1, # (Axis)
                                _AXIS           = 3,
                                _FACE           = 6,
                                _SYM_COUNT      = 4,
                                _LEAF_TYPE_EMPTY= 1,
                                _LEAF_TYPE_SUBPR= 1,
                                )
        for i in range(n_cuboid_ind_states):
            self.action_types["_CUBOID_ID_%d" % i] = 1
        self.action_types["_SUBPROGRAM_STOP"] = 1
        self.action_types["_STOP"] = 1

        self.state_id_to_action_type = {
            0: "_FLOAT",
            1: "_CUBOID",
            2: "_ATTACH",
            3: "_SQUEEZE",
            4: "_TRANSLATE",
            5: "_REFLECT",
            6: "_AXIS",
            7: "_FACE",
            8: "_SYM_COUNT",
            9: "_LEAF_TYPE_EMPTY",
            10: "_LEAF_TYPE_SUBPR",
        }
        for i in range(n_cuboid_ind_states):
            self.state_id_to_action_type[11 + i] = "_CUBOID_ID_%d" % i
        stop_id = PRE_CUBOID_IND + self.n_cuboid_ind_states
        self.state_id_to_action_type[stop_id] = "_SUBPROGRAM_STOP"
        self.state_id_to_action_type[stop_id + 1] = "_STOP"

        self.command_index = {
            'cuboid'            : self.resolution, 
            'attach'            : self.resolution + 1, 
            'squeeze'           : self.resolution + 2, 
            'translate'         : self.resolution + 3,
            'reflect'           : self.resolution + 4, 
            'X'                 : self.resolution + 5, 
            'Y'                 : self.resolution + 6, 
            'Z'                 : self.resolution + 7,
            'left'              : self.resolution + 8,
            'right'             : self.resolution + 9,
            'bot'               : self.resolution + 10,
            'top'               : self.resolution + 11,
            'back'              : self.resolution + 12,
            'front'             : self.resolution + 13,
            'sym_1'             : self.resolution + 14,
            'sym_2'             : self.resolution + 15,
            'sym_3'             : self.resolution + 16,
            'sym_4'             : self.resolution + 17,
            'leaftype_0'        : self.resolution + 18,
            'leaftype_1'        : self.resolution + 19,
        } 
        for i in range(n_cuboid_ind_states):
            self.command_index[_CUBOID_NAME_LIST[i]] = self.resolution + 20 + i
        self.command_index['$'] = self.resolution + 20 + n_cuboid_ind_states
        self.command_index['$$'] = self.resolution + 20 + n_cuboid_ind_states + 1

        self.face_list = _FACES
        #Tuple for (Type of command, number of continuous parameters)
        self.index_to_command = {x:0 for x in range(self.resolution)} 
        self.index_to_command.update(
            {
                self.resolution         : 1,
                self.resolution + 1     : 2,
                self.resolution + 2     : 3,
                self.resolution + 3     : 4,
                self.resolution + 4     : 5,
                self.resolution + 5     : 6,
                self.resolution + 6     : 6,
                self.resolution + 7     : 6,
                self.resolution + 8     : 7,
                self.resolution + 9     : 7,
                self.resolution + 10    : 7,
                self.resolution + 11    : 7,
                self.resolution + 12    : 7,
                self.resolution + 13    : 7,
                self.resolution + 14    : 8,
                self.resolution + 15    : 8,
                self.resolution + 16    : 8,
                self.resolution + 17    : 8,
                self.resolution + 18    : 9,
                self.resolution + 19    : 10,
            }
        )
        for i in range(self.n_cuboid_ind_states):
            self.index_to_command[self.resolution + 20 + i] = PRE_CUBOID_IND + i
        self.index_to_command[self.resolution + 20 + self.n_cuboid_ind_states] = PRE_CUBOID_IND + self.n_cuboid_ind_states
        self.index_to_command[self.resolution + 20 + self.n_cuboid_ind_states + 1] = PRE_CUBOID_IND + self.n_cuboid_ind_states + 1

        self.stop_action = self.resolution + 20 + n_cuboid_ind_states + 1
        self.subprogram_stop_action = self.resolution + 20 + n_cuboid_ind_states
        self.stop_expression = "$$"
        self.subprogram_stop_expression = "$"
        self.leaf_is_empty_action = self.resolution + 18
        self.leaf_is_subpr_action = self.resolution + 19
        self.hierarchy_allowed = True
        self.stats_by_parts = True

        self.init_state()
        self.init_action_limits()

    

    def init_state(self):

        total_dim = np.sum(list(self.action_types.values()))
        super(HSA3DAction, self).__init__(total_dim)

        state = [0 for i in range(self.resolution + self.n_cuboid_ind_states)]
        state[0] = 1 
        self.state = state

        self.conversion_delta = CONVERSION_DELTA

        self.one_scale_delta = (1 - 2 * self.conversion_delta)/(self.resolution - 1)

        
        self.index_to_expr = {}
        for key, value in self.command_index.items():
            self.index_to_expr[value] = key
        for key in range(self.resolution):
            self.index_to_expr[key] = "{:4.4f}".format(self.conversion_delta + (key * self.one_scale_delta))
        # self.reduction_value = th.FloatTensor([- 1e9])
        self.reduction_value = th.FloatTensor([float('-inf')])
        self.zero_value = th.FloatTensor([0])

    def init_action_limits(self):

        # Now action type and range of actions:
        count = 0
        self.action_limits = {}
        self.action_values = {}
        for key, value in self.action_types.items():
            self.action_limits[key] = (count, count + value)
            self.action_values[key] = list(range(count, count + value))
            count += value  

    
    def set_state(self, state):
        self.state = state.copy()

    def get_state(self):
        return self.state.copy()

    @staticmethod
    def is_cuboid(expression):
        valid = 'cuboid' in expression
        return valid

    @staticmethod
    def is_attach(expression):
        valid = 'attach' in expression
        return valid
    
    @staticmethod
    def is_squeeze(expression):
        valid = 'attach' in expression
        return valid

    @staticmethod
    def is_translate(expression):
        valid = 'attach' in expression
        return valid
        
    @staticmethod
    def is_reflect(expression):
        valid = 'reflect' in expression
        return valid

    @staticmethod
    def is_stop(expression):
        return expression == "$$"
    

    def is_stop_action(self, action):
        return action == self.stop_action
     
    @staticmethod
    def get_permissions(obs, expand_dims=False, to_bool=False):
        state = obs['state']
        if len(state.shape) < 1:
            state = state.unsqueeze(0)
        if expand_dims:
            state = state.unsqueeze(0)
        if to_bool:
            state =state.bool()
        return state

    def validate_action(self, action):
        cur_index = self.index_to_command[action]
        state = self.get_state()
        valid = self.state[cur_index]
        if not valid:
            print("WUT")
        
    def expression_to_action(self, expression_list):
        action_list = []
        program_id = 0
        for expr in expression_list:
            cur_action_list, program_finished = self.single_expression_to_action(expr, program_id)
            action_list.extend(cur_action_list)
            if program_finished:
                program_id += 1
        action_list = np.array(action_list, dtype=np.int32)
        return action_list

    def single_expression_to_action(self, expr, program_id):
        # There are 6 kinds of commands: cuboid, attach, squeeze, translate, reflect, stop
        # If program id == 0, slightly different parse.
        program_finished = False
        if "cuboid(" in expr:
            # SKIP bbox for subprograms
            if "bbox" in expr:
                if program_id == 0:
                    # only extract float of height
                    param_str = expr.split("(")[1][:-1].split(",")
                    param = param_str[1].strip()
                    param = np.array(float(param))
                    param = np.clip(param, SA_FLOAT_MIN + self.conversion_delta, SA_FLOAT_MAX - self.conversion_delta)
                    param = (param - (self.conversion_delta)) / self.one_scale_delta
                    param = np.round(param).astype(np.uint32)
                    action_list = [param]
                else:
                    action_list = []
            else:
                command_id = self.command_index["cuboid"]
                param_str = expr.split("(")[1][:-1].split(",")
                param_str = [x.strip() for x in param_str]
                param = np.array([float(x) for x in param_str[:-1]])
                param = np.clip(param, SA_FLOAT_MIN + self.conversion_delta, SA_FLOAT_MAX - self.conversion_delta)
                param = (param - (self.conversion_delta)) / self.one_scale_delta
                param = np.round(param).astype(np.uint32)
                action_list = [command_id, ] + list(param)
                if program_id == 0 and self.hierarchy_allowed:
                    value = int(int(param_str[-1]) > 0)
                    cuboid_type = self.command_index['leaftype_%s' % value]
                    action_list.append(cuboid_type)

        elif "attach(" in expr:
            command_id = self.command_index["attach"]
            param_str = expr.split("(")[1][:-1].split(",")
            param_str = [x.strip() for x in param_str]
            cuboid_id = self.command_index[param_str[1]]
            param = np.array([float(x) for x in param_str[2:]])
            param = np.clip(param, SA_FLOAT_MIN + self.conversion_delta, SA_FLOAT_MAX - self.conversion_delta)
            param = (param - (self.conversion_delta)) / self.one_scale_delta
            param = np.round(param).astype(np.uint32)
            action_list = [command_id, cuboid_id] + list(param)
        elif "squeeze(" in expr:
            command_id = self.command_index["squeeze"]
            param_str = expr.split("(")[1][:-1].split(",")
            param_str = [x.strip() for x in param_str]
            cuboid_id_1 = self.command_index[param_str[1]]
            cuboid_id_2 = self.command_index[param_str[2]]
            face_id = self.command_index[param_str[3]]
            param = np.array([float(x.strip()) for x in param_str[4:]])
            param = np.clip(param, SA_FLOAT_MIN + self.conversion_delta, SA_FLOAT_MAX - self.conversion_delta)
            param = (param - (self.conversion_delta)) / self.one_scale_delta
            param = np.round(param).astype(np.uint32)
            action_list = [command_id, cuboid_id_1, cuboid_id_2, face_id] + list(param)
        elif "translate(" in expr:
            command_id = self.command_index["translate"]
            param_str = expr.split("(")[1][:-1].split(",")
            param_str = [x.strip() for x in param_str]
            axis_id = self.command_index[param_str[1]]
            sym_count = self.command_index["sym_%s" % param_str[2]]
            param = np.array([float(x) for x in param_str[3:]])
            param = np.clip(param, SA_FLOAT_MIN + self.conversion_delta, SA_FLOAT_MAX - self.conversion_delta)
            param = (param - (self.conversion_delta)) / self.one_scale_delta
            param = np.round(param).astype(np.uint32)
            action_list = [command_id, axis_id, sym_count] + list(param)
        elif "reflect(" in expr:
            command_id = self.command_index["reflect"]
            param_str = expr.split("(")[1][:-1].split(",")
            param_str = [x.strip() for x in param_str]
            axis_id = self.command_index[param_str[1]]
            action_list = [command_id, axis_id,]
        elif "$$" == expr:
            action_list = [self.stop_action]
            program_finished = True
        elif "$" == expr:
            action_list = [self.subprogram_stop_action]
            program_finished = True
        return action_list, program_finished


    def action_to_expression(self, actions):
        # finish the first one:
        bbox_height = self.index_to_expr[actions[0]]# self.conversion_delta + (actions[0] * self.one_scale_delta)
        expression_list = ["bbox = cuboid(1.0, %s, 1.0, 0)" % bbox_height]
        pointer = 1
        size_ = actions.shape[0]
        program_id = 0
        cube_count = 0
        cur_cube_name = "cube0"
        subpr_index = 0
        while(pointer < size_):
            cur_command = actions[pointer]
            command_symbol = self.index_to_expr[cur_command]
            if command_symbol == "cuboid":
                cur_cube_name = "cube%d" % cube_count
                param = [self.index_to_expr[x] for x in actions[pointer +1: pointer + 4]]
                param_str = ", ".join(param)
                # param = np.array(actions[pointer + 1: pointer + 4])
                # param = self.conversion_delta + (param * self.one_scale_delta)
                if program_id == 0 and self.hierarchy_allowed:
                    valid = self.index_to_expr[actions[pointer + 4]]
                    valid = int(valid[-1])
                    subpr_index += valid
                    if valid:
                        input = subpr_index
                    else:
                        input = 0
                    shift = 5
                else:
                    input = 0
                    shift = 4
                cur_expr = ["%s = cuboid(%s, %d)" % (cur_cube_name, param_str, input)]
                cube_count += 1
            elif command_symbol == "attach":
                # retrieve one cuboid id and 6 floats:
                target_cube_name = self.index_to_expr[actions[pointer + 1]]

                param = [self.index_to_expr[x] for x in actions[pointer + 2: pointer + 8]]
                param_str = ", ".join(param)
                # param = np.array(actions[pointer + 2: pointer + 8])
                # param = self.conversion_delta + (param * self.one_scale_delta)
                # param_str = ", ".join([str(x) for x in param])
                cur_expr = ["attach(%s, %s, %s)" % (cur_cube_name, target_cube_name, param_str)]
                shift = 8
            elif command_symbol == "squeeze":
                target_cube_name_1 = self.index_to_expr[actions[pointer + 1]]
                target_cube_name_2 = self.index_to_expr[actions[pointer + 2]]
                face_name = self.index_to_expr[actions[pointer + 3]]

                param = [self.index_to_expr[x] for x in actions[pointer + 4: pointer + 6]]
                param_str = ", ".join(param)
                # param = np.array(actions[pointer + 4: pointer + 6])
                # param = self.conversion_delta + (param * self.one_scale_delta)
                # param_str = ", ".join([str(x) for x in param])
                cur_expr = ["squeeze(%s, %s, %s, %s, %s)" % (cur_cube_name, target_cube_name_1, 
                                                            target_cube_name_2, face_name, param_str)]
                shift = 6
            elif command_symbol == "translate":
                axis = self.index_to_expr[actions[pointer+1]]
                sym_count = self.index_to_expr[actions[pointer+2]]
                sym_count = int(sym_count[-1])
                param_str = self.index_to_expr[actions[pointer + 3]]
                # param = np.array(actions[pointer + 3])
                # param = self.conversion_delta + (param * self.one_scale_delta)
                cur_expr = ["translate(%s, %s, %d, %s)" % (cur_cube_name, axis, sym_count, param_str)]
                shift = 4
            elif command_symbol == "reflect":
                axis = self.index_to_expr[actions[pointer+1]]
                cur_expr = ["reflect(%s, %s)" % (cur_cube_name, axis)]
                shift = 2
            elif command_symbol == "$$":
                # stop:
                program_id += 1
                cube_count = 0
                cur_cube_name = "cube0"
                cur_expr = [command_symbol]
                shift = 1
            elif command_symbol == "$":
                # stop:
                program_id += 1
                cube_count = 0
                cur_cube_name = "cube0"
                cur_expr = [command_symbol]
                # Subprogram Left:
                subp_bbox_expr = ["bbox = cuboid(1.0, 1.0, 1.0, 0)"]
                cur_expr.extend(subp_bbox_expr)
                shift = 1
            expression_list.extend(cur_expr)
            pointer += shift
                
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
        sample_list = []
        for ind, valid in enumerate(self.state):
            if valid:
                key_type = self.state_id_to_action_type[ind]
                lims = self.action_limits[key_type]
                new_vals = list(range(lims[0], lims[1]))
                sample_list.extend(new_vals)
        if not sample_list:
            raise Exception("No action allowed!")
        output = np.random.choice(sample_list)
        return output
    
    
    def restrict_pred_action(self, prediction, obs):
        state = self.get_permissions(obs, expand_dims=False, to_bool=True)
        reduction_value = self.reduction_value.to(state.get_device())
        
        if prediction.dtype == torch.float16:
            reduction_value = reduction_value.half()
        mod_state = []
        for ind, key_name in self.state_id_to_action_type.items():
            lims = self.action_limits[key_name]
            extended_state = state[:, ind:ind+1].expand(-1, lims[1]-lims[0])
            mod_state.append(extended_state)
            
            # prediction[:, lims[0]:lims[1]] = th.where(state[:, ind: ind+1], prediction[:, lims[0]:lims[1]], reduction_value)
        mod_state = torch.cat(mod_state, 1)
        prediction = th.where(mod_state, prediction, reduction_value)
        # when there are too many this might be simpler:
        # all_states = []
        # for ind, key_name in self.state_id_to_action_type.items():
        #     lims = self.action_limits[key_name]
        #     all_states.append(state[:, ind: ind+1].expand(-1, lims[1] - lims[0]))
        # final_state = torch.cat(all_states, 1)
        # prediction = th.where(final_state, prediction, reduction_value)

        return prediction

    def get_restricted_entropy(self, distribution,  obs):
        # entropy = 0
        state = self.get_permissions(obs, expand_dims=False, to_bool=True)
        action_decision = distribution.distribution
        min_real = torch.finfo(action_decision.logits.dtype).min
        logits = torch.clamp(action_decision.logits, min=min_real)
        p_log_p = logits * action_decision.probs
        zero_value = self.zero_value.to(state.get_device())

        for ind, key_name in self.state_id_to_action_type.items():
            lims = self.action_limits[key_name]
            p_log_p[:, lims[0]:lims[1]] = th.where(state[:, ind:ind+1], p_log_p[:, lims[0]:lims[1]], zero_value)

        # all_states = []
        # for ind, key_name in self.state_id_to_action_type.items():
        #     lims = self.action_limits[key_name]
        #     all_states.append(state[:, ind: ind+1].expand(-1, lims[1] - lims[0]))
        # final_state = torch.cat(all_states, 1)
        # p_log_p = th.where(final_state, p_log_p, zero_value)

        
        sum_p_log_p = p_log_p.sum(-1)
        entropy = - sum_p_log_p
        return entropy  
    
    def get_action_accuracy(self, actions, predictions):
        match = (actions == predictions).float()
        overall_acc = th.mean(match)
        
        acc_dict = {
            'overall_acc': overall_acc
            }
        if self.stats_by_parts:
            for key, limit in self.action_limits.items():
                validity = (actions>=limit[0]) * (actions<limit[1])
                acc_dict["%s_acc" % key.lower()] = th.mean(match[validity])
        
        return acc_dict

    def dif_get_topk_actions(self, distribution, obs, k, with_extra=False):
        # raise ValueError("Not yet programmed")
        
        batch_size = distribution.shape[0]
        top_k_vals, top_k_inds = torch.topk(distribution, k= k, dim=1)
        # stop_contained = torch.any(top_k_inds[:, :-1] == self.stop_action, 1)

        top_k_vals = top_k_vals.cpu().data.numpy()
        top_k_inds = top_k_inds.cpu().data.numpy()
        state = self.get_permissions(obs, to_bool=True)
        final_val = 0
        for ind, key_name in self.state_id_to_action_type.items():
            lims = self.action_limits[key_name]
            final_val += state[:, ind] * (lims[1] - lims[0])
        ks = torch.ones(state.shape[0]).cuda() * k
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


class PSA3DAction(HSA3DAction):

    def __init__(self, *args, **kwargs):

        super(PSA3DAction, self).__init__(*args, **kwargs)

        self.hierarchy_allowed = False
        self.stats_by_parts = False