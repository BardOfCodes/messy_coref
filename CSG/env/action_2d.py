
import numpy as np
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete
import torch
import torch as th
# from .csg2d.parsers import DRAW_SHAPES, OPERATION_TYPES, expr2multi, labels2exps, multi2exp, multi_expr2multi, multi_multi2exp

ACTION_SPACE = [3, 3, 3, 7, 7, 7]
# MULTI_ACTION_SPACE = [3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7]
MULTI_ACTION_SPACE = [3, 3, 3, 49, 49, 25, 49, 49, 25, 49, 49, 25]
# ACTION_SPACE = [3, 48 +1, 48 +1, 24 + 1, 3, 3]
"""
MultiRestrictedAction Setup: [3, 7, 7, 7, 3, 3]
1: Command type: 3 
    0: Draw
    1: Operation
    2: Stop
2: Draw Specs:
    2.1: Draw location X: 32
    2.2: Draw location Y: 32
    2.3: Draw Scale: 32
    2.4: Draw Type: 3
    scale values: 8, 12, 16, 20, 24, 28, 32
    x,y values: 8, 16, 24, 32, 40, 48, 56
3: Operation Spec: 3
    0: Union
    1: Intersect
    2: Subtract
"""
        

class RestrictedAction(Discrete):

    def __init__(self, n):
        super(RestrictedAction, self).__init__(n)
        self.draw_allowed = 1
        self.op_allowed = 0
        self.stop_allowed = 0
        self.stop_expression = "$"
        # self.reduction_value = float(-1e6)
        # self.reduction_value = th.FloatTensor([- 1e9])
        self.reduction_value = th.FloatTensor([float('-inf')])
        self.zero_value = th.FloatTensor([0])
        self.target_space = Discrete(1)
        
        self.draw_actions = list(range(396))
        self.op_actions = list(range(396, 399))
        self.stop_actions = list(range(399, 400))

    def set_state(self, draw_allowed, op_allowed, stop_allowed):
        self.draw_allowed = draw_allowed
        self.op_allowed = op_allowed
        self.stop_allowed = stop_allowed

    def get_state(self):
        return self.draw_allowed, self.op_allowed, self.stop_allowed

    @staticmethod
    def label_to_expression(labels, unique_draw):
        size_ = labels.shape[0]
        expressions = labels2exps(labels, size_, unique_draw)
        return expressions
    
    @staticmethod
    def is_draw(expression):
        bool_list = [x in expression for x in DRAW_SHAPES]
        return any(bool_list)

    @staticmethod
    def is_operation(expression):
        bool_list = [x in expression for x in OPERATION_TYPES]
        return any(bool_list)

    @staticmethod
    def is_stop(expression):
        return expression == "$"

    def action_to_expression(self, action, unique_draw):
        action = np.array([action])
        expressions = labels2exps(action, 1, unique_draw)
        return expressions[0]

    def expression_to_action(self, action, unique_draw):
        output = unique_draw.index(action)
        return output
    def get_conditional_entropy(self, distribution):
        return distribution.entropy()
    
    def restrict_pred_action(self, prediction, obs):
        d_a, o_a, s_a = self.get_permissions(obs, expand_dims=True, to_bool=True)
        # prediction = prediction/2.0
        # zz = torch.nn.functional.softmax(prediction)
        # print((s_a==True).float().sum(), z[:, -1].max()) 
        # z = (s_a==1)
        # zzz = z * zz[:, -1]
        # print((zzz.sum()/(z.sum() + 1e-5)).item())
        # print(zz.max().item(), zzz.max().item())
        # print(prediction.max().item(), prediction.min().item())
        reduction_value = self.reduction_value.to(d_a.get_device())
        prediction[:,:396] = th.where(d_a, prediction[:, :396], reduction_value) 
        prediction[:,396:399] = th.where(o_a, prediction[:, 396:399], reduction_value)
        prediction[:,399:400] = th.where(s_a, prediction[:, 399:400], reduction_value)
        
        return prediction

    def get_log_prob(self, distribution, actions, obs):
        log_p = distribution.log_prob(actions)
        return log_p
                
    def get_entropy(self, distribution,  obs):
        # entropy = 0
        d_a, o_a, s_a = self.get_permissions(obs, expand_dims=True, to_bool=True)
        action_decision = distribution.distribution
        min_real = torch.finfo(action_decision.logits.dtype).min
        logits = torch.clamp(action_decision.logits, min=min_real)
        p_log_p = logits * action_decision.probs
        zero_value = self.zero_value.to(d_a.get_device())
        p_log_p[:,:396] = th.where(d_a, p_log_p[:, :396], zero_value) 
        p_log_p[:,396:399] = th.where(o_a, p_log_p[:, 396:399], zero_value)
        p_log_p[:,399:400] = th.where(s_a, p_log_p[:, 399:400], zero_value)
        sum_p_log_p = p_log_p.sum(-1)
        entropy = - sum_p_log_p
        return entropy  
    
    def get_topk_actions(self, distribution, obs, k):
        
        d_a, o_a, s_a = self.get_permissions(obs, to_bool=True)
        d_a = d_a.item()
        o_a = o_a.item()
        s_a = s_a.item()
        actual_k = min(k, 396 * int(d_a) + 3 * int(o_a) + int(s_a))
        if actual_k == 0:  
            raise Exception("No actions possible!")
        top_k_vals, top_k_inds = torch.topk(distribution, k=actual_k, dim=1)
        return top_k_vals, top_k_inds
    
    def get_max_action(self, distribution):
        _, max_actions = th.max(distribution.distribution.logits, 1)
        return max_actions
    
    
    def dif_get_topk_actions(self, distribution, obs, k):
        
        batch_size = distribution.shape[0]
        top_k_vals, top_k_inds = torch.topk(distribution, k=k, dim=1)
        d_a, o_a, s_a = self.get_permissions(obs, to_bool=True)
        ks = torch.ones(d_a.shape).cuda() * k
        actual_ks = torch.minimum(ks, 396 * d_a + 3 * o_a + s_a).long()
        min_valid = distribution > float('-inf')
        min_valid = min_valid.sum(1)
        actual_ks = torch.minimum(actual_ks, min_valid).long()
        top_k_val_list , top_k_ind_list = [],[]
        
        for i in range(batch_size):
            top_k_val_list.append(top_k_vals[i, :actual_ks[i].item()])
            top_k_ind_list.append(top_k_inds[i, :actual_ks[i].item()])
        return top_k_val_list, top_k_ind_list
                        
    @staticmethod
    def get_permissions(obs, expand_dims=False, to_bool=False):
        dummy = obs['draw_allowed']
        items = [obs['draw_allowed'], obs['op_allowed'], obs['stop_allowed']]
        if not len(dummy.shape) == 1:
            items =  [x[:,0] for x in items]
        if expand_dims:
            items = [x[:, None] for x in items]
        if to_bool:
            items = [x.bool() for x in items]
        d_a, o_a, s_a = items
        return d_a, o_a, s_a
        # for the first one:
    def sample(self):
        sample_list = []
        if self.draw_allowed:
            sample_list.extend(self.draw_actions)
        if self.op_allowed:
            sample_list.extend(self.op_actions)
        if self.stop_allowed:
            sample_list.extend(self.stop_actions)
        if not sample_list:
            raise Exception("No action allowed!")
        output = np.random.choice(sample_list)
        return output
    
    def get_all_log_prob(self, distribution, actions):
        log_prob = distribution.log_prob(actions)
        return log_prob
    
    def get_action_accuracy(self, actions, predictions):
        match = (actions == predictions).float()
        acc = th.mean(match)
        
        draw_commands = actions<=395
        draw_acc = th.mean(match[draw_commands])
        op_commands = th.logical_not(draw_commands)
        op_acc = th.mean(match[op_commands])
        stop_commands = actions>=399
        stop_acc = th.mean(match[stop_commands])
        
        acc_dict = {
            'overall_acc': acc, 
            'overall_draw_acc': draw_acc, 
            'op_acc': op_acc, 
            'stop_acc': stop_acc
        }
        
        return acc_dict
"""
USED ONLY FOR DATA GENERATION.
"""
class OpRestrictedAction(RestrictedAction):
    """
    Have specific switch for each operation.
    """   
    
    def __init__(self, n):
        
        super(OpRestrictedAction, self).__init__(n)
        self.add_allowed = 0
        self.sub_allowed = 0
        self.inter_allowed = 0
        self.add_actions = [396]
        self.sub_actions = [397]
        self.inter_actions = [398]
    
    def set_state(self, draw_allowed, add_allowed, inter_allowed, sub_allowed, 
                   stop_allowed):
        self.draw_allowed = draw_allowed
        self.add_allowed = add_allowed
        self.inter_allowed = inter_allowed
        self.sub_allowed = sub_allowed
        self.stop_allowed = stop_allowed

    def get_state(self):
        return self.draw_allowed, self.add_allowed, self.inter_allowed, \
            self.sub_allowed, self.stop_allowed

    def restrict_pred_action(self, prediction, obs):
        raise Exception("This Action state is not designed for such usage.")

    def get_entropy(self, distribution,  obs):
        raise Exception("This Action state is not designed for such usage.")

    def get_topk_actions(self, distribution, obs, k):
        raise Exception("This Action state is not designed for such usage.")
                        
    @staticmethod
    def get_permissions(obs, expand_dims=False, to_bool=False):
        dummy = obs['draw_allowed']
        items = [obs['draw_allowed'], obs['add_allowed'], obs['inter_allowed'], obs['sub_allowed'], obs['stop_allowed']]
        if not len(dummy.shape) == 1:
            items = [x[:,0] for x in items]
        
        if expand_dims:
            items = [x[:, None] for x in items]
        if to_bool:
            items= [x.bool() for x in items]
        d_a, add_a, inter_a, sub_a, s_a = items
        return d_a, add_a, inter_a, sub_a, s_a
    
    
        # for the first one:
    def sample(self, action_set = []):
        sample_list = []
        prob = []
        # Change the sampling of actions
        if self.draw_allowed:
            sample_list.extend(self.draw_actions)
            prob.extend([1,] * 396)
            # remove the ones in action_set:
            for id in action_set:
                if id < 396:
                    prob[id] = 0
        if self.add_allowed:
            sample_list.extend(self.add_actions)
            prob.extend([396,])
        if self.sub_allowed:
            sample_list.extend(self.sub_actions)
            prob.extend([396,])
        if self.inter_allowed:
            sample_list.extend(self.inter_actions)
            prob.extend([396,])
        if self.stop_allowed:
            sample_list.extend(self.stop_actions)
            prob.extend([1,])
        if not sample_list:
            sample_list = [None]
            prob = [1.0]
            # raise Exception("No action allowed!")
        prob = [float(x)/sum(prob) for x in prob]
        output = np.random.choice(sample_list, p=prob)
        return output


class RefactoredActionSpace(MultiDiscrete):

    def __init__(self, nvec, dtype=np.int64, seed=None):
        super(RefactoredActionSpace, self).__init__(nvec, seed)
        #  NVec expected to be: [3, 3, 3, 7, 7, 7]
        
        self.draw_allowed = 1
        self.op_allowed = 0
        self.stop_allowed = 0
        self.stop_expression = "$"
        self.target_space = Discrete(6)
        # self.reduction_value = th.FloatTensor([- 1e9])
        self.reduction_value = th.FloatTensor([float('-inf')])
        self.zero_value = th.FloatTensor([0])
        self.one_value = th.FloatTensor([1])
        self.is_draw = RestrictedAction.is_draw
        self.is_operation = RestrictedAction.is_operation
        self.is_stop = RestrictedAction.is_stop
        self.label_to_expression = RestrictedAction.label_to_expression
        self.get_permissions = RestrictedAction.get_permissions
        self.dtype = dtype
        self.init_draw_limits()
        
    def init_draw_limits(self):
        
        self.draw_specs = self.nvec[2:]
        self.draw_mins = np.array([8, 8, 8])
        self.draw_maxs = np.array([56, 56, 32])
        self.draw_steps = (self.draw_maxs - self.draw_mins)/(self.draw_specs[1:] -1)

    def set_state(self, draw_allowed, op_allowed, stop_allowed):
        self.draw_allowed = draw_allowed
        self.op_allowed = op_allowed
        self.stop_allowed = stop_allowed

    def get_state(self):
        return self.draw_allowed, self.op_allowed, self.stop_allowed

    def sample(self):
        selector_array = []
        if self.draw_allowed:
            draw_specs = (self.np_random.random_sample(
                self.draw_specs.shape) * self.draw_specs).astype(self.dtype)
            selector_array.append(0)
        else:
            draw_specs = np.array([0, ] * 4, dtype=self.dtype)
        
        if self.op_allowed:
            op_type = self.np_random.randint(3)
            selector_array.append(1)
        else:
            op_type = 0  # self.np_random.randint(3)
        
        if self.stop_allowed:
            selector_array.append(2)
            
        ac_type = np.random.choice(selector_array)
        
        output = np.concatenate([[ac_type],[op_type], draw_specs ])
        return output
    
    def expression_to_action(self, expression, unique_draw):
        # conver the exression into action target
        output = expr2multi(expression, self.draw_specs, self.draw_mins, self.draw_steps, self.dtype)
        return output
    
    def action_to_expression(self, action, unique_draw):
        # action = action[None, :]
        expression = multi2exp(action, self.draw_specs, self.draw_mins, self.draw_steps, self.dtype)
        return expression

    def get_entropy(self, distribution, obs):
        
        d_a, o_a, s_a = self.get_permissions(obs, to_bool=True)
        d_a_, o_a_, s_a_ = self.get_permissions(obs, expand_dims=True, to_bool=False)
        
        action_decision = distribution.distribution[0] 
        p_log_p = action_decision.logits * action_decision.probs
        sum_p_log_p = p_log_p[:, :1] * d_a_ + p_log_p[:, 1:2] * o_a_ + p_log_p[:, 2:3] * s_a_
        sum_p_log_p = sum_p_log_p.sum(-1)
        zero_value = self.zero_value.to(p_log_p.get_device())
        
        action_selector_bool = (d_a.long() + o_a.long() + s_a.long() > 1)
        primary_op = th.where(action_selector_bool,-sum_p_log_p, zero_value)
        draw_p_1 = th.where(d_a,distribution.distribution[2].entropy(), zero_value)
        draw_p_2 = th.where(d_a,distribution.distribution[3].entropy(), zero_value)
        draw_p_3 = th.where(d_a,distribution.distribution[4].entropy(), zero_value)
        draw_p_4 = th.where(d_a,distribution.distribution[5].entropy(), zero_value)
        op_p = th.where(o_a,distribution.distribution[1].entropy(), zero_value)
        
        
        entropy = primary_op + action_decision.probs[:, 1] * op_p + action_decision.probs[:, 2] * (draw_p_1 + draw_p_2 + draw_p_3 + draw_p_4)
        
        return entropy  
        

    def get_log_prob(self, distribution, actions, obs):
        

        d_a, o_a, s_a = self.get_permissions(obs, to_bool=True)
        
        action_selector_bool = (d_a.long() + o_a.long() + s_a.long() > 1)
        zero_value = self.zero_value.to(d_a.get_device())
        
        primary_op = th.where(action_selector_bool,distribution.distribution[0].log_prob(actions[:, 0]), zero_value)
        draw_p_1 = th.where(d_a,distribution.distribution[2].log_prob(actions[:, 2]), zero_value)
        draw_p_2 = th.where(d_a,distribution.distribution[3].log_prob(actions[:, 3]), zero_value)
        draw_p_3 = th.where(d_a,distribution.distribution[4].log_prob(actions[:, 4]), zero_value)
        draw_p_4 = th.where(d_a,distribution.distribution[5].log_prob(actions[:, 5]), zero_value)
        op_p = th.where(o_a,distribution.distribution[1].log_prob(actions[:, 1]), zero_value)
        
        log_p = primary_op + draw_p_1 + draw_p_2 + draw_p_3 + draw_p_4 + op_p
        return log_p
    
    
    def get_conditional_entropy(self, distribution):
        ### Conditional distributions are present.
        
        entropy_stack = th.stack([dist.entropy() for dist in distribution.distribution], dim=1)
        action_probs = distribution.distribution[0].probs
        
        draw_entropy = entropy_stack[:, 2:].sum(1) * action_probs[:,0]
        op_entropy = entropy_stack[:, 1:2].sum(1) * action_probs[:,1]
        action_entropy = entropy_stack[:,0]
        entropy = action_entropy + draw_entropy + op_entropy
        return entropy  
        
    def get_all_log_prob(self, distribution, actions):
        
        all_log_probs = th.stack([dist.log_prob(action) for dist, action in zip(distribution.distribution, th.unbind(actions, dim=1))], dim=1)
        action_log_probs = all_log_probs[:,0]
        zero_value = self.zero_value.to(action_log_probs.get_device())
        draw_log_probs = torch.where(actions[:, 0] == 0, all_log_probs[:, 2:].sum(-1) , zero_value)
        
        op_log_probs = torch.where(actions[:, 0] == 1, all_log_probs[:, 1:2].sum(-1) , zero_value)
        
        log_probs = action_log_probs + draw_log_probs + op_log_probs
        
        return log_probs
    
    def get_action_accuracy(self, actions, predictions):
        match = (actions == predictions).float()
        acc = th.mean(match)
        
        match = (actions[:, 0] == predictions[:, 0]).float()
        action_acc = th.mean(match)
        
        draw_commands = (actions[:,0:1] == 0).float()
        draw_per_ac_acc = draw_commands * ((actions[:, 2:] == predictions[:, 2:]).float())
        draw_acc = (draw_per_ac_acc.sum(1) == 4).float()
        
        
        draw_type_acc = (draw_per_ac_acc[:,0] == 1).float()
        x_acc = (draw_per_ac_acc[:,1] == 1).float()
        y_acc = (draw_per_ac_acc[:,2] == 1).float()
        size_acc = (draw_per_ac_acc[:,3] == 1).float()
        
        draw_acc = draw_acc.sum(-1) / draw_commands.sum()
        x_acc = x_acc.sum(-1) / draw_commands.sum()
        y_acc = y_acc.sum(-1) / draw_commands.sum()
        size_acc = size_acc.sum(-1) / draw_commands.sum()
        draw_type_acc = draw_type_acc.sum(-1) / draw_commands.sum()
        
        op_commands = (actions[:,0] == 1).float()
        op_acc = op_commands* (actions[:, 1] == predictions[:, 1]).float()
        op_acc = op_acc.sum() / op_commands.sum()
        
        stop_commands = (actions[:,0] == 2).float()
        stop_acc = stop_commands* match
        stop_acc = stop_acc.sum() / stop_commands.sum()
        
        acc_dict = {
            'overall_acc': acc, 
            'action_acc': action_acc,
            'overall_draw_acc': draw_acc, 
            'draw_type_acc': draw_type_acc,
            'x_acc': x_acc,
            'y_acc': y_acc,
            'size_acc': size_acc,
            'stop_acc': stop_acc,
            'op_acc': op_acc, 
        }
        return acc_dict

    def get_max_action(self, distribution):
        max_actions = [th.max(x.logits, 1)[1] for x in distribution.distribution]
        max_actions = torch.stack(max_actions, 1)
        return max_actions
    
    def restrict_pred_action(self, prediction, obs):
        # this is B, 30
        d_a, o_a, s_a = self.get_permissions(obs, expand_dims=True, to_bool=True)
        reduction_value = self.reduction_value.to(d_a.get_device())
        restricted_action_1 = th.where(d_a, prediction[:,:1], reduction_value)
        restricted_action_2 = th.where(o_a, prediction[:,1:2], reduction_value)
        restricted_action_3 = th.where(s_a, prediction[:,2:3], reduction_value)
        
        output = th.cat([restricted_action_1, restricted_action_2, restricted_action_3,
                         prediction[:,3:]], 1)
        return output


class MultiRefactoredActionSpace(RefactoredActionSpace):
    def __init__(self, nvec, dtype=np.int64, seed=None):
        super(MultiRefactoredActionSpace, self).__init__(nvec, dtype, seed)
        #  NVec expected to be: [3, 3, 3, 7, 7, 7]
        
    def init_draw_limits(self):
        
        self.draw_specs = self.nvec[2:]
        self.draw_mins = np.array([8, 8, 8, 8, 8, 8, 8, 8, 8])
        self.draw_maxs = np.array([56, 56, 32, 56, 56, 32, 56, 56, 32])
        self.draw_steps = (self.draw_maxs - self.draw_mins)/(self.draw_specs[1:] -1)

    def sample(self):
        selector_array = []
        if self.draw_allowed:
            draw_specs = (self.np_random.random_sample(
                self.draw_specs.shape) * self.draw_specs).astype(self.dtype)
            selector_array.append(0)
        else:
            draw_specs = np.array([0, ] * 10, dtype=self.dtype)
        
        if self.op_allowed:
            op_type = self.np_random.randint(3)
            selector_array.append(1)
        else:
            op_type = 0  # self.np_random.randint(3)
        
        if self.stop_allowed:
            selector_array.append(2)
            
        ac_type = np.random.choice(selector_array)
        
        output = np.concatenate([[ac_type],[op_type], draw_specs ])
        return output
    
    def expression_to_action(self, expression, unique_draw):
        # conver the exression into action target
        output = multi_expr2multi(expression, self.draw_specs, self.draw_mins, self.draw_steps, self.dtype)
        return output
    
    def adjust_expression(self, expression):
        for ind, expr in enumerate(expression):
            if expr[0] in DRAW_SHAPES:
                draw_type = DRAW_SHAPES.index(expr[0])
                params = expr[2:-1].split(",")
                params = [float(x) for x in params]
                sel_draw_mins = self.draw_mins[draw_type * 3:(draw_type+1) * 3]
                sel_draw_maxs = self.draw_maxs[draw_type * 3:(draw_type+1) * 3]
                params = [np.clip(np.round(x), sel_draw_mins[i],sel_draw_maxs[i]) for i, x in enumerate(params)]
                adjusted_expr  = expr[:2] + "%d,%d,%d" %(params[0], params[1],params[2])+expr[-1:]
                expression[ind] = adjusted_expr
        return expression
    
    def action_to_expression(self, action, unique_draw):
        # action = action[None, :]
        expression = multi_multi2exp(action, self.draw_specs, self.draw_mins, self.draw_steps, self.dtype)
        return expression

    
    def get_action_accuracy(self, actions, predictions):
        
        
        match = (actions[:, 0] == predictions[:, 0]).float()
        action_acc = th.mean(match)
        
        zero_value = self.zero_value.to(actions.get_device()).long()
        draw_commands = (actions[:,0:1] == 0).float()
        
        draw_type = actions[:,2:3]
        draw_s_params = torch.where(draw_type == 0, predictions[:, 3:6], zero_value)
        draw_t_params = torch.where(draw_type == 1, predictions[:, 6:9], zero_value)
        draw_c_params = torch.where(draw_type == 2, predictions[:, 9:12], zero_value)
        final_draw_pred = draw_s_params + draw_t_params + draw_c_params
        
        target_s_params = torch.where(draw_type == 0, actions[:, 3:6], zero_value)
        target_t_params = torch.where(draw_type == 1, actions[:, 6:9], zero_value)
        target_c_params = torch.where(draw_type == 2, actions[:, 9:12], zero_value)
        
        final_draw_target = target_s_params + target_t_params + target_c_params
        
        draw_per_ac_acc = draw_commands * ((final_draw_pred[:, :] == final_draw_target[:, :]).float())
        
        final_draw_target = draw_commands * ((actions[:, 2:] == predictions[:, 2:]).float())
        
        x_acc = (draw_per_ac_acc[:,0] == 1).float()
        y_acc = (draw_per_ac_acc[:,1] == 1).float()
        size_acc = (draw_per_ac_acc[:,2] == 1).float()
        draw_type_acc = draw_commands * (actions[:, 2:3] == predictions[:, 2:3]).float()
        draw_acc_array = draw_type_acc[:, 0] * (draw_per_ac_acc.sum(1) == 3).float()
        
        x_acc = x_acc.sum() / draw_commands.sum()
        y_acc = y_acc.sum() / draw_commands.sum()
        size_acc = size_acc.sum() / draw_commands.sum()
        draw_type_acc = draw_type_acc.sum() / draw_commands.sum()
        draw_acc = draw_acc_array.sum() / draw_commands.sum()
        
        op_commands = (actions[:,0] == 1).float()
        op_acc_array = op_commands* (actions[:, 1] == predictions[:, 1]).float()
        op_acc = op_acc_array.sum() / op_commands.sum()
        
        stop_commands = (actions[:,0] == 2).float()
        stop_acc_array = stop_commands* match
        stop_acc = stop_acc_array.sum() / stop_commands.sum()
        
        match = draw_acc_array + op_acc_array + stop_acc_array
        acc = th.mean(match)
        
        acc_dict = {
            'overall_acc': acc, 
            'action_acc': action_acc,
            'overall_draw_acc': draw_acc, 
            'draw_type_acc': draw_type_acc,
            'x_acc': x_acc,
            'y_acc': y_acc,
            'size_acc': size_acc,
            'stop_acc': stop_acc,
            'op_acc': op_acc, 
        }
        return acc_dict

    def get_entropy(self, distribution, obs):
        
        d_a, o_a, s_a = self.get_permissions(obs, to_bool=True)
        d_a_, o_a_, s_a_ = self.get_permissions(obs, expand_dims=True, to_bool=False)
        
        action_decision = distribution.distribution[0] 
        p_log_p = action_decision.logits * action_decision.probs
        sum_p_log_p = p_log_p[:, :1] * d_a_ + p_log_p[:, 1:2] * o_a_ + p_log_p[:, 2:3] * s_a_
        sum_p_log_p = sum_p_log_p.sum(-1)
        zero_value = self.zero_value.to(p_log_p.get_device())
        
        action_selector_bool = (d_a.long() + o_a.long() + s_a.long() > 1)
        primary_op = th.where(action_selector_bool,-sum_p_log_p, zero_value)
        
        
        op_p = th.where(o_a,distribution.distribution[1].entropy(), zero_value)
        
        draw_action = distribution.distribution[2]
        draw_p_1 = th.where(d_a,distribution.distribution[2].entropy(), zero_value)
        
        draw_p_2 = th.where(d_a,distribution.distribution[3].entropy(), zero_value)
        draw_p_3 = th.where(d_a,distribution.distribution[4].entropy(), zero_value)
        draw_p_4 = th.where(d_a,distribution.distribution[5].entropy(), zero_value)
        
        draw_p_5 = th.where(d_a,distribution.distribution[6].entropy(), zero_value)
        draw_p_6 = th.where(d_a,distribution.distribution[7].entropy(), zero_value)
        draw_p_7 = th.where(d_a,distribution.distribution[8].entropy(), zero_value)
        
        draw_p_8 = th.where(d_a,distribution.distribution[9].entropy(), zero_value)
        draw_p_9 = th.where(d_a,distribution.distribution[10].entropy(), zero_value)
        draw_p_10 = th.where(d_a,distribution.distribution[11].entropy(), zero_value)

        entropy = primary_op + action_decision.probs[:, 1] * op_p + action_decision.probs[:, 0] * (draw_p_1 \
            + draw_action.probs[:, 0] * (draw_p_2 + draw_p_3 + draw_p_4) +\
            + draw_action.probs[:, 1] * (draw_p_5 + draw_p_6 + draw_p_7) +\
            + draw_action.probs[:, 2] * (draw_p_8 + draw_p_9 + draw_p_10))
        
        return entropy  
        

    def get_log_prob(self, distribution, actions, obs):
        
        zero_value = self.zero_value.to(actions.get_device())
        
        all_log_probs = th.stack([dist.log_prob(action) for dist, action in zip(distribution.distribution, th.unbind(actions, dim=1))], dim=1)
        action_log_probs = all_log_probs[:,0]
        
        op_log_probs = torch.where(actions[:, 0] == 1, all_log_probs[:, 1:2].sum(-1) , zero_value)
        
        draw_type_log_probs = torch.where(actions[:, 0] == 0, all_log_probs[:, 2:3].sum(-1) , zero_value)
        draw_s_log_probs =  torch.where((actions[:, 0] == 0) & (actions[:, 2] == 0), all_log_probs[:, 3:6].sum(-1) , zero_value)
        draw_t_log_probs =  torch.where((actions[:, 0] == 0) & (actions[:, 2] == 1), all_log_probs[:, 6:9].sum(-1) , zero_value)
        draw_c_log_probs =  torch.where((actions[:, 0] == 0) & (actions[:, 2] == 2), all_log_probs[:, 9:12].sum(-1) , zero_value)
        
        log_probs = action_log_probs + op_log_probs + draw_type_log_probs + draw_s_log_probs + draw_t_log_probs + draw_c_log_probs
        
        return log_probs
    
    
    def dif_get_topk_actions(self, distribution, obs, k):
        
        batch_size = distribution[0].shape[0]
        top_k_val_list = []
        top_k_ind_list = []
        for distr in distribution:
            real_k = min(k, distr.shape[1])
            top_k_vals, top_k_inds = torch.topk(distribution, k=k, dim=1)
            top_k_val_list.append(top_k_vals)
            top_k_ind_list.append(top_k_inds)
        
        
        d_a, o_a, s_a = self.get_permissions(obs, to_bool=True)
        ks = torch.ones(d_a.shape).cuda() * k
        actual_ks = torch.minimum(ks, 396 * d_a + 3 * o_a + s_a).long()
        top_k_val_list , top_k_ind_list = [],[]
        
        for i in range(batch_size):
            top_k_val_list.append(top_k_vals[i, :actual_ks[i].item()])
            top_k_ind_list.append(top_k_inds[i, :actual_ks[i].item()])
        return top_k_val_list, top_k_ind_list
    
    
    def get_conditional_entropy(self, distribution):
        ### Conditional distributions are present.
        
        entropy_stack = th.stack([dist.entropy() for dist in distribution.distribution], dim=1)
        action_probs = distribution.distribution[0].probs
        draw_probs = distribution.distribution[2].probs
        draw_entropy = entropy_stack[:, 2:3].sum(1) * action_probs[:,0]
        op_entropy = entropy_stack[:, 1:2].sum(1) * action_probs[:,1]
        action_entropy = entropy_stack[:,0]
        draw_s_entropy = entropy_stack[:, 3:6].sum(1) * draw_probs[:,0] * action_probs[:,0]
        draw_t_entropy = entropy_stack[:, 6:9].sum(1) * draw_probs[:,1] * action_probs[:,0]
        draw_c_entropy = entropy_stack[:, 9:12].sum(1) * draw_probs[:,2] * action_probs[:,0]
        entropy = action_entropy + op_entropy + draw_entropy + draw_s_entropy + draw_t_entropy + draw_c_entropy
        return entropy  
        
    def get_all_log_prob(self, distribution, actions):
        
        zero_value = self.zero_value.to(actions.get_device())
        
        all_log_probs = th.stack([dist.log_prob(action) for dist, action in zip(distribution.distribution, th.unbind(actions, dim=1))], dim=1)
        action_log_probs = all_log_probs[:,0]
        
        op_log_probs = torch.where(actions[:, 0] == 1, all_log_probs[:, 1:2].sum(-1) , zero_value)
        
        draw_type_log_probs = torch.where(actions[:, 0] == 0, all_log_probs[:, 2:3].sum(-1) , zero_value)
        draw_s_log_probs =  torch.where((actions[:, 0] == 0) & (actions[:, 2] == 0), all_log_probs[:, 3:6].sum(-1) , zero_value)
        draw_t_log_probs =  torch.where((actions[:, 0] == 0) & (actions[:, 2] == 1), all_log_probs[:, 6:9].sum(-1) , zero_value)
        draw_c_log_probs =  torch.where((actions[:, 0] == 0) & (actions[:, 2] == 2), all_log_probs[:, 9:12].sum(-1) , zero_value)
        
        log_probs = action_log_probs + op_log_probs + draw_type_log_probs + draw_s_log_probs + draw_t_log_probs + draw_c_log_probs
        
        return log_probs
    


class OpRefactoredAction(RefactoredActionSpace):
    """
    Have specific switch for each operation.
    """   
    
    def __init__(self, n):
        
        super(OpRestrictedAction, self).__init__(n)
        self.add_allowed = 0
        self.sub_allowed = 0
        self.inter_allowed = 0
        self.add_actions = [396]
        self.sub_actions = [397]
        self.inter_actions = [398]
    
    def set_state(self, draw_allowed, add_allowed, inter_allowed, sub_allowed, 
                   stop_allowed):
        self.draw_allowed = draw_allowed
        self.add_allowed = add_allowed
        self.inter_allowed = inter_allowed
        self.sub_allowed = sub_allowed
        self.stop_allowed = stop_allowed

    def get_state(self):
        return self.draw_allowed, self.add_allowed, self.inter_allowed, \
            self.sub_allowed, self.stop_allowed

    def restrict_pred_action(self, prediction, obs):
        raise Exception("This Action state is not designed for such usage.")

    def get_entropy(self, distribution,  obs):
        raise Exception("This Action state is not designed for such usage.")

    def get_topk_actions(self, distribution, obs, k):
        raise Exception("This Action state is not designed for such usage.")
                        
    @staticmethod
    def get_permissions(obs, expand_dims=False, to_bool=False):
        dummy = obs['draw_allowed']
        items = [obs['draw_allowed'], obs['add_allowed'], obs['inter_allowed'], obs['sub_allowed'], obs['stop_allowed']]
        if not len(dummy.shape) == 1:
            items = [x[:,0] for x in items]
        
        if expand_dims:
            items = [x[:, None] for x in items]
        if to_bool:
            items= [x.bool() for x in items]
        d_a, add_a, inter_a, sub_a, s_a = items
        return d_a, add_a, inter_a, sub_a, s_a
    
    
        # for the first one:
    def sample(self, action_set = []):
        sample_list = []
        prob = []
        # Change the sampling of actions
        if self.draw_allowed:
            sample_list.extend(self.draw_actions)
            prob.extend([1,] * 396)
            # remove the ones in action_set:
            for id in action_set:
                if id < 396:
                    prob[id] = 0
        if self.add_allowed:
            sample_list.extend(self.add_actions)
            prob.extend([396,])
        if self.sub_allowed:
            sample_list.extend(self.sub_actions)
            prob.extend([396,])
        if self.inter_allowed:
            sample_list.extend(self.inter_actions)
            prob.extend([396,])
        if self.stop_allowed:
            sample_list.extend(self.stop_actions)
            prob.extend([1,])
        if not sample_list:
            sample_list = [None]
            prob = [1.0]
            # raise Exception("No action allowed!")
        prob = [float(x)/sum(prob) for x in prob]
        output = np.random.choice(sample_list, p=prob)
        return output
