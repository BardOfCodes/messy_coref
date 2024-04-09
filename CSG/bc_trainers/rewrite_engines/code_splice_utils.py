import torch as th
import numpy as np
import copy
from CSG.env.reward_function import chamfer

def bool_count(substitute):
    cur_comamnds = substitute['commands']
    cur_subexpr_types = [x['type'] for x in cur_comamnds]
    cur_subexpr_n_bool = len([x for x in cur_subexpr_types if x == "B"])
    return cur_subexpr_n_bool

def get_new_command(command_list, command_inds, prev_canonical_commands, candidate, use_canonical=True):
    command_start, command_end = command_inds # node['subexpr_info']['command_ids']
    if use_canonical:
        target_commands = candidate['canonical_commands'] + candidate['commands']
        current_canonical_commands = [copy.deepcopy(x) for x in prev_canonical_commands] # [x.copy() for x in node['subexpr_info']['canonical_commands']]
        current_canonical_commands[1]['param'] *= -1
        current_canonical_commands[0]['param'] = 1/(current_canonical_commands[0]['param'] + 1e-9)
        new_command_list = command_list[:command_start] + current_canonical_commands[::-1] + target_commands + command_list[command_end:]
    else:
        target_commands = candidate['commands']
        new_command_list = command_list[:command_start] + target_commands + command_list[command_end:]
    return new_command_list

def distill_transform_chains(command_list, device, dtype, mode="3D"):
    new_command_list = []
    # scale_base = th.tensor([1, 1, 1], device=device, dtype=dtype)
    # translate_base = th.tensor([0, 0, 0], device=device, dtype=dtype)
    if mode == "3D":
        scale_base = np.array([1, 1, 1], dtype=np.float32)
        translate_base = np.array([0, 0, 0], dtype=np.float32)
    else:
        scale_base = np.array([1, 1], dtype=np.float32)
        translate_base = np.array([0, 0], dtype=np.float32)

    transform_chain = []
    for command in command_list:
        if 'macro_mode' in command.keys():
            resolved_transform_list = resolve_transform_chain(transform_chain, scale_base, translate_base)
            new_command_list.extend(resolved_transform_list)
            new_command_list.append(command)
        else:
            c_type = command['type']
            c_symbol = command['symbol']
            if c_type == "B":
                resolved_transform_list = resolve_transform_chain(transform_chain, scale_base, translate_base)
                new_command_list.extend(resolved_transform_list)
                new_command_list.append(command)
                transform_chain = []
            elif c_type == "T":
                if c_symbol == "scale":
                    transform_chain.append(command)
                elif c_symbol == "translate":
                    transform_chain.append(command)
                elif c_symbol == "rotate":
                    resolved_transform_list = resolve_transform_chain(transform_chain, scale_base, translate_base)
                    new_command_list.extend(resolved_transform_list)
                    new_command_list.append(command)
                    transform_chain = []
            elif c_type == "D":
                resolved_transform_list = resolve_transform_chain(transform_chain, scale_base, translate_base)
                new_command_list.extend(resolved_transform_list)
                new_command_list.append(command)
                transform_chain = []
            else:
                resolved_transform_list = resolve_transform_chain(transform_chain, scale_base, translate_base)
                new_command_list.extend(resolved_transform_list)
                new_command_list.append(command)
                transform_chain = []

    return new_command_list

def resolve_transform_chain(transform_chain, scale_base, translate_base):
    # scale_value = scale_base.clone()
    # translate_value = translate_base.clone()
    scale_value = scale_base.copy()
    translate_value = translate_base.copy()

    for command in transform_chain:
        c_symbol = command['symbol']
        if c_symbol == "translate":
            param = command['param']
            translate_value += scale_value * param
        elif c_symbol == "scale":
            param = command['param']
            scale_value *= param
    resolved_commands = []
    # Now we have the chain. 
    # check which should go first?
    
    if ((scale_value - scale_base)**2).mean() > 1e-4:    
        command = {"type": "T", "symbol": "scale", "param": scale_value}
        resolved_commands.append(command)
    if (translate_value**2).mean() > 1e-4:
        first_trans = translate_value /scale_value
        if np.sum(np.abs(first_trans) <= 1) < np.sum(np.abs(translate_value) <= 1):
            command = {"type": "T", "symbol": "translate", "param": first_trans}
            resolved_commands.insert(0, command)
        else:
            command = {"type": "T", "symbol": "translate", "param": translate_value}
            resolved_commands.append(command)
    
    return resolved_commands

def match_to_cache(cache, target):
    R = th.sum(th.logical_and(cache, target), (1, 2, 3))/ \
        (th.sum(th.logical_or(cache, target), (1, 2, 3)) + 1e-9)
    return R

def get_scores(prediction, target):
    R = th.logical_and(prediction, target).sum()/th.logical_or(prediction, target).sum()
    return R

def get_batched_scores(prediction, target):
    R = th.sum(th.logical_and(prediction, target), (1, 2, 3))/th.sum(th.logical_or(prediction, target), (1, 2, 3))
    return R

def get_masked_scores(cached_expression_shapes, target_shape, target_mask):
    
    R = th.sum(th.logical_and(th.logical_and(cached_expression_shapes, target_shape), target_mask), (1, 2, 3)) / \
            (th.sum(th.logical_and(th.logical_or(cached_expression_shapes, target_shape), target_mask), (1, 2, 3)) + 1e-6)
    return R

def batched_masked_scores(cache, targets, masks):
    R = th.sum(th.logical_and(th.logical_and(cache, targets), masks), -1) / \
        (th.sum(th.logical_and(th.logical_or(cache, targets), masks), -1) + 1e-6)
    return R

def get_2d_scores(prediction, target_np):
    real_pred_canvas = prediction.cpu().numpy()
    R = 100 - chamfer(target_np[None, :, :], real_pred_canvas[None, :, :])[0]
    return R