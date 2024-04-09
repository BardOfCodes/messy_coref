import time
import torch as th
import numpy as np
import _pickle as cPickle
import os
from .code_splice_utils import get_scores
from CSG.env.reward_function import chamfer
from CSG.env.csg3d.languages import GraphicalMCSG3DCompiler
from CSG.env.csg2d.languages import GraphicalMCSG2DCompiler


REWARD_THRESHOLD = 0.00


def top_down_prune(graph, parser, compiler, target, threshold=0.01, mode="OLD"):
    non_terminals = []
    n_nodes = len(graph.nodes)
    if n_nodes == 2:
        return graph
    
        
    boolean_nodes = [graph.nodes[i] for i in range(n_nodes) if graph.nodes[i]['type'] == "B"]
    
    if not boolean_nodes:
        return graph
    
    reward_nodes = [n['reward'] for n in boolean_nodes]
    reward_threshold = np.max(reward_nodes) - threshold
    selected_nodes = [n for n in boolean_nodes if n['reward'] >= reward_threshold]
    expr_len_nodes = [n['expression_length'] for n in selected_nodes]
    min_selected_len = np.min(expr_len_nodes)
    selected_nodes = [n for n in selected_nodes if n['expression_length'] == min_selected_len]
    selected_node = selected_nodes[0]
    # Now we need to attach the transforms above this node:
    cur_node = selected_node
    cur_node_id = cur_node['node_id']
    cur_parent_id = cur_node['parent']
    cur_parent = graph.nodes[cur_parent_id]
    while(not cur_parent['type'] == "ROOT"):
        # print("found", cur_parent['symbol'])
        if cur_parent['type'] == "B":
            # Then override:
            grand_parent_id = cur_parent['parent']
            grand_parent = graph.nodes[grand_parent_id]
            # print("child, parent, grand", cur_node_id, cur_parent_id, grand_parent_id)
            graph.remove_edge(cur_parent_id, cur_node_id)
            graph.remove_edge(grand_parent_id, cur_parent_id)
            graph.add_edge(grand_parent_id, cur_node_id)
            cur_node['parent'] = grand_parent_id
            cur_parent['parent'] = None
            cur_parent['children'] = None
            grand_parent['children'] = [cur_node_id]
            cur_parent = grand_parent
            cur_parent_id = cur_parent['node_id']
        else:
            cur_node = cur_parent
            cur_node_id = cur_node['node_id']
            grand_parent_id = cur_parent['parent']
            grand_parent = graph.nodes[grand_parent_id]
            cur_parent = grand_parent
            cur_parent_id = grand_parent_id
    
    # Remove the other parts: 
    # check if rewards match out:
    start_node_id = graph.nodes[0]['children'][0]
    start_node = graph.nodes[start_node_id]

    commands = compiler.tree_to_command(graph, start_node)
    expression = parser.get_expression(commands)
    commands = parser.parse(expression)
    if mode == "OLD":
        graph = compiler.command_tree(commands, target, add_sweeping_info=True)
    else:
        raise ValueError("Need to be reconfigured.")
        # graph = compiler.command_tree(commands, target.bool(), add_sweeping_info=True, add_splicing_info=True, enable_subexpr_targets=True)
        # graph = compiler.get_masking_based_validity(graph)

    return graph

def bottom_up_prune(graph, parser, compiler, target):
    # Top down:
    n_nodes = len(graph.nodes)
    if n_nodes == 2:
        return graph
    
    
    start_node_id = graph.nodes[0]['children'][0]
    start_node = graph.nodes[start_node_id]
    state_list = [start_node]
    no_parent_clause = False
    n_cuts = 0
    while(state_list):
        cur_node = state_list[0]
        state_list = state_list[1:]
        cur_node_id = cur_node['node_id']
        cur_parent_id = cur_node['parent']
        cur_children_id = cur_node['children']
        cur_node_type = cur_node['type']
        # print(cur_node['symbol'])
        if cur_node_type == "B":
            validity = cur_node['validity']
            if not validity[0]:
                # Output is 0
                # print("found empty canvas")
                # This should be already taken care of.
                # Go to previous bool
                # Connect other child to parent of bool
                new_parent_id = cur_parent_id
                child_id = cur_node_id
                while(True):
                    new_parent = graph.nodes[new_parent_id]
                    parent_type = new_parent['type']
                    # print(new_parent['symbol'])
                    if parent_type == "B":
                        bool_node = new_parent
                        break
                    elif parent_type == "ROOT":
                        # print("0 output connected to root")
                        no_parent_clause = True
                        break
                    else:
                        # print("setting new id")
                        child_id = new_parent_id
                        new_parent_id = new_parent['parent']
                if not no_parent_clause:
                    # We found the boolean node for edit
                    bool_id = bool_node['node_id']
                    bool_parent_id = bool_node['parent']
                    bool_parent = graph.nodes[bool_parent_id]
                    bool_children_id = bool_node['children']
                    bool_children_id = [x for x in bool_children_id if not x == child_id]
                    if len(bool_children_id)> 1:
                        raise ValueError("something wrong with val 0. bool children len > 1")
                    bool_child_id = bool_children_id[0]
                    bool_child = graph.nodes[bool_child_id]
                    
                    # severe this branch
                    graph.remove_edge(bool_parent_id, bool_id)
                    bool_parent['children'].remove(bool_id)
                    bool_node['parent'] = None
                    
                    graph.remove_edge(bool_id, bool_child_id)
                    bool_node['children'].remove(bool_child_id)
                    bool_child['parent'] = None
                                    
                    graph.add_edge(bool_parent_id, bool_child_id)
                    bool_child['parent'] = bool_parent_id
                    bool_parent['children'].append(bool_child_id)
                    
                    # Attach new:
                    # print("Finished CUT 0")
                    n_cuts += 1
                    
            elif not validity[1]:
                # Matches child 1
                bool_parent_id = cur_node['parent']
                bool_parent = graph.nodes[bool_parent_id]
                bool_child_id = cur_node['children'][0]
                bool_child = graph.nodes[bool_child_id]
                
                #severe
                cur_node['parent'] = None
                bool_parent['children'].remove(cur_node_id)
                graph.remove_edge(bool_parent_id, cur_node_id)
                
                bool_child['parent'] = None
                cur_node['children'].remove(bool_child_id)
                graph.remove_edge(cur_node_id, bool_child_id)
                
                bool_child['parent'] = bool_parent_id
                bool_parent['children'].append(bool_child_id)
                graph.add_edge(bool_parent_id, bool_child_id)
                # print("Finished CUT 1")
                n_cuts += 1
                
            elif not validity[2]:
                # Matches child 2
                # print('here')
                bool_parent_id = cur_node['parent']
                bool_parent = graph.nodes[bool_parent_id]
                bool_child_id = cur_node['children'][1]
                bool_child = graph.nodes[bool_child_id]
                
                #severe
                cur_node['parent'] = None
                bool_parent['children'].remove(cur_node_id)
                graph.remove_edge(bool_parent_id, cur_node_id)
                
                bool_child['parent'] = None
                cur_node['children'].remove(bool_child_id)
                graph.remove_edge(cur_node_id, bool_child_id)
                
                bool_child['parent'] = bool_parent_id
                bool_parent['children'].append(bool_child_id)
                graph.add_edge(bool_parent_id, bool_child_id)
                # print("Finished CUT 2")
                n_cuts += 1
            else:
                # Everything matches just add the children:
                for child_id in cur_children_id[::-1]:
                    child = graph.nodes[child_id]
                    state_list.insert(0, child)
        else:
            # Just add the children:
            for child_id in cur_children_id[::-1]:
                child = graph.nodes[child_id]
                state_list.insert(0, child)
        if n_cuts == 10:
            break
        if no_parent_clause:
            break
        
    start_node_id = graph.nodes[0]['children'][0]
    start_node = graph.nodes[start_node_id]

    commands = compiler.tree_to_command(graph, start_node)
    expression = parser.get_expression(commands)
    commands = parser.parse(expression)
    graph = compiler.command_tree(commands, target, add_sweeping_info=False)
    return graph
  
def gs_singular_parse(cur_expression, target, parser, compiler, td_mode="OLD", reward_threshold=0.01):

    command_list = parser.parse(cur_expression)
    command_tree = compiler.command_tree(command_list, target=target, add_sweeping_info=True)
    ## How to Update:
    top_down_pruned = top_down_prune(command_tree, parser, compiler, target, mode=td_mode, threshold=reward_threshold)
    
    start_node_id = top_down_pruned.nodes[0]['children'][0]
    start_node = top_down_pruned.nodes[start_node_id]
    updated_command_list = compiler.tree_to_command(top_down_pruned, start_node) 
    # Min best tree
    best_expression = parser.get_expression(updated_command_list)
    td_delta = len(cur_expression) - len(best_expression)
    
    bottom_up_pruned = bottom_up_prune(top_down_pruned, parser, compiler, target)

    start_node_id = bottom_up_pruned.nodes[0]['children'][0]
    start_node = bottom_up_pruned.nodes[start_node_id]
    updated_command_list = compiler.tree_to_command(bottom_up_pruned, start_node) 
    # Min best tree
    best_expression = parser.get_expression(updated_command_list)
    bu_delta = len(cur_expression) - len(best_expression) - td_delta
    
    # compiler.reset()
    # cmd_list = parser.parse(best_expression)
    # compiler._compile(cmd_list)
    # output = compiler._output <=0
    # iou = th.logical_and(output, target).sum() / th.logical_or(output, target).sum()
    # old_iou = command_tree.nodes[1]['reward']
    # if iou < old_iou - 0.1:
    #     print("WHAT")
    #     ... 
    
    return best_expression, td_delta, bu_delta

def parallel_gs(proc_id, mini_prog_dict, temp_env, save_loc, td_mode, reward_threshold, length_alpha=0, count_limit=9998):

    th.backends.cudnn.benchmark = True
    # compiler = temp_env.program_generator.compiler
    if "3D" in temp_env.language_name:
        compiler = GraphicalMCSG3DCompiler(resolution=temp_env.program_generator.compiler.resolution,
                                    scale=temp_env.program_generator.compiler.scale,
                                    draw_mode=temp_env.program_generator.compiler.draw.mode)
    else:
        compiler = GraphicalMCSG2DCompiler(resolution=temp_env.program_generator.compiler.resolution,
                                    scale=temp_env.program_generator.compiler.scale,
                                    draw_mode=temp_env.program_generator.compiler.draw.mode)

    temp_env.program_generator.set_execution_mode(th.device("cuda"), th.float16)
    parser = temp_env.program_generator.parser

    compiler.set_to_cuda()
    compiler.set_to_half()
    compiler.reset()


    previous_rewards = []
    updated_rewards = []
    previous_lengths = []
    updated_lengths = []
    bu_delta = []
    td_delta = []
    
    start_time = time.time()
    prog_objs= []

    with th.no_grad():
        # with th.cuda.amp.autocast():
            for iteration, (key, value) in enumerate(mini_prog_dict):
                if iteration % 100 == 0:
                    print("Process: ", proc_id, "Iteration: ", iteration)
                    if updated_rewards:
                        print("New Avg", np.nanmean(updated_rewards),
                                "Prev Avg", np.nanmean(previous_rewards))
                cur_slot = key[0]
                cur_target = key[1]
                obs = temp_env.reset_to_target(cur_slot, cur_target)
                target_np = obs['obs']
                target = th.from_numpy(target_np).cuda().bool()
                # key = (cur_slot, cur_target)
                cur_expression = value['expression']
                prev_reward = value['reward']
                
                best_expression, td_delta_sin, bu_delta_sin = gs_singular_parse(cur_expression, target, parser, compiler, td_mode=td_mode, reward_threshold=reward_threshold)
                td_delta.append(td_delta_sin)
                bu_delta.append(bu_delta_sin)

                actions = temp_env.action_space.expression_to_action(best_expression)
                try: 
                    real_expression = temp_env.action_space.action_to_expression(actions) 
                except:
                    print(actions)
                    print(best_expression)
                    print("previous", cur_expression)
                    print("WUT")
                real_pred_canvas = temp_env.program_generator.execute(real_expression, return_numpy=True)

                if "3D" in temp_env.language_name:
                    real_pred_canvas = temp_env.program_generator.execute(best_expression, return_numpy=False, return_bool=True)
                    real_reward = get_scores(real_pred_canvas, target).item()
                else:
                    real_pred_canvas = temp_env.program_generator.execute(best_expression, return_numpy=True, return_bool=True)
                    real_reward = 100 - chamfer(target_np[None, :, :], real_pred_canvas[None, :, :])[0]
                real_reward = real_reward + length_alpha * len(best_expression)

                if real_reward >= (prev_reward - REWARD_THRESHOLD) and len(best_expression) < len(cur_expression):
                    updated_lengths.append(len(real_expression))
                    updated_rewards.append(real_reward)
                    prog_obj = dict(expression=real_expression,
                                    slot_id=cur_slot,
                                    target_id=cur_target,
                                    reward=real_reward,
                                    origin="GS",
                                    do_fail=False,
                                    cs_fail=False,
                                    log_prob=0)
                    prog_objs.append(prog_obj)
                else:
                    # Add lengths as well
                    updated_rewards.append(prev_reward)
                    updated_lengths.append(len(cur_expression))
                previous_rewards.append(prev_reward)
                previous_lengths.append(len(cur_expression))

    if len(prog_objs) > count_limit:
        prog_objs.sort(key=lambda x: x['reward'], reverse=True)
        prog_objs = prog_objs[:count_limit]
            
    if not previous_rewards:
        previous_rewards = [0]
        updated_rewards = [0]
        previous_lengths = [1]
        updated_lengths = [1]
    
    with open(save_loc + "/gs_%d.pkl" % proc_id, 'wb') as f:
        cPickle.dump([prog_objs, updated_rewards, previous_rewards, updated_lengths, previous_lengths, 
                      td_delta, bu_delta], f)
  