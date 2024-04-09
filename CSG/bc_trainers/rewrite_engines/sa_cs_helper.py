import copy
from re import S
import re
import torch as th
from CSG.env.csg3d.languages import MCSG3DParser, PCSG3DParser, GraphicalMCSG3DCompiler, MCSG3DCompiler
from CSG.env.shape_assembly.graph_compiler import GraphicalSACompiler
from CSG.env.shape_assembly.parser import SAParser

from CSG.env.csg3d.parser_utils import pcsg_to_ntcsg, ntcsg_to_pcsg
from CSG.env.csg3d.languages import boolean_commands
import time
import random
import numpy as np
import _pickle as cPickle
import os
import re

from .code_splice_utils import get_masked_scores, get_new_command, get_scores, bool_count, get_batched_scores, distill_transform_chains
from .graph_sweep_helper import gs_singular_parse
from .code_splice_helper import get_base_log_prob, get_batched_candidate_log_probs, get_candidate_log_probs
from .sa_utils import get_master_program_struct, sa_convert_hcsg_commands_to_cuda, get_cube_id_from_hcsg, min_copy_master_program_struct

DUMMY_TYPE = "DUMMY"
PROB_BATCH_LIMIT = 300

def parallel_sa_code_splice(proc_id, mini_prog_dict, temp_env, subexpr_cache, save_dir, top_k=15,
                         max_boolean_count=11, rewrite_limit=20, node_masking_req=1.0,
                         add_dummy_node=False, apply_gs=True, return_top_k=1, quantize_expr=True,
                         use_canonical=True, policy=None, use_probs=False, logprob_threshold=0.1, reward_based_thresh=False,
                         valid_nodes=["B", "D", "DUMMY"], higher_language=False, max_valid_length=96, max_complexity=16,
                         length_alpha=0):
    # Deal with this late

    device, dtype = th.device("cuda"), th.float16
    th.backends.cudnn.benchmark = True
    temp_env.program_generator.set_execution_mode(th.device("cuda"), dtype)

    base_parser = temp_env.program_generator.parser
    base_compiler = temp_env.program_generator.compiler
    action_resolution = temp_env.action_space.resolution
    action_space = temp_env.action_space
    
    graph_compiler = GraphicalSACompiler(resolution=temp_env.program_generator.compiler.resolution,
                                scale=temp_env.program_generator.compiler.scale,
                                draw_mode=temp_env.program_generator.compiler.draw.mode)

    graph_compiler.set_to_cuda()
    graph_compiler.set_to_half()
    graph_compiler.reset()
    base_compiler.set_to_cuda()
    base_compiler.set_to_half()
    base_compiler.reset()



    updated_rewards = []
    previous_rewards = []

    previous_length = []
    updated_length = []

    previous_logprob = []
    updated_logprob = []
    prog_objs = []
    num_rewrites = []

    st = time.time()
    tot_programs = len(mini_prog_dict)
    
    failed_keys = []
    with th.no_grad():
        for iteration, (key, value) in enumerate(mini_prog_dict):
            if iteration % 10 == 0:
                print("cur iteration %d of %d. Cur Time %f" %
                    (iteration, tot_programs, time.time() - st))
                if updated_rewards:
                    print("New Avg", np.nanmean(updated_rewards),
                            "Prev Avg", np.nanmean(previous_rewards))
            improved_progs = []
            cur_slot, cur_target, origin = key
            base_expr = value['expression']
            original_expr = base_expr.copy()
            selected_base_expr = base_expr.copy()
            # mcsg_expr = base_parser.convert_to_mcsg(basecsg_expr)
            original_reward = value['reward']

            obs = temp_env.reset_to_target(cur_slot, cur_target)
            target_np = obs['obs']
            target = th.from_numpy(target_np).cuda().bool()
            unsqueezed_target = target.unsqueeze(0)
            counter = 0
            # Get initial log prob

            while(counter <= rewrite_limit):
                inserted_dummy_node = False
                counter += 1
                command_list = base_parser.parse(selected_base_expr)
                base_compiler._compile(command_list)
                output = base_compiler._output.clone()
                output = (output <= 0)

                if use_probs:
                    # Get the latent execution:
                    initial_logprob = get_base_log_prob(policy, selected_base_expr, action_space, output, target, temp_env.perm_max_len)
                    # print("Current Log Prob", initial_logprob)
                else:
                    initial_logprob = 0

                R = get_scores(output, target)
                R = R + length_alpha * len(selected_base_expr)
                initial_reward = R.item()
                max_reward = initial_reward  # .item()
                if counter == 1:
                    original_logprob = initial_logprob

                master_program_struct = get_master_program_struct(temp_env, base_parser, base_compiler, graph_compiler, selected_base_expr)
                # Now I need the shape_targets as well
                sa_expression, hcsg_commands = graph_compiler.master_struct_to_sa_and_hcsg(master_program_struct)
                # Now get the targets:
                # Convert the parameters:
                if add_dummy_node:
                    master_prog = master_program_struct["master_sa_commands"]
                    expr_cuboid_count = len([x for x in master_prog if "cuboid(" in x]) - 1
                    if expr_cuboid_count < temp_env.state_machine.master_max_prim:
                        inserted_dummy_node = True
                        s = re.split(r'[()]', master_prog[0])
                        args = [a.strip() for a in s[1].split(',')]
                        master_y_lims = float(args[1])
                        dummy_cube_name = "cube%d" % expr_cuboid_count 
                        # This can be better done.
                        dummy_commands = [{"type": "B", "symbol": "union", "canvas_count": 2}, {
                            "type": "DUMMY", "symbol": "DUMMY", "cube_name": dummy_cube_name, "master_y_lims": master_y_lims}]
                        hcsg_commands = dummy_commands + hcsg_commands

                converted_commands = sa_convert_hcsg_commands_to_cuda(hcsg_commands, device, dtype)
                graph = graph_compiler._hcsg_command_tree(converted_commands, target, reset=True, enable_subexpr_targets=True, 
                                                                add_splicing_info=True)
                # now retrieve the target for corresponding ids:

                cube_dict = master_program_struct['cube_dict']
                for cube_name, cur_dict in cube_dict.items():
                    cur_id = get_cube_id_from_hcsg(hcsg_commands, cube_name)
                    graph_id = cur_id + 1
                    node = graph.nodes[graph_id]
                    cur_dict['canonical_target'] = node["subexpr_info"]['canonical_target'].clone()
                    cur_dict['canonical_masked_iou'] = node["subexpr_info"]['canonical_masked_iou']
                
                if add_dummy_node and inserted_dummy_node:
                    # add details for dummy:
                    cur_cube_struct = {
                        "cube_name": dummy_cube_name,
                        "has_subprogram": False,
                        "subpr_sa_commands": [],
                        "subpr_hcsg_commands": [],
                        "sa_action_length": 0,
                        "splicable": True,
                        "parent": None,
                        "siblings": [],
                    }                
                    cur_id = get_cube_id_from_hcsg(hcsg_commands, dummy_cube_name)
                    graph_id = cur_id + 1
                    node = graph.nodes[graph_id]
                    cur_cube_struct['canonical_target'] = node["subexpr_info"]['canonical_target'].clone()
                    cur_cube_struct['canonical_masked_iou'] = node["subexpr_info"]['canonical_masked_iou']
                    cube_dict[dummy_cube_name]  = cur_cube_struct
                    # Create the different master program_list:
                    # Based on the bbox of the thing.
                    dummy_master_program_struct = get_master_program_struct(temp_env, base_parser, base_compiler, graph_compiler, selected_base_expr)
                    previous_sa_commands = dummy_master_program_struct["master_sa_commands"]
                    dummy_master_program_struct["master_sa_commands"] = previous_sa_commands[:-1] + node['dummy_sa_commands'] + ["$$"]
                    dummy_master_program_struct["master_hcsg_commands"].extend(node['dummy_hcsg_commands'])
                    dummy_master_program_struct["master_hcsg_commands"][0]["canvas_count"] += 1
                    dummy_master_program_struct['cube_dict'] = cube_dict
                    dummy_master_program_struct["master_sa_action_length"] += (5 + 8) #cuboid + attach
                    
                
                cube_names = []
                target_list = []
                mask_list = []
                for cube_name, cur_dict in cube_dict.items():
                    # Use masking rate to remove things.
                    masked_iou_valid  = cur_dict['canonical_masked_iou'] <= node_masking_req
                    if cur_dict['splicable'] and masked_iou_valid:
                        cube_names.append(cube_name)
                        subexpr_target = cur_dict['canonical_target'][:, :, :, 0].reshape(-1)
                        subexpr_masks = cur_dict['canonical_target'][:, :, :, 1].reshape(-1)
                        target_list.append(subexpr_target)
                        mask_list.append(subexpr_masks)
                
                # Now for each target, retrieve the k-nns
                if len(target_list) > 0:
                    target_list = th.stack(target_list, 0)
                    mask_list = th.stack(mask_list, 0)
                    candidate_list = subexpr_cache.get_candidates(
                        target_list, mask_list, cube_names, top_k)
                else:
                    candidate_list = []
                all_outputs = []
                base_expr_list = []
                # For logprob
                action_list = []
                all_lengths = []
                
                target_list = []
                action_lens = []

                zero_actions = np.zeros(temp_env.perm_max_len)
                for candidate in candidate_list:
                    node = cube_dict[candidate['cube_name']]
                    masked_iou = node['canonical_masked_iou']
                    cube_name = node['cube_name']
                    if add_dummy_node and inserted_dummy_node:
                        if cube_name == dummy_cube_name:
                            # need to add two lines to the master program:
                            master_struct_to_use = dummy_master_program_struct
                        else:
                            master_struct_to_use = master_program_struct
                    else:
                        master_struct_to_use = master_program_struct



                    # cur_subexpr = node['commands']
                    # Now this part will be different for different languages:
                    total_length = master_struct_to_use["master_sa_action_length"] - node["sa_action_length"] + candidate["sa_action_length"]
                    length_valid = total_length < max_valid_length

                    candidate_valid = length_valid and len(candidate["subpr_sa_commands"]) > 0
                    if candidate_valid:
                        # create new expression
                        replacement_sa_program = candidate["subpr_sa_commands"]
                        replacement_hcsg_commands = candidate['subpr_hcsg_commands']
                        new_master_struct = min_copy_master_program_struct(master_struct_to_use)
                        new_master_struct['cube_dict'][cube_name]["has_subprogram"] = True
                        new_master_struct['cube_dict'][cube_name]["subpr_sa_commands"] = replacement_sa_program.copy()
                        new_master_struct['cube_dict'][cube_name]["subpr_hcsg_commands"] = replacement_hcsg_commands.copy()

                        new_sa_expression, new_hcsg_commands = graph_compiler.master_struct_to_sa_and_hcsg(new_master_struct)
                        new_command_list = base_parser.parse(new_sa_expression)

                        complexity = base_compiler._get_complexity(new_command_list)
                        if complexity > temp_env.max_expression_complexity:
                            print("High complexity of %d" % complexity)
                            continue

                        base_compiler._csg_compile(new_hcsg_commands)
                        output = base_compiler._output.detach().clone()
                        output = (output <= 0)
                        all_outputs.append(output)
                        base_expr_list.append(new_sa_expression)
                        all_lengths.append(len(new_sa_expression))
                        # For log probs:
                        if use_probs:
                            cur_action = action_space.expression_to_action(new_sa_expression)
                            target_list.append(cur_action.copy())
                            action_len = cur_action.shape[0]
                            new_action = zero_actions.copy()
                            try:
                                new_action[:action_len] = cur_action
                            except:
                                print("SOME BUG HERE")
                            action_list.append(new_action)
                            action_lens.append(action_len)

                if len(all_outputs) > 0:
                    # Get all log prob:
                    if use_probs:
                        batch_size = len(all_outputs)
                        if batch_size > PROB_BATCH_LIMIT:
                            all_log_probs = get_batched_candidate_log_probs(policy, target, all_outputs, action_list, target_list, action_lens, mode_2d=False)
                        else:
                            all_log_probs = get_candidate_log_probs(policy, target, all_outputs, action_list, target_list, action_lens, mode_2d=False)
                        # Remove candidates based on scores:
                        all_outs = []
                        base_exprs = []
                        sel_log_probs = []
                        for ind, log_prob in enumerate(all_log_probs):
                            score = -(log_prob - original_logprob)
                            if reward_based_thresh:
                                logprob_threshold =  1.5 * np.exp(-2 * original_reward)
                            if score < logprob_threshold:
                                all_outs.append(all_outputs[ind])
                                base_exprs.append(base_expr_list[ind])
                                sel_log_probs.append(log_prob)
                        del all_log_probs, initial_logprob
                        all_outputs = all_outs
                        base_expr_list = base_exprs

                if len(all_outputs) > 0:

                    all_outputs = th.stack(all_outputs, 0)
                    R = get_batched_scores(all_outputs, unsqueezed_target)
                    length_tax = th.tensor(np.array(all_lengths) * length_alpha, device=R.device)
                    R = R + length_tax
                    max_ind = th.argmax(R)
                    cur_max_reward = R[max_ind].item()
                    if use_probs:
                        cur_log_prob = sel_log_probs[max_ind]
                    else:
                        cur_log_prob = 0

                    reward_improved = False
                    if cur_max_reward > max_reward:
                        reward_improved = True
                    elif cur_max_reward == max_reward:
                        new_expr = base_expr_list[max_ind]
                        if len(new_expr) < len(original_expr):
                            reward_improved = True

                    if reward_improved:
                        selected_base_expr = base_expr_list[max_ind]  # .copy()
                        max_reward = cur_max_reward  # .clone()

                        prog_obj = dict(expression=selected_base_expr.copy(),
                                        slot_id=cur_slot,
                                        target_id=cur_target,
                                        reward=max_reward,
                                        origin="CS",
                                        do_fail=False,
                                        cs_fail=False,
                                        log_prob=cur_log_prob)
                        improved_progs.append(prog_obj)
                    else:
                        # Reward is stale
                        break
                else:
                    # No candidates!
                    break
            n_progs = len(improved_progs)
            if n_progs > 0:
                selected_k = min(return_top_k, n_progs)
                for idx in range(selected_k):
                    cur_prog_obj = improved_progs[-1-idx]
                    prog_objs.append(cur_prog_obj)
                # print(selected_ntcsg_expression)
                max_reward = improved_progs[-1]['reward']
                print("reward updated to", max_reward, "from", original_reward)
                updated_rewards.append(max_reward)
                updated_length.append(len(selected_base_expr))
                num_rewrites.append(counter - 1)
                updated_logprob.append(cur_log_prob)
            else:
                updated_rewards.append(original_reward)
                updated_length.append(len(original_expr))
                updated_logprob.append(original_logprob)
                failed_keys.append(key)

            previous_length.append(len(original_expr))
            previous_rewards.append(original_reward)
            previous_logprob.append(original_logprob)

    with open(save_dir + "/cs_%d.pkl" % proc_id, 'wb') as f:
        cPickle.dump([prog_objs, updated_rewards, previous_rewards,
                     updated_length, previous_length, updated_logprob, previous_logprob, num_rewrites, failed_keys], f)
