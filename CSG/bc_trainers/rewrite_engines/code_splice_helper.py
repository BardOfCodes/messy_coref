import math
from re import S
import torch as th
from CSG.env.csg3d.languages import MCSG3DParser, PCSG3DParser, GraphicalMCSG3DCompiler, MCSG3DCompiler
from CSG.env.csg2d.languages import MCSG2DParser, PCSG2DParser, GraphicalMCSG2DCompiler, MCSG2DCompiler

from CSG.env.csg3d.parser_utils import pcsg_to_ntcsg, ntcsg_to_pcsg
from CSG.env.csg3d.languages import boolean_commands
import time
import random
import numpy as np
import _pickle as cPickle
import os
from CSG.utils.profile_utils import profileit
from CSG.env.reward_function import chamfer

from .code_splice_utils import get_masked_scores, get_new_command, get_scores, get_2d_scores, bool_count, get_batched_scores, distill_transform_chains
from .graph_sweep_helper import gs_singular_parse

DUMMY_TYPE = "DUMMY"
PROB_BATCH_LIMIT = 100

# @profileit("../logs/cs.profile")
def parallel_code_splice(proc_id, mini_prog_dict, temp_env, subexpr_cache, save_dir, top_k=15,
                         max_boolean_count=11, rewrite_limit=20, node_masking_req=1.0,
                         add_dummy_node=False, apply_gs=True, return_top_k=1, quantize_expr=True,
                         use_canonical=True, policy=None, use_probs=False, logprob_threshold=0.1, reward_based_thresh=False,
                         valid_nodes=["B", "D", "DUMMY"], higher_language=False, max_valid_length=96, max_complexity=16,
                         length_alpha=0):

    device, dtype = th.device("cuda"), th.float16
    th.backends.cudnn.benchmark = True
    temp_env.program_generator.set_execution_mode(th.device("cuda"), dtype)

    base_parser = temp_env.program_generator.parser
    action_resolution = temp_env.action_space.resolution
    action_space = temp_env.action_space
    if "3D" in temp_env.language_name:
        mcsg_parser = MCSG3DParser(temp_env.program_generator.parser.module_path)
        graph_compiler = GraphicalMCSG3DCompiler(resolution=temp_env.program_generator.compiler.resolution,
                                    scale=temp_env.program_generator.compiler.scale,
                                    draw_mode=temp_env.program_generator.compiler.draw.mode)
    else:
        mcsg_parser = MCSG2DParser(temp_env.program_generator.parser.module_path)
        graph_compiler = GraphicalMCSG2DCompiler(resolution=temp_env.program_generator.compiler.resolution,
                                    scale=temp_env.program_generator.compiler.scale,
                                    draw_mode=temp_env.program_generator.compiler.draw.mode)


    graph_compiler.set_to_cuda()
    # graph_compiler.set_to_half()
    graph_compiler.set_to_full()
    graph_compiler.reset()

    mcsg_parser.set_device(device)
    mcsg_parser.set_tensor_type(dtype)


    updated_rewards = []
    previous_rewards = []

    previous_length = []
    updated_length = []

    previous_logprob = []
    updated_logprob = []
    prog_objs = []
    num_rewrites = []
    zero_actions = np.zeros(temp_env.perm_max_len)

    st = time.time()
    tot_programs = len(mini_prog_dict)


    if "3D" in temp_env.language_name:
        reward_func = get_scores
        mode_2d = False
    else:
        mode_2d = True
        reward_func = get_2d_scores

    failed_keys = []
    save_info = dict()
    with th.no_grad():
        # with th.cuda.amp.autocast():
        for iteration, (key, value) in enumerate(mini_prog_dict):
            if iteration % 10 == 0:
                print("cur iteration %d of %d. Cur Time %f" %
                    (iteration, tot_programs, time.time() - st))
                if updated_rewards:
                    print("New Avg", np.nanmean(updated_rewards),
                            "Prev Avg", np.nanmean(previous_rewards))
                    # iou = reward + length_alpha * length 
                    # prev_iou = [x - length_alpha * previous_length[ind] for ind, x in enumerate(previous_rewards)]
                    # new_iou = [x - length_alpha * updated_length[ind] for ind, x in enumerate(updated_rewards)]
                    # print("New Avg IOU", np.nanmean(new_iou))
                    # print("Prev Avg IOU", np.nanmean(prev_iou))
            improved_progs = []
            cur_slot, cur_target, origin = key
            basecsg_expr = value['expression']
            original_expr = basecsg_expr.copy()
            selected_basecsg_expr = basecsg_expr.copy()
            mcsg_expr = base_parser.convert_to_mcsg(basecsg_expr)

            obs = temp_env.reset_to_target(cur_slot, cur_target)
            target_np = obs['obs']
            target = th.from_numpy(target_np).cuda().bool()
            unsqueezed_target = target.unsqueeze(0)
            counter = 0
            # Get initial log prob
            if "3D" in temp_env.language_name:
                target_for_reward = target
            else:
                target_for_reward = target_np
                
            save_info[key] = dict(final_reward=0, reward_delta=0,
                                program_seq=[], original_expr_seq=[], reward_seq=[])
            saver_program_seq = []
            saver_reward_seq = []
            saver_original_program_seq = []

            while(counter <= rewrite_limit):
                counter += 1
                expr_bool_count = len(
                    [x for x in mcsg_expr if x in boolean_commands])
                command_list = mcsg_parser.parse(mcsg_expr)
                graph_compiler._compile(command_list)
                output = graph_compiler._output.clone()
                output = (output <= 0)

                if use_probs:
                    # Get the latent execution:
                    initial_logprob = get_base_log_prob(policy, selected_basecsg_expr, action_space, output, target, temp_env.perm_max_len)
                    # print("Current Log Prob", initial_logprob)
                else:
                    initial_logprob = 0


                R = reward_func(output, target_for_reward)
                if not mode_2d:
                    R = R.item()
                # R = get_scores(output, target)
                R = R + length_alpha * len(selected_basecsg_expr)
                initial_reward = R
                max_reward = initial_reward  # .item()
                if counter == 1:
                    original_reward = initial_reward
                    original_logprob = initial_logprob

                if add_dummy_node:
                    if expr_bool_count < (max_boolean_count - 1):
                        # This can be better done.
                        dummy_commands = [{"type": "B", "symbol": "union"}]
                        command_list = dummy_commands + command_list
                        command_list = command_list + [{"type": "DUMMY", "symbol": "DUMMY"}]
                graph = graph_compiler.command_tree(command_list, target, enable_subexpr_targets=True,
                                                        add_splicing_info=True)

                graph_nodes = [graph.nodes[i] for i in graph.nodes]
                node_ids = []
                target_list = []
                mask_list = []
                for node_id, node in enumerate(graph_nodes):
                    # Use masking rate to remove things.

                    if node['type'] in valid_nodes:
                        masking_rate = node['subexpr_info']['canonical_masking_rate']
                        if masking_rate <= node_masking_req:
                            # search for the replacement:
                            node_ids.append(node_id)
                            if mode_2d:
                                if use_canonical:

                                    subexpr_target = node['subexpr_info']['canonical_target'][:, :, 0].reshape(-1)
                                    subexpr_masks = node['subexpr_info']['canonical_target'][:, :, 1].reshape(-1)
                                else:
                                    subexpr_target = node['subexpr_info']['expr_target'][:, :, 0].reshape(-1)
                                    subexpr_masks = node['subexpr_info']['expr_target'][:, :, 1].reshape(-1)
                            else:
                                if use_canonical:

                                    subexpr_target = node['subexpr_info']['canonical_target'][:, :, :, 0].reshape(-1)
                                    subexpr_masks = node['subexpr_info']['canonical_target'][:, :, :, 1].reshape(-1)
                                else:
                                    subexpr_target = node['subexpr_info']['expr_target'][:, :, :, 0].reshape(-1)
                                    subexpr_masks = node['subexpr_info']['expr_target'][:, :, :, 1].reshape(-1)
                            target_list.append(subexpr_target)
                            mask_list.append(subexpr_masks)
                # Now for each target, retrieve the k-nns
                if len(target_list) > 0:
                    target_list = th.stack(target_list, 0)
                    mask_list = th.stack(mask_list, 0)
                    candidate_list = subexpr_cache.get_candidates(
                        target_list, mask_list, node_ids, top_k)
                else:
                    candidate_list = []

                all_outputs = []
                all_lengths = []
                base_expr_list = []
                # For logprob
                action_list = []
                target_list = []
                action_lens = []

                for candidate in candidate_list:
                    if not candidate['commands']:
                        continue
                    node = graph_nodes[candidate['node_id']]
                    masked_iou = node['subexpr_info']['canonical_masked_iou']
                    cur_subexpr = node['subexpr_info']['commands']
                    # Now this part will be different for different languages:
                    if higher_language:
                        # length is checked ter
                        length_valid = True
                    else:
                        cur_subexpr_n_bool = len(
                            [x for x in cur_subexpr if x['type'] == "B"])
                        total_bool_allowed = (
                            max_boolean_count - expr_bool_count) + cur_subexpr_n_bool
                        if add_dummy_node:
                            if candidate['node_id'] == 2:
                                total_bool_allowed -= 1
                            length_valid = candidate['bool_count'] <= total_bool_allowed

                    candidate_valid = candidate['masked_iou'] > (masked_iou - 0.1) and length_valid
                    # candidate_valid = length_valid
                    if candidate_valid:
                    # if True:
                        # This part as well.
                        command_inds = node['subexpr_info']['command_ids']
                        prev_canonical_commands = node['subexpr_info']['canonical_commands']
                        new_command_list = get_new_command(command_list, command_inds, prev_canonical_commands,
                            candidate, use_canonical=use_canonical)
                        if higher_language:

                            if '3D' in temp_env.language_name:
                                new_command_list = distill_transform_chains(new_command_list, device, dtype)
                            else:
                                new_command_list = distill_transform_chains(new_command_list, device, dtype, mode="2D")
                            # new_command_list = distill_transform_chains(new_command_list, device, dtype)

                        if add_dummy_node:
                            command_types = [x['type'] for x in new_command_list]
                            if DUMMY_TYPE in command_types:
                                ind = command_types.index(DUMMY_TYPE)
                                # print("dummy index", ind)
                                new_command_list = new_command_list[1:ind] + new_command_list[ind + 1:]
                        # Perform a complexity check:
                        if higher_language:
                            try:
                                complexity = graph_compiler._get_complexity(new_command_list)
                            except:
                                print("FAILED TO PARSE NEW CMD LIST FOR COMPLEXITY")
                                continue
                            if complexity > max_complexity:
                                # reject
                                print("candidate rejected as complexity is %d" % complexity)
                                continue
                        base_expr = base_parser.get_expression(new_command_list, clip=True,
                                                            quantize=quantize_expr, resolution=action_resolution)
                        if higher_language:
                            cur_action = action_space.expression_to_action(base_expr)
                            action_len = cur_action.shape[0]
                            length_valid = action_len < max_valid_length
                            if not length_valid:
                                continue
                        
                        temp_command_list = base_parser.parse(base_expr)
                        graph_compiler._compile(temp_command_list)
                        output = graph_compiler._output.detach().clone()
                        output = (output <= 0)
                        all_outputs.append(output)
                        base_expr_list.append(base_expr)
                        all_lengths.append(len(base_expr))
                        # For log probs:
                        if use_probs:
                            cur_action = action_space.expression_to_action(base_expr)
                            target_list.append(cur_action.copy())
                            action_len = cur_action.shape[0]
                            new_action = zero_actions.copy()
                            new_action[:action_len] = cur_action
                            action_list.append(new_action)
                            action_lens.append(action_len)
                
                        

                if len(all_outputs) > 0:
                    # Get all log prob:
                    if use_probs:
                        all_log_probs = get_batched_candidate_log_probs(policy, target, all_outputs, action_list, target_list, action_lens, mode_2d=mode_2d)
                        # Remove candidates based on scores:
                        all_outs = []
                        base_exprs = []
                        sel_log_probs = []
                        all_lengths_real = []
                        for ind, log_prob in enumerate(all_log_probs):
                            score = -(log_prob - original_logprob)
                            if reward_based_thresh:
                                logprob_threshold =  1.5 * np.exp(-2 * original_reward)
                            if score < logprob_threshold:
                                all_outs.append(all_outputs[ind])
                                base_exprs.append(base_expr_list[ind])
                                sel_log_probs.append(log_prob)
                                all_lengths_real.append(all_lengths[ind])
                        del all_log_probs, initial_logprob
                        all_outputs = all_outs
                        base_expr_list = base_exprs
                        all_lengths = all_lengths_real

                if len(all_outputs) > 0:

                    all_outputs = th.stack(all_outputs, 0)
                    if "3D" in temp_env.language_name:
                        R = get_batched_scores(all_outputs, unsqueezed_target)
                        length_tax = th.tensor(np.array(all_lengths) * length_alpha, device=R.device)
                        R = R + length_tax
                        # Add the length cost
                        max_ind = th.argmax(R)
                        cur_max_reward = R[max_ind].item()
                    else:
                        real_pred_canvas = all_outputs.cpu().numpy()
                        temp_target_np = unsqueezed_target.expand(real_pred_canvas.shape[0], -1, -1)
                        temp_target_np = temp_target_np.cpu().numpy()
                        R = 100 - chamfer(temp_target_np, real_pred_canvas)
                        length_tax = np.array(all_lengths) * length_alpha
                        R = R + length_tax
                        max_ind = np.argmax(R)
                        cur_max_reward = R[max_ind]

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
                    #saver info:
                    saver_program_seq.extend(base_expr_list)
                    saver_reward_seq.extend(R.cpu().numpy().tolist())
                    saver_original_program_seq.extend([selected_basecsg_expr] * len(base_expr_list))
                    
                    if reward_improved:
                        selected_basecsg_expr = base_expr_list[max_ind]  # .copy()
                        # TODO or not?
                        if apply_gs:
                            selected_basecsg_expr, _, _ = gs_singular_parse(selected_basecsg_expr, target,
                                                                            base_parser, graph_compiler, reward_threshold=0.0)

                                            
                        if "3D" in temp_env.language_name:
                            real_pred_canvas = temp_env.program_generator.execute(selected_basecsg_expr, return_numpy=False, return_bool=True)
                            max_reward = get_scores(real_pred_canvas, target).item()
                            max_reward = max_reward + length_alpha * len(selected_basecsg_expr)
                        else:
                            real_pred_canvas = temp_env.program_generator.execute(selected_basecsg_expr, return_numpy=True, return_bool=True)
                            max_reward = 100 - chamfer(target_np[None, :, :], real_pred_canvas[None, :, :])[0]

                        # max_reward = cur_max_reward  # .clone()

                        prog_obj = dict(expression=selected_basecsg_expr.copy(),
                                        slot_id=cur_slot,
                                        target_id=cur_target,
                                        reward=max_reward,
                                        origin="CS",
                                        do_fail=False,
                                        cs_fail=False, # since the process is stochastic, allow rerun.
                                        log_prob=cur_log_prob)
                        improved_progs.append(prog_obj)
                        mcsg_expr = base_parser.convert_to_mcsg(
                            selected_basecsg_expr)
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
                updated_rewards.append(max_reward)
                updated_length.append(len(selected_basecsg_expr))
                num_rewrites.append(counter)
                updated_logprob.append(cur_log_prob)
            else:
                updated_rewards.append(original_reward)
                updated_length.append(len(original_expr))
                updated_logprob.append(original_logprob)
                failed_keys.append(key)
            
            save_info[key]['final_reward'] = max_reward
            save_info[key]['reward_delta'] = max_reward - original_reward
            save_info[key]['program_seq'] = saver_program_seq
            save_info[key]['original_expr_seq'] = saver_original_program_seq
            save_info[key]['reward_seq'] = saver_reward_seq

            previous_length.append(len(original_expr))
            previous_rewards.append(original_reward)
            previous_logprob.append(original_logprob)

    with open(save_dir + "/cs_log_%d.pkl" % proc_id, 'wb') as f:
        cPickle.dump(save_info, f)
        
    with open(save_dir + "/cs_%d.pkl" % proc_id, 'wb') as f:
        cPickle.dump([prog_objs, updated_rewards, previous_rewards,
                     updated_length, previous_length, updated_logprob, previous_logprob, num_rewrites, failed_keys], f)


def get_base_log_prob(policy, basecsg_expr, action_space, output, target_canvas, max_len):
    input_obs = th.stack([target_canvas, output], 0).float()
    actions = th.zeros(size=(max_len,), device="cuda")
    new_actions = action_space.expression_to_action(basecsg_expr)
    action_len = new_actions.shape[0]
    new_actions = th.from_numpy(new_actions).cuda()
    actions[:action_len] = new_actions
    actions = th.stack([actions, actions], 0)
    # zeros_ = th.stack([zeros_, zeros_], 0).long()
    final_input = {
                "obs": input_obs,
                "previous_steps" : actions,
                "cur_step" : th.LongTensor([action_len, action_len]).cuda(),
            }
    targets = th.cat([new_actions, new_actions], 0)
    with th.no_grad():
        # with th.cuda.amp.autocast():
            _, all_log_prob, _, _ = policy.tformer_evaluate_actions_and_acc(final_input, targets)
    original_log_probs = all_log_prob.mean().item()
    return original_log_probs


def get_candidate_log_probs(policy, target, all_outputs, action_list, target_list, orig_action_lens, mode_2d=False):
    latent_executes = th.stack(all_outputs, 0)
    batch_size = latent_executes.shape[0]
    if mode_2d:
        originals = target.unsqueeze(0).expand(batch_size, -1, -1)
    else:
        originals = target.unsqueeze(0).expand(batch_size, -1, -1, -1)
    actions = np.array(action_list).astype(np.int32)
    actions = th.from_numpy(actions).cuda()
    action_lens = np.array(orig_action_lens)
    action_lens = th.from_numpy(action_lens).cuda()
    targets = np.concatenate(target_list, 0)
    targets = th.from_numpy(targets).cuda()
    _jump = targets.shape[0]
    final_input = {
                "obs": th.cat([originals, latent_executes], 0).float(),
                "previous_steps" : th.cat([actions, actions], 0),
                "cur_step" : th.cat([action_lens, action_lens], 0),
            }
    targets = th.cat([targets, targets], 0)

    with th.no_grad():
        # with th.cuda.amp.autocast():
            _, all_log_prob, _, _ = policy.tformer_evaluate_actions_and_acc(final_input, targets)
    cum_index = 0
    avg_log_prob_list = []
    for index in range(batch_size):
        # for each sum of the right ones.
        action_len = orig_action_lens[index]
        indices = (cum_index, cum_index + action_len)
        avg_log_prob = all_log_prob[indices[0]: indices[1]].sum() + all_log_prob[_jump + indices[0]: _jump + indices[1]].sum()
        avg_log_prob /= float(2 *action_len)
        avg_log_prob_list.append(avg_log_prob)
        cum_index += action_len
    avg_log_prob_list = th.stack(avg_log_prob_list, 0).cpu().numpy()
    return avg_log_prob_list

def get_batched_candidate_log_probs(policy, target, all_outputs, action_list, target_list, action_lens, batch_size=PROB_BATCH_LIMIT, mode_2d=False):
    n_samples = len(all_outputs)
    n_batch = math.ceil(n_samples/batch_size)
    all_log_probs = []
    for batch_idx in range(n_batch):
        probs_1 = get_candidate_log_probs(policy, target, all_outputs[batch_idx * batch_size: (batch_idx + 1) * batch_size], 
                                          action_list[batch_idx * batch_size: (batch_idx + 1) * batch_size], 
                                          target_list[batch_idx * batch_size: (batch_idx + 1) * batch_size], 
                                          action_lens[batch_idx * batch_size: (batch_idx + 1) * batch_size],
                                          mode_2d)
        all_log_probs.append(probs_1)
    all_log_probs = np.concatenate(all_log_probs, 0)
    return all_log_probs
