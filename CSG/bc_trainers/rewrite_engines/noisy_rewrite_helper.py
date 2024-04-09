from telnetlib import theNULL
import time
import torch as th
import numpy as np
import _pickle as cPickle
import os
from .code_splice_utils import get_scores, get_2d_scores, get_batched_scores
from CSG.env.reward_function import chamfer
from itertools import compress


def parallel_nr(proc_id, mini_prog_dict, temp_env, save_loc, 
                n_augs, noise_rate, reward_threshold_pow, 
                numeric_sigma=0.05, discrete_delta=1,
                max_augs_per_sample=3, length_alpha=0):
    
    th.backends.cudnn.benchmark = True
    compiler = temp_env.program_generator.compiler
    parser = temp_env.program_generator.parser
    resolution =temp_env.action_space.resolution
    device = th.device("cuda")
    dtype = th.float32
    compiler.set_to_cuda()
    compiler.set_to_full()
    compiler.reset()

    parser.set_device(device)
    parser.set_tensor_type(dtype)
    
    updated_rewards = []
    previous_rewards = []
    
    start_time = time.time()
    prog_objs= []
    # th.autograd.set_detect_anomaly(True)
    if "3D" in temp_env.language_name:
        reward_func = get_scores
    else:
        reward_func = get_2d_scores
    stt = time.time()
    uniform_sampler = th.distributions.uniform.Uniform(low=1e-9, high=1)
    
    # Create it based on rewards
    # change_rates = []
    # rewards = []
    # success_rates = []
    # best_rewards = []
    
    action_softmatrix, mask_softmatrix = temp_env.action_space.create_action_softmax_and_mask(numeric_sigma=numeric_sigma, discrete_delta=discrete_delta)
    action_softmatrix = th.tensor(action_softmatrix, device=device)
    mask_softmatrix = th.tensor(mask_softmatrix, device=device)
    
    for iteration, (key, value) in enumerate(mini_prog_dict):
        
        if iteration % 100 ==0:
            print("Proc ID", proc_id, "Cur Iter", iteration)
            print("time", time.time() - stt)
            if updated_rewards:
                print("New Avg", np.nanmean(updated_rewards),
                        "Prev Avg", np.nanmean(previous_rewards))
                print("N Progs", len(prog_objs))
        cur_slot = key[0]
        cur_target = key[1]
        obs = temp_env.reset_to_target(cur_slot, cur_target)
        # Time:
        cur_expression = value['expression']
        # target_np = temp_env.program_generator.execute(cur_expression, return_numpy=True, return_bool=False)
        target_np = obs['obs']
        target = th.from_numpy(target_np).cuda()
        target_bool = target.bool()
        unsqueezed_target = target_bool.unsqueeze(0)
        # target = target.half()
        key = (cur_slot, cur_target)
        original_reward = value['reward']
        bs_reward = value['bs_reward']
        # To understand the behaviour set reward manually.
        
        
        threshold = max(original_reward ** reward_threshold_pow, bs_reward)
        reward_relation = 1 - original_reward
               
        # 1) Get the actions
        actions = temp_env.action_space.expression_to_action(cur_expression)
        # 2) Convert to Soft Distribution, N, L, 75
        actions = th.tensor(actions, device=device, dtype=th.long)
        action_sf = th.index_select(action_softmatrix, dim=0, index=actions)
        mask_sf = th.index_select(mask_softmatrix, dim=0, index=actions)
        action_sf = action_sf.unsqueeze(0)
        action_sf = action_sf.repeat([n_augs, 1, 1])
        # 3) ADD Gumbel Noise N, L, 75
        u_noise = uniform_sampler.rsample(action_sf.shape).to(action_sf.device)
        gumbel_noise = -th.log(-th.log(u_noise)) * mask_sf
        noisy_action_sf = action_sf + gumbel_noise * noise_rate * reward_relation
        # 4) Retrieve argmax actions N, L
        noisy_actions = th.argmax(noisy_action_sf, 2)
        # 5) Evaluate performance
        # remove exact_matches
        # Measures: 
        # % change in program
        # change_rate = (noisy_actions != actions).float().mean(1).mean(0).item()
        # change_rates.append(change_rate)
        # rewards.append(original_reward)
        bool_invalid = th.all(noisy_actions == actions, 1)
        noisy_actions = noisy_actions.cpu().data.numpy()
        expression_list = []
        execution_list = []
        
        for i in range(n_augs):
            if bool_invalid[i]:
                continue
            else:
                # get expression
                # get execution
                expr = temp_env.action_space.action_to_expression(noisy_actions[i])
                command_list = parser.parse(expr)
                compiler._compile(command_list)
                output_sdf = compiler._output
                output_sdf = (output_sdf.detach() <= 0)
                expression_list.append(expr)
                execution_list.append(output_sdf)
        if execution_list:
            executions = th.stack(execution_list, 0)
            
            if "3D" in temp_env.language_name:
                R = get_batched_scores(executions, unsqueezed_target)
            else:
                #TBD
                R = 100 - chamfer(temp_target_np, real_pred_canvas)
            # filter 
            best_indices = (-R).argsort()
            R = R.cpu().data.numpy() + length_alpha * len(expr)
            
            # argsort? and store top 3
            # success_rate = (R > threshold).sum()
            # success_rates.append(success_rate)
            # best_rewards.append(R[best_indices[0]])
            counts = min(max_augs_per_sample, len(executions))
            for ind in range(counts):
                max_ind = best_indices[ind]
                cur_reward = R[max_ind]
                if cur_reward > threshold:
                    updated_rewards.append(cur_reward)
                    previous_rewards.append(original_reward)
                    
                    prog_obj = dict(expression=expression_list[max_ind],
                                    slot_id=cur_slot,
                                    target_id=cur_target,
                                    reward=cur_reward,
                                    origin="NR",
                                    do_fail=False, # Don't do for this since it is locally optimum
                                    cs_fail=False,
                                    log_prob=0)
                    prog_objs.append(prog_obj)
            
            # print(R.max(), original_reward, counts)
        else:
            print("failed augmentation")
            # success_rates.append(0)
            # best_rewards.append(original_reward)
         
    if not updated_rewards:
        updated_rewards = [0]
        previous_rewards = [0]
        
    
    failed_keys = []
    with open(save_loc + "/nr_%d.pkl" % proc_id, 'wb') as f:
        cPickle.dump([prog_objs, updated_rewards, previous_rewards, failed_keys], f)
  

    # with open(save_loc + "/stats_%d.pkl" % proc_id, 'wb') as f:
    #     cPickle.dump([change_rates, rewards, success_rates, best_rewards], f)