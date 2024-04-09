import time
import torch as th
import numpy as np
import _pickle as cPickle
import os
from .code_splice_utils import get_scores, get_2d_scores
from CSG.env.reward_function import chamfer


def parallel_do(proc_id, mini_prog_dict, temp_env, save_loc, n_steps, lr, quantize_expr=True, length_alpha=0):
    
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
    updated_ious = []
    previous_ious = []
    
    start_time = time.time()
    prog_objs= []
    # th.autograd.set_detect_anomaly(True)
    sigmoid_func = th.nn.Sigmoid()
    end = np.log(10)
    start = np.log(3)
    # if amp_autocast:
    #     scaler = th.cuda.amp.GradScaler()
    scale_factors = np.exp(np.arange(start,  end, (end-start)/float(n_steps)))

    if "3D" in temp_env.language_name:
        reward_func = get_scores
    else:
        reward_func = get_2d_scores
    stt = time.time()
    failed_keys = []
    save_info = dict()
    
    for iteration, (key, value) in enumerate(mini_prog_dict):
        if iteration % 10 ==0:
            print("Proc ID", proc_id, "Cur Iter", iteration)
            print("time", time.time() - stt)
            if updated_rewards:
                print("New Avg", np.nanmean(updated_rewards),
                        "Prev Avg", np.nanmean(previous_rewards))
                print("New IoU", np.nanmean(updated_ious),
                        "Prev IoU", np.nanmean(previous_ious))
        cur_slot = key[0]
        cur_target = key[1]
        obs = temp_env.reset_to_target(cur_slot, cur_target)
        # Time:
        cur_expression = value['expression']
        # target_np = temp_env.program_generator.execute(cur_expression, return_numpy=True, return_bool=False)
        target_np = obs['obs']
        target = th.from_numpy(target_np).cuda()
        target_bool = target.bool()
        # target = target.half()
        key = (cur_slot, cur_target)
        original_reward = value['reward']
        prev_reward = value['reward']

        if "3D" in temp_env.language_name:
            target_for_reward = target
        else:
            target_for_reward = target_np
        # other option:
        # if "2D" in temp_env.language_name:
        #     real_pred_canvas = temp_env.program_generator.execute(cur_expression, return_numpy=False, return_bool=True)
        #     prev_reward = get_scores(target, real_pred_canvas)

        command_list, variable_list = parser.differentiable_parse(cur_expression, add_noise=False, noise_rate=0.0)
        if not variable_list:
            continue
        optim = th.optim.Adam(variable_list, lr=lr)
        best_reward = prev_reward - 0.0001
        copied_command_list = None
        # Try with autocast:
        save_info[key] = dict(final_reward=0, reward_delta=0,
                              program_seq=[], reward_seq=[])
        # positive_ratio = (target == 1).float().sum()/target.numel()
        # balance_ceoff = positive_ratio ** 0.0
        # balance_mat = th.where(target == 1, 1/balance_ceoff, balance_ceoff)
        saver_program_seq = []
        saver_reward_seq = []
        for i in range(n_steps):
            # with th.cuda.amp.autocast():
            # Make this traced

            scale_factor = scale_factors[i]

            compiler._compile(command_list)
            output_sdf = compiler._output
            # z = (output_sdf.detach() <= 0)
            output_tanh = th.tanh(output_sdf * scale_factor)
            output_shape = sigmoid_func(-output_tanh * scale_factor)
            # Weighted MSE?
            # output_loss = th.sum((output_shape - target) ** 2 * balance_mat) / target.numel()
            output_loss = th.nn.functional.mse_loss(output_shape, target)

            optim.zero_grad()
            output_loss.backward()
            optim.step()


            # print(reward)
            command_list = parser.rebuild_command_list(cur_expression, variable_list, command_list)
            temp_expression = parser.get_expression(command_list, quantize=quantize_expr, resolution=resolution)
            with th.no_grad():
                real_pred_canvas = temp_env.program_generator.execute(temp_expression, return_numpy=False, return_bool=True)
            # reward = reward_func(pred_canvas, target_for_reward) # + length_alpha * len(cur_expression)
            reward = get_scores(real_pred_canvas, target_bool)# .item()
            # print(output_loss.item(), reward, prev_reward)
            reward = reward + length_alpha * len(temp_expression)
            if reward > best_reward:
                best_reward = reward.item()
                copied_command_list = parser.copy_command_list(command_list)
                
            # Saver Info:
            saver_cur_prog = parser.copy_command_list(command_list)
            expression = parser.get_expression(saver_cur_prog, quantize=quantize_expr, resolution=resolution)
            saver_program_seq.append(expression)
            saver_reward_seq.append(reward.item())
            
        if copied_command_list:
            best_expression = parser.get_expression(copied_command_list, quantize=quantize_expr, resolution=resolution)
            # best_expression = parser.get_expression(copied_command_list, quantize=quantize_expr, resolution=resolution, clip=False)
            if "3D" in temp_env.language_name:
                real_pred_canvas = temp_env.program_generator.execute(best_expression, return_numpy=False, return_bool=True)
                real_reward = get_scores(real_pred_canvas, target_bool).item()
            else:
                real_pred_canvas = temp_env.program_generator.execute(best_expression, return_numpy=True, return_bool=True)
                real_reward = 100 - chamfer(target_np[None, :, :], real_pred_canvas[None, :, :])[0]
            real_reward = real_reward + length_alpha * len(best_expression)
        else:
           real_reward = original_reward
           
        if real_reward > original_reward:
            # print("updating to", real_reward, "previously", prev_reward)
            updated_rewards.append(real_reward)
            prog_obj = dict(expression=best_expression,
                             slot_id=cur_slot,
                             target_id=cur_target,
                             reward=real_reward,
                             origin="DO",
                             do_fail=True, # Don't do for this since it is locally optimum
                             cs_fail=False,
                             log_prob=0)
            prog_objs.append(prog_obj)
            updated_ious.append(real_reward - length_alpha * len(best_expression))
        else:
            print("best reward only", real_reward, "previously", original_reward)
            updated_rewards.append(original_reward)
            updated_ious.append(original_reward - length_alpha * len(value['expression']))

        previous_rewards.append(original_reward)
        previous_ious.append(original_reward - length_alpha * len(value['expression']))
        
        save_info[key]['final_reward'] = real_reward
        save_info[key]['reward_delta'] = real_reward - original_reward
        save_info[key]['program_seq'] = saver_program_seq
        save_info[key]['reward_seq'] = saver_reward_seq
            
    if not updated_rewards:
        updated_rewards = [0]
        previous_rewards = [0]
    
    # with open(save_loc + "/do_log_%d.pkl" % proc_id, 'wb') as f:
    #     cPickle.dump(save_info, f)
    
    with open(save_loc + "/do_%d.pkl" % proc_id, 'wb') as f:
        cPickle.dump([prog_objs, updated_rewards, previous_rewards, failed_keys], f)