from telnetlib import theNULL
import time
import torch as th
import numpy as np
import _pickle as cPickle
import os
from .code_splice_utils import get_scores, get_2d_scores
from CSG.env.reward_function import chamfer
import open3d as o3d
from CSG.env.csg3d.linear_compiler import LinearMCSG3DCompiler

POINT_CLOUD_LOC = "/home/aditya/fcsg_gt_points/"

def parallel_do(proc_id, mini_prog_dict, temp_env, save_loc, n_steps, lr, quantize_expr=True, amp_autocast=False, length_alpha=0):
    
    th.backends.cudnn.benchmark = True
    # compiler = temp_env.program_generator.compiler
    
    ## HACK
    old_compiler = temp_env.program_generator.compiler
    compiler = LinearMCSG3DCompiler(resolution=old_compiler.resolution, scale=old_compiler.scale, 
                                    scad_resolution=old_compiler.scad_resolution, device=old_compiler.device, draw_mode=old_compiler.draw.mode)
    
    
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
    sigmoid_func = th.nn.Sigmoid()
    end = np.log(10)
    start = np.log(3)
    if amp_autocast:
        scaler = th.cuda.amp.GradScaler()
    scale_factors = np.exp(np.arange(start,  end, (end-start)/float(n_steps)))

    if "3D" in temp_env.language_name:
        reward_func = get_scores
    else:
        reward_func = get_2d_scores
    stt = time.time()
    failed_keys = []
    
    ## HACK
    # Load
    
    for iteration, (key, value) in enumerate(mini_prog_dict):
        compiler.draw.reset()
        compiler.reset()
        if iteration % 10 ==0:
            print("Proc ID", proc_id, "Cur Iter", iteration)
            print("time", time.time() - stt)
            if updated_rewards:
                print("New Avg", np.nanmean(updated_rewards),
                        "Prev Avg", np.nanmean(previous_rewards))
        cur_slot = key[0]
        cur_target = key[1]
        obs = temp_env.reset_to_target(cur_slot, cur_target)
        # Time:
        cur_expression = value['expression']
        # target_np = temp_env.program_generator.execute(cur_expression, return_numpy=True, return_bool=False)
        target_np = obs['obs']
        ## HACK 
        target_np =target_np.reshape(-1)
        target = th.from_numpy(target_np).cuda()
        target_bool = target.bool()
        # target = target.half()

        if "3D" in temp_env.language_name:
            target_for_reward = target
        else:
            target_for_reward = target_np
            
        key = (cur_slot, cur_target)
        # value['reward'] = 0
        original_reward = value['reward']
        prev_reward = 0# value['reward']
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
        
        # load point cloud:
        gt_pc_name = os.path.join(POINT_CLOUD_LOC, "%s_%s.pkl" % (cur_slot, cur_target))
        samples = cPickle.load(open(gt_pc_name, "rb"))
        gt_points = samples["points"].astype(np.float32)
        # surface_selection_index = np.random.randint(0, points.shape[0], num_surface_points)
        gt_points[:, 0] = -gt_points[:, 0]
        gt_points = th.tensor(gt_points, device=device, dtype=dtype)
        # extend points and target
        new_targets = samples['occupancies']
        new_targets = np.unpackbits(new_targets)
        new_targets = th.tensor(new_targets, device=device).float()
        n_positives = target.sum()
        n_negatives = target.shape[0] - n_positives
        # balanced loss:
        # weights = th.zeros_like(target)
        # weights[target==1] = n_negatives/target.shape[0]
        # weights[target==0] = n_positives/target.shape[0]
        
        # coord_array = None
        all_info = th.cat([gt_points, new_targets.unsqueeze(1)], axis=-1)
        
        inner_points = all_info[all_info[:,-1]==1]
        outer_points = all_info[all_info[:,-1]==0]
        num_sample_points = 2048
        for i in range(n_steps):
            # with th.cuda.amp.autocast():
            # Make this traced
            inner_index = np.random.randint(0, inner_points.shape[0], num_sample_points//2)
            outer_index = np.random.randint(0, outer_points.shape[0], num_sample_points//2)
            samples = th.cat([inner_points[inner_index], outer_points[outer_index]], axis=0)
            temp_target = th.cat([target, samples[:, -1]])# .cuda()
            # temp_target = samples[:, -1]
            # target = target.half()
            target_for_reward = temp_target.bool()
            base_coords = compiler.draw.base_coords
            coord_array = th.cat([base_coords, samples[:, :-1]], 0)
            # coord_array = samples[:, :-1]

            scale_factor = scale_factors[i]

            compiler._compile(command_list, coord_array=coord_array)
            output_sdf = compiler._output
            z = (output_sdf.detach() <= 0)
            output_tanh = th.tanh(output_sdf * scale_factor)
            output_shape = sigmoid_func(-output_tanh * scale_factor)
            # output_loss = ((output_shape - target) ** 2) * weights
            # output_loss = output_loss.mean()
            output_loss = th.nn.functional.mse_loss(output_shape, temp_target)

            optim.zero_grad()
            output_loss.backward()
            optim.step()
            compiler.draw.reset()
            compiler.reset()


            pred_canvas = (output_sdf.detach() <= 0)# .cpu().data.numpy()

            # reward = reward_func(pred_canvas, target_for_reward)
            reward = th.logical_and(pred_canvas, target_for_reward).sum() / th.logical_or(pred_canvas, target_for_reward).sum()
            reward = reward + length_alpha * len(cur_expression)
            
            
            command_list = parser.rebuild_command_list(cur_expression, variable_list, command_list)
            if reward > best_reward:
                # print(output_loss.item(), reward, prev_reward)
                best_reward = reward.item()
                copied_command_list = parser.copy_command_list(command_list)

        if copied_command_list:
            best_expression = parser.get_expression(copied_command_list, quantize=quantize_expr, resolution=resolution)
            if "3D" in temp_env.language_name:
                compiler.draw.reset()
                compiler.reset()
                command_list = parser.parse(best_expression)
                compiler._compile(command_list)
                output_sdf = compiler._output
                real_pred_canvas = (output_sdf.detach() <= 0)
                
                target = th.from_numpy(target_np).cuda()
                target_bool = target.bool()
                real_reward = th.logical_and(real_pred_canvas, target_bool).sum() / th.logical_or(real_pred_canvas, target_bool).sum()
                real_reward = real_reward.item()
                
            real_reward = real_reward + length_alpha * len(best_expression)
            

        else:
           real_reward = original_reward
        print(real_reward)
        if real_reward > original_reward:
            print("updating to", real_reward, "previously", original_reward)
            updated_rewards.append(real_reward)
        else:
            print("best reward only", real_reward, "previously", original_reward)
            updated_rewards.append(original_reward)

        previous_rewards.append(original_reward)
        
        real_reward = max(real_reward, original_reward + 0.001)
        prog_obj = dict(expression=best_expression,
                            slot_id=cur_slot,
                            target_id=cur_target,
                            reward=real_reward,
                            origin="DO",
                            do_fail=True, # Don't do for this since it is locally optimum
                            cs_fail=False,
                            log_prob=0)
        prog_objs.append(prog_obj)
    if not updated_rewards:
        updated_rewards = [0]
        previous_rewards = [0]
        
    
    with open(save_loc + "/do_%d.pkl" % proc_id, 'wb') as f:
        cPickle.dump([prog_objs, updated_rewards, previous_rewards, failed_keys], f)
  
