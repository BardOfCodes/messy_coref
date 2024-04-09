import time
import numpy as np
import torch
import _pickle as cPickle
# from CSG.env.csg2d.differentiable_stack import myround

NOISE_LIM = 8
def get_opt_programs(selected_programs, selected_distances, batch_diff_stack, N_STEPS, LR):
    
    batch_diff_stack.generate_batch_stack(selected_programs)
    optimizer = torch.optim.Adam([batch_diff_stack.variables], LR)
    for i in range(N_STEPS):
        # for j in range(num_eps):
        outputs = batch_diff_stack.get_top()
        distances_th = torch.from_numpy(selected_distances).cuda()
        loss = torch.nn.functional.mse_loss(outputs, distances_th)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_diff_stack.bulk_compute_stack()
        if i % 20 == 0:
            print("Iter Num %d of %d. Loss: %f" % (i, N_STEPS, loss))
        
    batch_diff_stack.variables = batch_diff_stack.variables.detach()
    # batch_diff_stack.variables[:, :2]  = myround(batch_diff_stack.variables[:, :2] , base=1, clamp_min=-64, clamp_max=64)
    # batch_diff_stack.variables[:, 2]  = myround(batch_diff_stack.variables[:, 2], base=1, clamp_min=-64, clamp_max=64)
    
    final_programs = batch_diff_stack.get_float_expression()
    # Correct the tree at this point:
    # Convert to ints
    # Check once more

    outputs = batch_diff_stack.get_top().detach().cpu().numpy()
    return final_programs, outputs


def get_opt_programs_with_perturb(selected_programs, selected_distances, batch_diff_stack, N_STEPS, LR):
    
    batch_diff_stack.generate_batch_stack(selected_programs, NOISE_LIM=NOISE_LIM)
    
    optimizer = torch.optim.Adam([batch_diff_stack.variables], LR)
    for i in range(N_STEPS):
        # for j in range(num_eps):
        outputs = batch_diff_stack.get_top()
        distances_th = torch.from_numpy(selected_distances).cuda()
        loss = torch.nn.functional.mse_loss(outputs, distances_th)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_diff_stack.bulk_compute_stack()
        if i % 20 == 0:
            print("Iter Num %d of %d. Loss: %f" % (i, N_STEPS, loss))
        
    batch_diff_stack.variables = batch_diff_stack.variables.detach()
    # batch_diff_stack.variables[:, :2]  = myround(batch_diff_stack.variables[:, :2] , base=1, clamp_min=-64, clamp_max=64)
    # batch_diff_stack.variables[:, 2]  = myround(batch_diff_stack.variables[:, 2], base=1, clamp_min=-64, clamp_max=64)
    
    final_programs = batch_diff_stack.get_float_expression()
    # Correct the tree at this point:
    # Convert to ints
    # Check once more

    outputs = batch_diff_stack.get_top().detach().cpu().numpy()
    return final_programs, outputs


def get_opt_programs_with_perturb_occupancy(selected_programs, selected_distances, batch_diff_stack, N_STEPS, LR):
    
    batch_diff_stack.generate_batch_stack(selected_programs, NOISE_LIM=NOISE_LIM)
    ## Add Perturbations:
    optimizer = torch.optim.Adam([batch_diff_stack.variables], LR)
    ### Set the scalar value.
    
    start_param = np.log(0.1)
    end_param = np.log(0.01)
    alpha_param = np.arange(start_param, end_param, -(start_param - end_param)/N_STEPS)
    alpha_param = np.exp(alpha_param)
    for i in range(N_STEPS):
        # torch.nn.init.constant_(batch_diff_stack.draw_obj.scaler.m, alpha_param[i])
        batch_diff_stack.draw_obj.scaler.m = alpha_param[i]
        outputs = batch_diff_stack.get_top()
        distances_th = torch.from_numpy(selected_distances).cuda()
        loss = torch.nn.functional.mse_loss(outputs, distances_th)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_diff_stack.bulk_compute_stack()
        if i % 20 == 0:
            print("Iter Num %d of %d. Loss: %f" % (i, N_STEPS, loss))
        
    batch_diff_stack.variables = batch_diff_stack.variables.detach()
    # batch_diff_stack.variables[:, :2]  = myround(batch_diff_stack.variables[:, :2] , base=1, clamp_min=-64, clamp_max=64)
    # batch_diff_stack.variables[:, 2]  = myround(batch_diff_stack.variables[:, 2], base=1, clamp_min=-64, clamp_max=64)
    
    final_programs = batch_diff_stack.get_float_expression()
    # Correct the tree at this point:
    # Convert to ints
    # Check once more

    outputs = batch_diff_stack.get_top().detach().cpu().numpy()
    return final_programs, outputs

def run_parallel(proc_id, collected_episodes, batch_diff_stack, parser, mod_env, LR, N_STEPS, save_loc,
                 batch_size, metric_obj, optimization_function, round=0):
    
    final_programs = []
    ep_list = []
    start_time = time.time()
    ## create the diff opts for the episodes: 
    num_eps = len(collected_episodes)
    # diff_stack_list = [DifferentiableStack(max_len=11, canvas_shape=[64, 64]) for x in range(num_eps)]
    target_canvases = [collected_episodes[x]['target_canvas'] for x in range(num_eps)] 
    slot_ids = [collected_episodes[x]['slot_id'] for x in range(num_eps)] 
    target_ids = [collected_episodes[x]['target_id'] for x in range(num_eps)] 
    pred_programs = [collected_episodes[x]['pred_programs'] for x in range(num_eps)]
    initial_expression = [collected_episodes[x]['pred_expression'] for x in range(num_eps)]
    initial_rewards = [collected_episodes[x]['reward'] for x in range(num_eps)]
    distances = [collected_episodes[x]['distances'] for x in range(num_eps)]
    distances = np.stack(distances, 0)
    num_batches = np.ceil(num_eps/batch_size).astype(int)
    pred_canvases = np.zeros([0, 64, 64])
    for batch_idx in range(num_batches):
        if batch_idx % 1 == 0:
            print("Batch Id %d of %d" % (batch_idx, num_batches))
        selected_programs = pred_programs[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        selected_distances = distances[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        optimized_programs, pred_canvas = optimization_function(selected_programs, selected_distances, batch_diff_stack, N_STEPS, LR)
        pred_canvases = np.concatenate([pred_canvases, pred_canvas], 0)
        final_programs.extend(optimized_programs)
    
    reward_ratio = []
    ep_accept_list = []
    for i, final_program in enumerate(final_programs):
        final_expression = parser.prog_to_float_exp(final_program)
        ## Try refactoring: 
        target_canvas = target_canvases[i]
        pred_canvas = (pred_canvases[i:i+1] > 0).astype(np.float)
        # expr_failed, new_expression = mod_env.refactor_expression(final_expression, target_canvas)
        # if not expr_failed:
        #     final_expression = new_expression
        # Now do it again after adjustment
        final_expression = mod_env.action_space.adjust_expression(final_expression)
        
        # expr_failed, new_expression = mod_env.refactor_expression(final_expression, target_canvas)
        # if not expr_failed:
        #     final_expression = new_expression
        # now clean the expression float to int:
        
        # token_invalid = [x not ind_env.unique_draw for x in final_expression]
        # if (any(token_invalid)):
        #     # print("rejected due to action space restriction")
        #     ep_accept_list.append(0)
        # else:
        # print(final_expression)
        refactored_observations, refactored_actions, refactored_rewards = mod_env.generate_experience(final_expression, target_canvas, slot_ids[i], target_ids[i])
        
        if (refactored_rewards[-1] <= initial_rewards[i]):
            # print("rejected due to smaller reward %f %f" % (refactored_rewards[-1], initial_rewards[i]))
            ep_accept_list.append(0)
            final_expression = initial_expression[i]
            refactored_observations, refactored_actions, refactored_rewards = mod_env.generate_experience(final_expression, target_canvas, slot_ids[i], target_ids[i])
        else:
            ep_accept_list.append(1)
            reward_ratio.append(refactored_rewards[-1]/(initial_rewards[i] + 1e-9)) 
            
        episode_length = len(final_expression)
        diff_opt_reward = mod_env.reward.active_reward_func(pred_canvas, target_canvas)
        diff_opt_reward = diff_opt_reward**mod_env.reward.power
        pred_canvas = refactored_observations['obs'][-1,1:2].copy()
        
        
        episode_dict = {'observations': dict(),
                        'actions': None,
                        'rewards' : None,
                        'length': episode_length,
                        'slot_id': slot_ids[i],
                        'target_id': target_ids[i],
                        'final_expression': final_expression,
                        'pred_canvas': pred_canvas,
                        'target_canvas': target_canvas
                        }
        
        info = {}
        info['target_expression'] = final_expression.copy()
        info['predicted_expression'] = final_expression.copy()
        info['target_canvas'] = target_canvas.copy()
        info['predicted_canvas'] = pred_canvas.copy()
        info['target_id'] = target_ids[i]
        info['slot_id'] = slot_ids[i]
        info['log_prob'] = 0
        # for key, value in refactored_observations.items():
        #     episode_dict['observations'][key] = value.copy()
        metric_obj.update_metrics(info, current_length=episode_length, current_reward=refactored_rewards[-1], diff_opt_reward=diff_opt_reward)
            
        # print(refactored_actions.max())
        # Actions:
        episode_dict['actions'] = refactored_actions.copy()
        # Rewards:
        episode_dict['rewards'] = refactored_rewards
        ep_list.append(episode_dict)
        ep_accept_list.append(1)
        
    print("final median reward ratio", np.median(reward_ratio))
    print("Accepted episodes ", len(reward_ratio), "from ", len(final_programs))
    end_time = time.time()
    print("Total_time ", end_time - start_time)
    
    file_name = save_loc + "/round_%d_process_%d.pkl" % (round, proc_id)
    with open(file_name, 'wb') as f:
        cPickle.dump([ep_list, reward_ratio, ep_accept_list, metric_obj], f)
  
