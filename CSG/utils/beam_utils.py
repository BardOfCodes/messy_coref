
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import gym
import numpy as np
from operator import itemgetter
from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method
import _pickle as cPickle
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
from pathlib import Path
import torch as th
from CSG.utils.profile_utils import profileit
from CSG.env.reward_function import chamfer

try:
     set_start_method('spawn')
except RuntimeError:
    pass

def beam_search(model, env, target_id, beam_size, beam_state_size=None, selector='log_probability'):
    
    # Define beam state
    # prev_state, prev_expression log_probability
    observation = env.reset_to_target(target_id)
    state_list = [dict(observation=observation, internal_state=env.get_internal_state(), 
                       rewards=[], dones=[], infos=[], log_probability=0)]
    
    finished_seqs = []
    
    while(state_list):
        # print('len_of state', len(state_list))
        # get distribution take top three actions
        new_state_list = []
        for state in state_list:
            obs= state['observation']
            internal_state = state['internal_state']
            cur_log_prob = state['log_probability']
            # in Q network, this will get a prob distr. made from q-vals
            # That is not quite accurate.
            action_set = get_actions(model, env, obs, beam_size)
            
            actual_beam_size = action_set[1].shape[1]
            for j in range(actual_beam_size):
                new_temp_state = dict(previous_state=state, log_probability=cur_log_prob + action_set[0][0,j].item(), 
                                      action = action_set[1][0,j].item())
                new_state_list.append(new_temp_state)
            
        selected_state_list = return_criteria(new_state_list, "log_probability", with_seq=True)
        # selected_state_list = [(x,y) for x,y in zip(new_state_list, criteria)]
        selected_state_list.sort(key=lambda x: x[1], reverse=True)
        selected_state_list = selected_state_list[:beam_state_size]
        # selected_state_list = [x[0] for x in selected_state_list]
            
        state_list = []
        # For the selected - add obs and make new states:
        for cur_state_meta, score in selected_state_list:
            state = cur_state_meta['previous_state']
            action = cur_state_meta['action']
            cur_log_prob = cur_state_meta['log_probability']
            obs= state['observation']
            internal_state = state['internal_state']
            env.set_internal_state(internal_state)
            observations, rewards, dones, infos = env.step(action)
            new_internal_state = env.get_internal_state()
            
            new_reward_list = state['rewards'].copy() + [rewards]
            new_dones_list = state['dones'].copy() + [dones]
            new_infos_list = state['infos'].copy() + [infos]
            new_state = dict(observation=observations, internal_state=new_internal_state, rewards=new_reward_list,
                                dones=new_dones_list, infos=new_infos_list, log_probability=cur_log_prob)
            if dones:
                finished_seqs.append(new_state)
            else:
                state_list.append(new_state)
    criteria = return_criteria(finished_seqs, selector)
    index, element = max(enumerate(criteria), key=itemgetter(1))
    sel_seq = finished_seqs[index]
    
    return sel_seq['rewards'], sel_seq['dones'], sel_seq['infos'], sel_seq['log_probability']


def get_actions(model, env, obs, beam_size):
    obs_tensor, _ = model.obs_to_tensor(obs)
    action_distr = model.get_logits(obs_tensor)
    # action_distr = action_distr.distribution.logits
    # This will become a action space dependant instruction
    top_k_vals, top_k_inds = env.action_space.get_topk_actions(action_distr, obs_tensor, k=beam_size)
    # top_k_vals, top_k_inds = torch.topk(action_distr, k=beam_size, dim=1)
    # Can set a probability threshold here to remove certain actions.    
    return top_k_vals, top_k_inds

def get_batch_actions(model, env, obs, beam_size, active_counts=None, stochastic_beam_search=False):

    obs_tensor, _ = model.obs_to_tensor(obs)
    if active_counts:
        model.features_extractor.extractor.x_count = active_counts

    action_distr = model.get_logits(obs_tensor)
    # batch_size = obs_tensor['obs'].shape[0]
    # size_2 = env.action_space.stop_action + 1
    # action_distr = torch.rand((batch_size, size_2)).to(obs_tensor['obs'].device)
    # action_distr = env.action_space.restrict_pred_action(action_distr, obs_tensor)

    if stochastic_beam_search:
        u_noise = torch.rand(action_distr.shape).to(action_distr.device)
        gumbel_noise = -torch.log(-torch.log(u_noise))
        action_distr += gumbel_noise
        # Do Nucleus sampling instead - mark the indices beyond cumsum to 0.9 as -inf 
        # action_distr = adapt_to_nucleaus(action_distr)
        
    # action_distr = action_distr.distribution.logits
    # This will become a action space dependant instruction
    top_k_vals, top_k_inds = env.action_space.dif_get_topk_actions(action_distr, obs_tensor, k=beam_size, with_extra=True)
    # top_k_vals, top_k_inds = torch.topk(action_distr, k=beam_size, dim=1)
    # Can set a probability threshold here to remove certain actions.    
    return top_k_vals, top_k_inds

def adapt_to_nucleaus(action_distr, limit=0.95):
    probs = th.nn.functional.softmax(action_distr, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < limit 
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    sorted_log_probs = torch.log(sorted_probs)
    sorted_log_probs[~nucleus] = float('-inf')
    res = sorted_log_probs.gather(-1, indices.argsort(1, descending=False))
    return res 
     
# Depreciated
def parallel_beam_search(proc_id, model, env, targets, metric_obj, 
                         beam_size, beam_state_size, beam_selector, save_loc):
    for idx, target in enumerate(targets):
        rewards, dones, infos, log_probs = beam_search(model, env, target, beam_size=beam_size, 
                                            beam_state_size=beam_state_size, selector=beam_selector)
        
        if (idx % 100 == 0): 
            print('episode_counts', idx, 'proc_id', proc_id, infos[-1]['predicted_expression'])

        # unpack values so that the callback can access the local variables
        cur_rewards = rewards
        current_length = len(cur_rewards)
        # current_length = len(info['predicted_expression'])

        reward = cur_rewards[-1]
        info = infos[-1]
        log_prob = np.sum(log_probs)
        # result_queue.put((rewards, info))
        metric_obj.update_metrics(info, current_length=current_length, current_reward=reward, log_prob=log_prob)
    
    with open(save_loc + "/_%d.pkl" % proc_id, 'wb') as f:
        cPickle.dump(metric_obj, f)
  
# @profileit("../logs/psa_beam.stats")
def batch_parallel_beam_search(proc_id, model, env, slot_ids, target_ids, metric_obj, 
                               all_program_metric_obj, return_all,
                               beam_size, beam_state_size, beam_selector, save_loc,
                               batch_size, stochastic_bs, reward_evaluation_limit,
                               length_alpha=0):
    th.backends.cudnn.benchmark = True
    env.program_generator.set_execution_mode(th.device("cuda"), th.float16)

    
    batch_idx = np.ceil(len(target_ids) / float(batch_size)).astype(int)
    episode_counts = 0
    for idx in range(batch_idx):
        cur_slots = slot_ids[idx * batch_size: (idx + 1) * batch_size]
        cur_targets = target_ids[idx * batch_size: (idx + 1) * batch_size]
        sel_seq, all_programs = tfomer_batch_beam_search(model, env, cur_slots, cur_targets, beam_size=beam_size, 
                                            beam_state_size=beam_state_size, selector=beam_selector, return_all=return_all,
                                            stochastic_bs=stochastic_bs, reward_evaluation_limit=reward_evaluation_limit,
                                            length_alpha=length_alpha)
        
        n_out = len(sel_seq)
        for j in range(n_out):
            # unpack values so that the callback can access the local variables
            cur_rewards = sel_seq[j][0]
            info = sel_seq[j][1]
            log_prob = sel_seq[j][2]
            current_length = len(info['predicted_expression'])
            # Now that its done, 
            metric_obj.update_metrics(info, current_length=current_length, current_reward=cur_rewards[-1], log_prob=log_prob)
            if return_all:
                for item in all_programs[j]:
                    cur_rewards = item[0]
                    info = item[1]
                    log_prob = item[2]
                    current_length = len(info['predicted_expression'])
                    # Now that its done, 
                    all_program_metric_obj.update_metrics(info, current_length=current_length, 
                                                          current_reward=cur_rewards[-1], log_prob=log_prob)
            episode_counts += 1
        
            if (episode_counts % 100 == 0): 
                print('proc_id', proc_id, info['predicted_expression'])
            if (episode_counts % 100 == 0): 
                print('episode_counts', episode_counts)
    
    with open(save_loc + "/_%d.pkl" % proc_id, 'wb') as f:
        cPickle.dump([metric_obj, all_program_metric_obj], f)
    
  
    
def return_list(*args):
    return []


def return_zero(*args):
    return 0


def return_one(*args):
    return 1

def obs_list_to_array(env, obs_list):
    final_obs = defaultdict(return_list)
    for obs in obs_list:
        for key, value in obs.items():
            final_obs[key].append(value)
    
    for key, value in final_obs.items():
        final_obs[key] = np.stack(value, 0)
    
    return dict_to_obs(env.observation_space, final_obs)
     
    

def batch_beam_search(model, env, slot_ids, target_ids, beam_size, beam_state_size=None, selector='log_probability', stochastic_bs=False):
    
    finished_seqs = defaultdict(return_list)
    # Return a bulk set of observations:
    obs_list, internal_state_list = env.reset_to_target_ids(slot_ids, target_ids)
    batch_size = len(target_ids)
    # Use the target_id, and action state. 
    
    state_list = [dict(internal_state=internal_state_list[i], 
                       rewards=[], log_probability=0, target_id=target_ids[i]) for i in range(batch_size)]
    while(state_list):
        # for next round:
        
        # For selection
        temp_state_list = []

        acc_obs = obs_list_to_array(env, obs_list)
        batch_size = acc_obs['obs'].shape[0]
        
        action_set = get_batch_actions(model, env, acc_obs, beam_size, stochastic_beam_search=stochastic_bs)
        
        new_state_list_dict = defaultdict(return_list)
        for b in range(batch_size):
            cur_target_id = state_list[b]['target_id']
            cur_action_set = [action_set[0][b], action_set[1][b]]
            actual_beam_size = cur_action_set[1].shape[0]
            cur_log_prob = state_list[b]['log_probability']
            for j in range(actual_beam_size):
                new_temp_state = dict(previous_state=state_list[b], log_probability=cur_log_prob + cur_action_set[0][j], 
                                    action = cur_action_set[1][j])
                new_state_list_dict[cur_target_id].append(new_temp_state)
                # We have forward passed all instances for 
        for ind in target_ids:
            new_state_list = new_state_list_dict[ind]
            cur_selected_state_list = return_criteria(new_state_list, "log_probability", with_seq=True)
            # selected_state_list = [(x,y) for x,y in zip(new_state_list, criteria)]
            cur_selected_state_list.sort(key=lambda x: x[1], reverse=True)
            cur_selected_state_list = cur_selected_state_list[:beam_state_size]
            temp_state_list.extend(cur_selected_state_list)
            
            
        obs_list = []
        state_list= []
        for cur_state_meta, score in temp_state_list:
            state = cur_state_meta['previous_state']
            action = cur_state_meta['action']
            cur_log_prob = cur_state_meta['log_probability']
            
            # obs= state['observation']
            target_id = state['target_id']
            internal_state = state['internal_state']
            env.set_internal_state(internal_state)
            observations, rewards, dones, infos = env.step(action)
            new_internal_state = env.get_internal_state()
            
            new_reward_list = state['rewards'].copy() + [rewards]
            new_state = dict(internal_state=new_internal_state, rewards=new_reward_list,
                             log_probability=cur_log_prob, target_id=target_id)
            # target_id_list.append(target_id)
            if dones:
                new_state['info'] = infos
                finished_seqs[target_id].append(new_state)
            else:
                state_list.append(new_state)
                obs_list.append(observations)
        
    sel_seq_list = []
    for i in target_ids:
        cur_finished_seqs = finished_seqs[i]
        criteria = return_criteria(cur_finished_seqs, selector)
        index, element = max(enumerate(criteria), key=itemgetter(1))
        sel_seq_list.append([cur_finished_seqs[index]['rewards'], cur_finished_seqs[index]['info']])
    
    return sel_seq_list
    
    

def tfomer_batch_beam_search(model, env, slot_ids, target_ids, beam_size, beam_state_size=None, selector='log_probability', 
                             return_all=False, stochastic_bs=False, reward_evaluation_limit=100, selection_mode="absolute_best",
                             length_alpha=0):
    
    
    finished_seqs = defaultdict(return_list)
    # Return a bulk set of observations:
    model.features_extractor.extractor.beam_partial_init=True
    model.beam_partial_init = True
    env.reset_on_done = True
    obs_list, internal_state_list = env.reset_to_target_ids(slot_ids, target_ids)
    batch_size = len(target_ids)
    # Use the target_id, and action state. 
    
    state_list = [dict(internal_state=internal_state_list[i], log_probability=0, rewards=[], slot_id=slot_ids[i], target_id=target_ids[i]) for i in range(batch_size)]
    # For the counts:
    init_batch_size = batch_size
    active_target_counts = [1 for x in range(init_batch_size)]
    target_2_ind_dict = {str(slot_ids[i]) + "$" + str(target_ids[i]): i for i in range(init_batch_size)}
    while(state_list):
        # for next round calculate x_count:
        # For selection
        temp_state_list = []

        acc_obs = obs_list_to_array(env, obs_list)
        batch_size = acc_obs['obs'].shape[0]
        
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
                action_set = get_batch_actions(model, env, acc_obs, beam_size, active_target_counts, stochastic_beam_search=stochastic_bs)
        
        target_count_dict = defaultdict(return_zero)
        new_state_list_dict = defaultdict(return_list)
        for b in range(batch_size):
            cur_target_id = state_list[b]['target_id']
            cur_slot_id = state_list[b]['slot_id']
            cur_action_set = [action_set[0][b], action_set[1][b]]
            actual_beam_size = cur_action_set[1].shape[0]
            cur_log_prob = state_list[b]['log_probability']
            for j in range(actual_beam_size):
                ## If the sequence is finished directly add it to temp state_list
                # select top k action from the rest?
                new_temp_state = dict(previous_state=state_list[b], log_probability=cur_log_prob + cur_action_set[0][j], 
                                    action = cur_action_set[1][j])
                key = (cur_slot_id, cur_target_id)
                new_state_list_dict[key].append(new_temp_state)
                # We have forward passed all instances for 
        for idx, cur_target_id in enumerate(target_ids):
            cur_slot_id = slot_ids[idx]
            key = (cur_slot_id, cur_target_id)
            new_state_list = new_state_list_dict[key]
            cur_selected_state_list = return_criteria(new_state_list, "log_probability", with_seq=True)
            # selected_state_list = [(x,y) for x,y in zip(new_state_list, criteria)]
            cur_selected_state_list.sort(key=lambda x: x[1], reverse=True)
            # Here also if there are stops then add
            stop_contained = [x[0]['action']==env.action_space.stop_action for x in cur_selected_state_list[:beam_state_size]]
            cur_selected_state_list = cur_selected_state_list[:beam_state_size + np.sum(stop_contained, dtype=np.int32)]
            temp_state_list.extend(cur_selected_state_list)
        
            
            
        obs_list = []
        state_list= []
        for cur_state_meta, score in temp_state_list:
            state = cur_state_meta['previous_state']
            action = cur_state_meta['action']
            cur_log_prob = cur_state_meta['log_probability']
            
            # obs= state['observation']
            target_id = state['target_id']
            slot_id = state['slot_id']
            internal_state = state['internal_state']
            env.set_minimal_internal_state(internal_state)
            observations, rewards, dones, infos = env.step(action)
            new_internal_state = env.get_minimal_internal_state()
            cur_key = str(slot_id) + "$" + str(target_id)
            
            new_reward_list = [rewards] # state['rewards'].copy() + [rewards]
            new_state = dict(internal_state=new_internal_state, log_probability=cur_log_prob, rewards=new_reward_list, slot_id=slot_id, target_id=target_id)
            # target_id_list.append(target_id)
            if dones:
                if rewards > 0:
                    fin_state = {'info': infos, 'rewards': new_reward_list, "log_probability":cur_log_prob}
                    finished_seqs[cur_key].append(fin_state)
            else:
                state_list.append(new_state)
                obs_list.append(observations)
                target_count_dict[cur_key] += 1
        
        ## create the active_target_counts
        active_target_counts = [0, ] * init_batch_size
        for key, value in target_count_dict.items():
            active_target_counts[target_2_ind_dict[key]] = value
            
    sel_seq_list = []
    all_programs = []
    for key in target_2_ind_dict.keys():
        cur_finished_seqs = finished_seqs[key]
        # Make unique
        if len(cur_finished_seqs) == 0:

            info = env.empty_info_dict.copy()
            slot_id, target_id = key.split("$")
            info['slot_id'] = slot_id
            info['target_id'] = int(target_id)
            info['reward'] = [-0.1]
            sel_seq_list.append(([-0.1], info,  -np.inf))
            if return_all:
                all_programs.append(([-0.1], info,  -np.inf))
            # Only evaluate for the top 100 with highest log likelihood.
        else:
            cur_finished_seqs.sort(key=lambda x: -x['log_probability'])
            cur_finished_seqs = cur_finished_seqs[:reward_evaluation_limit]
            slot_id, target_id = cur_finished_seqs[0]['info']['slot_id'], cur_finished_seqs[0]['info']['target_id']
            target_tensor, expression = env.program_generator.get_executed_program(slot_id, target_id, return_numpy=False, return_bool=True)
            for seq in cur_finished_seqs:
                pred_expression = seq['info']['predicted_expression']
                if "3D" in env.language_name:
                    pred_canvas = env.program_generator.execute(pred_expression, return_numpy=False, return_bool=True)
                    R = th.logical_and(pred_canvas, target_tensor).sum()/th.logical_or(pred_canvas, target_tensor).sum()
                    R = R.item()
                    seq['info']['predicted_canvas'] = pred_canvas.data.cpu().numpy()
                    seq['info']['target_canvas'] = target_tensor.data.cpu().numpy()
                else:
                    # pred_canvas = env.program_generator.execute(pred_expression, return_numpy=False, return_bool=True)
                    # R = th.logical_and(pred_canvas, target_tensor).sum()/th.logical_or(pred_canvas, target_tensor).sum()
                    # reward = R.item()
                    target_np = target_tensor.cpu().numpy()
                    pred_canvas = env.program_generator.execute(pred_expression, return_numpy=True, return_bool=True)
                    R = 100 - chamfer(target_np[None, :, :], pred_canvas[None, :, :])[0]
                reward = R + length_alpha * len(pred_expression)
                seq['info']['reward'] = [reward]
                seq['rewards'] = [reward]

            criteria = return_criteria(cur_finished_seqs, selector, length_alpha=length_alpha)
            index, element = max(enumerate(criteria), key=itemgetter(1))
            if selection_mode == "absolute_best":
                sel_item = cur_finished_seqs[index]
            else:
                threshold = element - 0.001 # HACKY
                subset_seq_list = [x for x in cur_finished_seqs if x['rewards'][-1] >threshold]
                expr_lens = [len(x['info']['predicted_expression']) for x in cur_finished_seqs]
                index, len_element =  min(enumerate(expr_lens), key=itemgetter(1))
                sel_item = cur_finished_seqs[index]
            sel_seq_list.append((sel_item['rewards'], sel_item['info'],  sel_item["log_probability"]))
            if return_all:
                # only keep unique ones:
                all_expr = []
                cur_programs = []
                for fin_seq in cur_finished_seqs:
                    cur_expr = fin_seq['info']['predicted_expression']
                    if cur_expr not in all_expr:
                        cur_programs.append((fin_seq['rewards'], fin_seq['info'],  fin_seq["log_probability"]))
                        all_expr.append(cur_expr)
                cur_programs.sort(key=lambda x: x[0])
                all_programs.append(cur_programs)
    model.features_extractor.extractor.beam_partial_init = False
    model.beam_partial_init = False
    return sel_seq_list, all_programs
    
    
def return_criteria(seqs, selector, with_seq=False, length_alpha=0):
    
    # Beam search over!
    if selector == 'rewards':
        # Select program with best reward
        # TODO: DISCOUNTED REWARDS!!
        lengths = [len(x['info']['predicted_expression']) for x in seqs]

        if with_seq:
            criteria = [(x, sum(x['rewards'])) for ind, x in enumerate(seqs)]
        else:
            criteria = [sum(x['rewards']) for ind, x in enumerate(seqs)]
            # criteria = [sum(x['rewards']) for ind, x in enumerate(seqs)]
        # z = [sum(x['rewards']) for x in seqs]
        # print(np.max(z), len(z), np.mean(z))
    elif selector == 'log_probability':
        # select program with best probability
        if with_seq:
            criteria = [(x, x['log_probability']) for x in seqs]
        else:
            criteria = [x['log_probability'] for x in seqs] 
        
    return criteria       

