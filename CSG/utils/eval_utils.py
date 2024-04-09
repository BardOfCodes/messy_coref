
from collections import defaultdict
import profile
import random
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import gym
import numpy as np
from operator import itemgetter
from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
import torch
import torch as th
## TODO: Remove HACK
from .metrics import MetricObj, BASE_METRICS, EXTRACTOR
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method
import _pickle as cPickle
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
from .beam_utils import beam_search, batch_beam_search, batch_parallel_beam_search, tfomer_batch_beam_search
from pathlib import Path
from CSG.utils.profile_utils import profileit
import os
import traceback
try:
     set_start_method('spawn')
except RuntimeError:
    pass


# For beam = 1
def CSG_evaluate(
    policy: "base_class.BaseAlgorithm.Policy",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    gt_program=True,
    save_predictions=False,
    logger=None,
    n_call=0,
    exhaustive=False,
    extractor_class="",
    **kwargs,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
   
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])
    
    for cur_env in env.envs:
            env.program_generator.set_execution_mode(th.device("cuda"), th.float16)

    n_envs = env.num_envs
    episode_counts = np.zeros(n_envs, dtype="int")

    slot_target_tuple_list = []
    for slot in env.envs[0].program_generator.program_lengths:
        targets = env.envs[0].program_generator.active_gen_objs[slot][0]
        tuples = [(slot, x) for x in targets]
        slot_target_tuple_list.extend(tuples)
    if exhaustive:
        n_eval_episodes = len(slot_target_tuple_list)
        # make a list from program generator
    else:
        n_samples = min(len(slot_target_tuple_list), n_eval_episodes)
        slot_target_tuple_list = random.sample(slot_target_tuple_list, n_eval_episodes)         

    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
    

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    metric_extrator = EXTRACTOR[extractor_class]
    metric_obj = MetricObj(BASE_METRICS, metric_extrator, env, gt_program, save_predictions=save_predictions)
    observations = env.reset()
    slt_counter = 0
    for ind in range(n_envs):
        env.envs[ind].program_generator.reset_data_distr()
        slot, target_counter = slot_target_tuple_list[slt_counter]
        obs_dict = env.envs[ind].reset_to_eval_target(slot, target_counter)
        for key, value in observations.items():
            value[ind] = obs_dict[key]
        slt_counter += 1
    states = None
    logit_sums = np.zeros(n_envs)
    while (episode_counts < episode_count_targets).any():
        # actions, states = model.predict(observations, state=states, deterministic=deterministic)
        with torch.no_grad():
            actions, logits, states = policy.predict_with_logits(observations, state=states, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                logit_sums[i] += logits[i]
                if callback is not None:
                    callback(locals(), globals())

                if done:
                    info['log_prob'] = logit_sums[i].copy()
                    metric_obj.update_metrics(info, current_length=current_lengths[i], current_reward=current_rewards[i], log_prob=info['log_prob'])
                    
                    episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    logit_sums[i] = 0
                    cur_episode = np.sum(episode_counts)
                    if (cur_episode % 100 == 0): 
                        print('episode_counts', cur_episode,'Prediction', info['predicted_expression'])
                        print('episode_counts', cur_episode,'GT', info['target_expression'])
                    if episode_counts[i] < episode_count_targets[i]:
                        slot, target_counter = slot_target_tuple_list[slt_counter]
                        slt_counter += 1
                        obs_dict = env.envs[i].reset_to_eval_target(slot, target_counter)
                        for key, value in observations.items():
                            value[i] = obs_dict[key]
                        
    return metric_obj, None
   
def parallel_CSG_beam_evaluate(
    policy: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    gt_program=True,
    beam_k=1,
    beam_state_size=None,
    beam_selector="log_probability",
    save_predictions=False,
    logger=None,
    n_call=0,
    exhaustive=False,
    save_loc='',
    beam_n_proc=5,
    beam_n_batch=100,
    return_all=False,
    stochastic_bs=False,
    extractor_class="",
    reward_evaluation_limit=1000,
    length_alpha=0,
    *args, **kwargs,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
   
    Path(save_loc).mkdir(parents=True, exist_ok=True)
    
    if isinstance(env, VecEnv):
        env = env.envs[0]
    

    env.program_generator.set_execution_mode(th.device("cpu"), th.float16)
    env.program_generator.reload_data()
                
    target_ids, target_slots = get_target_ids_and_slots(env, exhaustive, beam_n_proc, n_eval_episodes)
    

    policy.enable_beam_mode()
    policy.share_memory()
    keep_trying = True
    n_tries = 0
    beam_to_try = list(range(beam_n_proc))
    while(keep_trying):
        metric_extractor = EXTRACTOR[extractor_class]
        metric_obj = MetricObj(BASE_METRICS, metric_extractor, env, gt_program, save_predictions=save_predictions)
        if return_all:
            all_program_metric_obj = MetricObj(BASE_METRICS, metric_extractor, env, gt_program, save_predictions=save_predictions)
        else:
            all_program_metric_obj = None
            
        th.cuda.empty_cache()
        print("Beam size = %d" % beam_n_batch)
        processes = []
        # beam_to_try = [0]
        for proc_id in beam_to_try:
            print("Starting process %d" % proc_id)
            # Update Env to have only the relevant information
            
            env.program_generator.load_specific_data(target_slots[proc_id], target_ids[proc_id])
            p = mp.Process(target=batch_parallel_beam_search, args=(proc_id, policy, env, target_slots[proc_id],
                                                            target_ids[proc_id], metric_obj,
                                                            all_program_metric_obj, return_all,
                                                            beam_k, beam_state_size, beam_selector, save_loc, beam_n_batch, stochastic_bs, reward_evaluation_limit, length_alpha))
            p.start()
            processes.append(p) 
            
            # batch_parallel_beam_search(proc_id, policy, env, target_slots[proc_id],
            #                            target_ids[proc_id], metric_obj, all_program_metric_obj, 
            #                            return_all, beam_k, beam_state_size, beam_selector, save_loc, 
            #                            beam_n_batch, stochastic_bs, reward_evaluation_limit, length_alpha)
            # Update Env to have all data.
        
        for p in processes:
            p.join()
            
        new_beam_to_try = []
        for i in beam_to_try:
            file_name = save_loc + "/_%d.pkl" % i
            try:
                with open(file_name, 'rb') as f:
                    temp_metric_obj, temp_all_program_metric_obj = cPickle.load(f)
                    metric_obj.fuse(temp_metric_obj)
                    if return_all:
                        all_program_metric_obj.fuse(temp_all_program_metric_obj)
                    os.system("rm %s" % (file_name))
            except Exception as ex:
                # print(ex)
                # print(traceback.format_exc())
                print("Failed Process %d" % i)
                # This means the Beam search failed. Try it with smaller batch size?
                new_beam_to_try.append(i)
        if new_beam_to_try:
            keep_trying = True
            n_tries += 1
            if n_tries == 3: 
                # Try reducing the number of processes
                print("Reduced Processes Thrice. Still failed. :(")
                raise ValueError("Could not perform with %d processes and %d batch size" % (beam_n_proc, beam_n_batch))
            else:
                beam_n_batch  = int(beam_n_batch / 2)
                beam_to_try = new_beam_to_try
        else:
            print("FINISHED Beam Search!!")
            keep_trying = False
            print("working setting: %d processes and %d batch size" % (beam_n_proc, beam_n_batch))
            new_conf = dict(beam_n_proc=beam_n_proc, beam_n_batch=beam_n_batch)

        
    env.program_generator.reload_data()
    env.program_generator.set_execution_mode(th.device("cuda"), th.float16)
    policy.disable_beam_mode()

    return metric_obj, all_program_metric_obj, new_conf

def get_target_ids_and_slots(env, exhaustive, beam_n_proc, n_eval_episodes):
    target_ids = [[] for j in range(beam_n_proc)]
    target_slots = [[] for j in range(beam_n_proc)]
    
    slot_target_tuple_list = []
    for slot in env.program_generator.program_lengths:
        targets = env.program_generator.active_gen_objs[slot][0]
        tuples = [(slot, x) for x in targets]
        slot_target_tuple_list.extend(tuples)
        
    if exhaustive:
        n_eval_episodes = len(slot_target_tuple_list)
        # make a list from program generator
    else:
        sample_count = min(len(slot_target_tuple_list), n_eval_episodes)
        slot_target_tuple_list = random.sample(slot_target_tuple_list, sample_count)  

    for ind, slot_target_tuple in enumerate(slot_target_tuple_list):
        target_slots[ind%beam_n_proc].append(slot_target_tuple[0])
        target_ids[ind%beam_n_proc].append(slot_target_tuple[1])
    return target_ids,target_slots

def batch_CSG_beam_evaluate(
    policy: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    gt_program=True,
    beam_k=1,
    beam_state_size=None,
    beam_selector="log_probability",
    save_predictions=False,
    logger=None,
    n_call=0,
    exhaustive=False,
    beam_n_batch = 10,
    return_all=False,
    stochastic_bs=False,
    extractor_class="",
    *args, **kwargs
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
   
    # if not isinstance(env, VecEnv):
    #     env = DummyVecEnv([lambda: env])

    if isinstance(env, VecEnv):
        env = env.envs[0]
    # model settings
    n_envs = 1
    metric_extrator = EXTRACTOR[extractor_class]
    metric_obj = MetricObj(BASE_METRICS, metric_extrator, env, gt_program, save_predictions=save_predictions)
    if return_all:
        metric_extrator = EXTRACTOR[extractor_class]
        all_program_metric_obj = MetricObj(BASE_METRICS, metric_extrator, env, gt_program, save_predictions=save_predictions)
    else:
        all_program_metric_obj = None
    
    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    env.program_generator.reset_data_distr()
    target_ids = []
    slot_ids = []
    if exhaustive:
        if exhaustive:
            for slot in env.program_generator.program_lengths:
                targets = env.program_generator.active_gen_objs[slot][0]
                target_ids.extend([x for x in targets])
                slot_ids.extend([slot for x in targets])
            n_eval_episodes = len(target_ids)
    else:
        slot_target_tuple_list = []
        for slot in env.program_generator.program_lengths:
            targets = env.program_generator.active_gen_objs[slot][0]
            tuples = [(slot, x) for x in targets]
            slot_target_tuple_list.extend(tuples)
        n_samples = min(len(slot_target_tuple_list), n_eval_episodes)
        slot_target_tuple_list = random.sample(slot_target_tuple_list, n_samples)  

        for ind, slot_target_tuple in enumerate(slot_target_tuple_list):
            slot_ids.append(slot_target_tuple[0])
            target_ids.append(slot_target_tuple[1])
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
    
    iter = 0
    
    while (episode_counts < episode_count_targets).any():
        # Parallel forward to numb of envs:
        cur_target_ids = target_ids[iter * beam_n_batch:(iter + 1) * beam_n_batch]
        cur_slot_ids = slot_ids[iter * beam_n_batch:(iter + 1) * beam_n_batch]
        sel_seq, all_programs = tfomer_batch_beam_search(policy, env, cur_slot_ids, cur_target_ids, beam_size=beam_k,
                                                         beam_state_size=beam_state_size, selector=beam_selector,
                                                         return_all=return_all, stochastic_bs=stochastic_bs)
        n_out = len(sel_seq)
        for j in range(n_out):
            # unpack values so that the callback can access the local variables
            cur_rewards = sel_seq[j][0]
            info = sel_seq[j][1]
            log_prob = sel_seq[j][2]
            current_length = len(cur_rewards)
            # Now that its done, 
            metric_obj.update_metrics(info, current_length=current_length, current_reward=cur_rewards[-1], log_prob=log_prob)
            episode_counts[0] += 1
            if (episode_counts[0] % 100 == 0): 
                print('episode_counts', episode_counts[0],  info['predicted_expression'])
        
            if return_all:
                for item in all_programs[j]:
                    cur_rewards = item[0]
                    info = item[1]
                    log_prob = item[2]
                    current_length = len(cur_rewards)
                    # Now that its done, 
                    all_program_metric_obj.update_metrics(info, current_length=current_length, current_reward=cur_rewards[-1], log_prob=log_prob)
        # Set next set of targets: 
        iter += 1

    return metric_obj, all_program_metric_obj
    