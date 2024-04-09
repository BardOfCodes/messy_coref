
from curses.ascii import BS
import random
import os
import _pickle as cPickle
from pathlib import Path
from collections import defaultdict
import torch as th
import numpy as np
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info

ALL_ORIGINS = ["BS", "DO", "GS", "CS", "WS", ]
MERGABLE_ORIGIN_TYPES = ["BS", "DO", "GS", "CS" ]
MERGABLE_SANS_BS_TYPES = ["DO", "GS", "CS" ]
REWRITE_ORIGINS = ["DO", "GS", "CS"]
LOG_MULTIPLIER = 1.5

def format_data(batch):
    cur_dict = batch[0]
    all_dict = {}
    for key, item in cur_dict.items():
        # value = th.cat([x[key] for x in batch], 0)
        value = th.stack([x[key] for x in batch], 0)
        all_dict[key] = value# th.from_numpy(value).detach()# .to("cuda", non_blocking=True)
        if key == 'cur_step':
            all_steps = value
    
    target = all_dict.pop("target")

    target_list = []
    for ind, count in enumerate(all_steps):
        target_list.append(target[ind, :count])
    target = th.cat(target_list)

    return all_dict, target

def format_rl_data(batch):
    cur_dict = batch[0]
    all_dict = {}
    for key, item in cur_dict.items():
        # value = th.cat([x[key] for x in batch], 0)
        value = th.stack([x[key] for x in batch], 0)
        all_dict[key] = value# th.from_numpy(value).detach()# .to("cuda", non_blocking=True)
        if key == 'cur_step':
            all_steps = value
    
    target = all_dict.pop("target")
    reward = all_dict.pop("reward")

    target_list = []
    reward_list = []
    for ind, count in enumerate(all_steps):
        reward_list.append(reward[ind].expand(count))
        target_list.append(target[ind, :count])
    target = th.cat(target_list)
    reward = th.cat(reward_list)
    all_dict['reward'] = reward
    return all_dict, target

def format_data_single_env(batch):
    cur_dict = batch[0]
    all_dict = {}
    for key, item in cur_dict.items():
        value = np.stack([x[key] for x in batch], 0)
        all_dict[key] = th.from_numpy(value).detach()
        if key == 'cur_step':
            all_steps = value
    
    target = all_dict.pop("target")

    target_list = []
    for ind, count in enumerate(all_steps):
        target_list.append(target[ind, :count])
    target = th.cat(target_list)

    return all_dict, target

def probabilistic_program_list(best_program_dict, probability_type=1, n_rewrites_per_item=1,  temperature=1, gen_sample_rate=0.25, 
                               n_samples_per_item=10, rewrite_origins=REWRITE_ORIGINS, ensure_BS=False, multiply_input=False,
                               n_bins=5):
    program_list = []
    rewrite_dict = defaultdict(list)
    bs_dict = defaultdict(list)
    generative_dict = defaultdict(list)
    
    # divi the different origins:
    for key, value in best_program_dict.items():
        if value:
            origin = key[2]
            if origin in rewrite_origins:
                new_key = (key[0], key[1])
                rewrite_dict[new_key].extend(value)
            elif origin == "BS":
                new_key = (key[0], key[1])
                assert len(value) == 1, "for single bs only"
                bs_dict[new_key].extend(value[:1])
            elif origin in ["WS", "NR"]:
                new_key = (key[0], key[1])
                # assert len(value) == 1, "for single WS only"
                generative_dict[new_key].extend(value)
            
    # Truncate Rewrite dict: 
    for key, value in rewrite_dict.items():
        value.sort(key=lambda x: x['reward'], reverse=True)
        min_reward = bs_dict[key][0]['reward']
        sel_values = [x for x in value if x['reward'] > min_reward]
        cur_n = min(len(sel_values), n_rewrites_per_item)
        selected_list = sel_values[:cur_n]
        rewrite_dict[key] = selected_list
    
    # construct merged with probability:
    for key, value in bs_dict.items():
        # WS
        gen_key = ("WS", key[1])
        gen_items = [x for x in generative_dict[gen_key]]
        bs_items = [x for x in bs_dict[key]]
        rewrite_items = [x for x in rewrite_dict[key]]
        
        for item in gen_items:
            item['data_probability'] = 1 / len(gen_items) * gen_sample_rate
            
        if probability_type == 1:
            bs_prob = max(0.25, 1/(len(rewrite_items)+ 1))
            bs_reward = bs_items[0]['reward']
            for item in bs_items:
                item['data_probability'] = bs_prob * (1 - gen_sample_rate)
            rewrite_reward_deltas = np.array([x['reward'] - bs_reward for x in rewrite_items])
            rewrite_reward_deltas_temp = rewrite_reward_deltas / temperature
            rewrite_probs = np.exp(rewrite_reward_deltas_temp)/np.exp(rewrite_reward_deltas_temp).sum()
            for ind, item in enumerate(rewrite_items):
                item['data_probability'] = (1 - bs_prob) * rewrite_probs[ind] * (1 - gen_sample_rate)
        elif probability_type == 2:
            rewrite_reward_deltas = [x['reward'] for x in rewrite_items]
            rewrite_reward_deltas.append(bs_items[0]['reward'])
            rewrite_reward_deltas = np.array(rewrite_reward_deltas)
            rewrite_reward_deltas_temp = rewrite_reward_deltas / temperature
            rewrite_probs = np.exp(rewrite_reward_deltas_temp)/np.exp(rewrite_reward_deltas_temp).sum()
            for ind, item in enumerate(rewrite_items):
                item['data_probability'] = rewrite_probs[ind] * (1 - gen_sample_rate)
            for ind, item in enumerate(bs_items):
                item['data_probability'] = rewrite_probs[-1] * (1 - gen_sample_rate)
        elif probability_type == 3:
            # split the reward range.
            # Assign programs to each partition
            # assign program probability based on the number of programs in each partition * partition probability.
            if rewrite_items:
                rewrite_reward_deltas = [x['reward'] for x in rewrite_items]
                rewrite_reward_deltas.append(bs_items[0]['reward'])
                rewrite_reward_deltas = np.array(rewrite_reward_deltas)
                # divide the reward_deltas by number of programs in bin
                min_r, max_r = np.min(rewrite_reward_deltas), np.max(rewrite_reward_deltas)
                bin_size = (max_r - min_r)/n_bins
                reward_bin = [(x-min_r)/bin_size for x in rewrite_reward_deltas]
                uniques, counts = np.unique(reward_bin, return_counts=True)
                unique_counts = {x:y for x,y in zip(uniques, counts)}
                reward_div = np.array([unique_counts[x] for x in reward_bin])
                rewrite_reward_deltas = rewrite_reward_deltas / reward_div
            else:
                rewrite_reward_deltas = np.array([bs_items[0]['reward']])
            
            rewrite_reward_deltas_temp = rewrite_reward_deltas / temperature
            rewrite_probs = np.exp(rewrite_reward_deltas_temp)/np.exp(rewrite_reward_deltas_temp).sum()
            for ind, item in enumerate(rewrite_items):
                item['data_probability'] = rewrite_probs[ind] * (1 - gen_sample_rate)
            for ind, item in enumerate(bs_items):
                item['data_probability'] = rewrite_probs[-1] * (1 - gen_sample_rate)
                
        all_progs = gen_items + bs_items + rewrite_items
        # Now sample n samples from all progs
        prob_weights = [x['data_probability'] for x in all_progs]
        # print("prob sum", np.sum(prob_weights))        
        if multiply_input:
            cur_n_samples = int(n_samples_per_item * len(all_progs))
        else:
            cur_n_samples = n_samples_per_item

        sampled_progs = random.choices(all_progs, weights=prob_weights, k=cur_n_samples)
        if ensure_BS:
            # there is atmost 5% chance it does not get included in Prob 1 and more in Prob 2.
            # Ensure that we have the BS program strictly.
            bs_program = bs_items[0]
            keys = [x['origin'] for x in sampled_progs]
            if not "BS" in keys:
                sampled_progs.append(bs_program)
        
        program_list.extend(sampled_progs)
    return program_list

    

def get_program_list(best_program_dict, valid_origins=ALL_ORIGINS, merge_origins=MERGABLE_ORIGIN_TYPES, n_prog_per_item=1, select_random=False, BS_gated=False):
    program_list = []
    merged_dict = defaultdict(list)
    if BS_gated:
        bs_dict = defaultdict(list)
    for key, value in best_program_dict.items():
        if value:
            origin = key[2]
            if origin in valid_origins:
                if origin in merge_origins:
                    new_key = (key[0], key[1])
                    merged_dict[new_key].extend(value)
                else:
                    program_list.extend(value)
            if BS_gated and origin == "BS":
                    new_key = (key[0], key[1])
                    bs_dict[new_key].extend(value[:1])
    
    for key, value in merged_dict.items():
        value.sort(key=lambda x: x['reward'], reverse=True)
        if BS_gated:
            if key in bs_dict.keys():
                min_reward = bs_dict[key][0]['reward']
            else:
                min_reward = -10000
            sel_values = [x for x in value if x['reward'] >= min_reward]
        else:
            sel_values = value
        cur_n = min(len(sel_values), n_prog_per_item)
        if select_random:
            selected_dict = random.sample(sel_values, cur_n)
        else:
            selected_dict = sel_values[:cur_n]
        for value in selected_dict:
                program_list.append(value)
        
    return program_list

def get_rand_sample_list(program_dict, sample_count, n_proc, sorted=False, exhaustive=False, valid_origin=MERGABLE_ORIGIN_TYPES, n_prog_per_item=1, select_random=False,
                         remove_do_fail=False, remove_cs_fail=False, temperature=0.2):
    
    merged_dict = defaultdict(list)
    for key, value in program_dict.items():
        new_key = (key[0], key[1])
        origin_type = key[2]
        if origin_type in valid_origin:
            merged_dict[new_key].extend(value)
    # Keep its sorted.
    for key, value in merged_dict.items():
        value.sort(key=lambda x: x['reward'], reverse=True)
        if remove_do_fail:
            value = [x for x in value if not x['do_fail']]
        if remove_cs_fail:
            value = [x for x in value if not x['cs_fail']]
        if value:
            merged_dict[key] = value


    keys = list(merged_dict.keys())
    n_keys = len(keys)
    if exhaustive:
        sample_count = n_keys
    
    if sorted:
        sorting_list = [(ind, merged_dict[key][0]) for ind, key in enumerate(keys)]
        sorting_list.sort(key= lambda x: x[1]['reward'])
        sorting_list = [x[0] for x in sorting_list]
        rand_indexes = sorting_list[:sample_count]
    else:
        n_samples = min(n_keys, sample_count)
        rand_indexes = random.sample(range(n_keys), n_samples)
        # HACK
        # sorting_list = [merged_dict[key][0] for ind, key in enumerate(keys)]
        # rewrite_reward_deltas = np.array([-x['reward'] for x in sorting_list])
        # rewrite_reward_deltas_temp = rewrite_reward_deltas / temperature
        # rewrite_probs = np.exp(rewrite_reward_deltas_temp)/np.exp(rewrite_reward_deltas_temp).sum()
        # rewrite_probs = rewrite_probs + 1e-9
        # rand_indexes = np.random.choice(range(n_keys), p=rewrite_probs, size=n_samples,  replace=False)


    target_slots = [[] for j in range(n_proc)]
    target_ids = [[] for j in range(n_proc)]
    mini_prog_dicts = [[] for x in range(n_proc)]

    for ind in rand_indexes:
        key = keys[ind]
        value = merged_dict[key] # Value should always be > 1
        if value:
            cur_n = min(len(value), n_prog_per_item)
            if select_random:
                # how to sample? Sample by reward:
                rewrite_reward_deltas = np.array([x['reward'] for x in value])
                rewrite_reward_deltas_temp = rewrite_reward_deltas / temperature
                rewrite_probs = np.exp(rewrite_reward_deltas_temp)/np.exp(rewrite_reward_deltas_temp).sum()
                selected_dict = random.choices(value, weights=rewrite_probs, k=cur_n)
                # selected_dict = random.sample(value, cur_n)
            else:
                if exhaustive:
                    selected_dict = value[:cur_n]
                    # bs_item = [x for x in value if x['origin'] == "BS"]
                    # if not bs_item in selected_dict:
                    #     selected_dict.extend(bs_item)
                else: 
                    selected_dict = value[:cur_n]
        else:
            selected_dict = []

        index = ind % n_proc
        cur_slot_id = key[0]
        cur_target_id = key[1]
        for item in selected_dict:
            origin = item['origin']
            mini_prog_dicts[index].append(((cur_slot_id, cur_target_id, origin), item))
            target_slots[index].append(cur_slot_id)
            target_ids[index].append(cur_target_id)
    
    return target_slots, target_ids, mini_prog_dicts

def ntcsg_to_pcsg(exprs):
    for ind, expr in enumerate(exprs):
        if "," in expr:
            command_symbol = expr.split("(")[0]
            param_str = expr.split("(")[1][:-1]
            param = [float(x.strip()) for x in param_str.split(",")]
            new_params = [0, ] * 6
            new_params[0], new_params[1], new_params[2] = param[1], param[0], param[2]
            new_params[3], new_params[4], new_params[5] = param[4] * 2, param[3] * 2, param[5] * 2

            # new_params[0], new_params[1], new_params[2] = param[0], param[1], param[2]
            # new_params[3], new_params[4], new_params[5] = param[3] * 2, param[4] * 2, param[5] * 2
            param_str = ", ".join(["%f"%x for x in new_params])
            if command_symbol == "ellipsoid":
                command_symbol = "sphere"
            draw_expr = "%s(%s)" %(command_symbol, param_str)
            exprs[ind] = draw_expr
    return exprs

def dataloader_format(cur_prog_dict, temp_env, latent_execution, le_add_noise, fetch_reward=False):

        actions = cur_prog_dict['actions']
        actions = th.from_numpy(actions).to("cuda", non_blocking=True)
        ep_length = cur_prog_dict['seq_length']
        ep_length = th.tensor(ep_length).to("cuda", non_blocking=True)
        # reward = cur_prog_dict['reward']
        if latent_execution:
            if "csg_expression" in cur_prog_dict.keys():
                csg_expr = cur_prog_dict["csg_expression"]
                temp_env.program_generator.compiler._csg_compile(csg_expr)
                target_canvas = temp_env.program_generator.compiler.get_output_shape(False)
            else:
                expr = cur_prog_dict['expression']
                target_canvas = temp_env.program_generator.execute(expr, return_numpy=False, 
                                                                   add_noise=le_add_noise)
        else:
            slot_id, target_id = cur_prog_dict['slot_id'], cur_prog_dict['target_id']
            try:
                target_canvas, expression = temp_env.program_generator.get_executed_program(slot_id, target_id, return_numpy=False)
            except Exception as ex:
                print(ex)
                print(slot_id, target_id)
                raise ex

        output = {
            "obs": target_canvas,
            "previous_steps" : actions,
            "target" : actions,
            "cur_step" : ep_length,
        }
        if fetch_reward:
            reward = th.tensor(cur_prog_dict['reward']).to("cuda", non_blocking=True)
            output['reward'] = reward
        # output = dict_to_obs(temp_env.observation_space, output)
        return output

class LabelSmoothing(th.nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx=None, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.criterion = th.nn.KLDivLoss(reduction="batchmean")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = th.nonzero(target.data == self.padding_idx)
        # if mask.dim() > 0:
        #     true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())