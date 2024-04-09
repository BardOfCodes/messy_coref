import torch as th
import random
import numpy as np
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info

from .utils import dataloader_format
from itertools import cycle, islice
import math
import time
    
class BCEnvDataset(th.utils.data.IterableDataset):
    def __init__(self, bc_env, n_iters, *args, **kwargs):
        super(BCEnvDataset, self).__init__(*args, **kwargs)
        self.n_iters = n_iters
        self.env = bc_env
        for env in self.env.envs:
            env.program_generator.set_execution_mode(th.device("cuda"), th.float32)
    
        self.n_envs = len(self.env.envs) 
    def __len__(self):
        # let it be the len of pr
        return self.n_iters
    
    def __getitem__(self, idx):
        obs_list = []
        with th.no_grad():
            for env in self.env.envs:
                cur_obs = env.minimal_reset()
                obs_list.append(cur_obs)
        final_dict = {}
        for key in cur_obs.keys():
            final_dict[key] = th.stack([x[key] for x in obs_list], 0)
        return final_dict

    def iterable_func(self, env_id):
        while(True):
            obs_list = []
            with th.no_grad():
                # for j in range(self.n_envs):
                cur_obs = self.env.envs[env_id].minimal_reset()
                    # obs_list.append(cur_obs)
            # final_dict = {}
            # for key in cur_obs.keys():
            #     final_dict[key] = th.stack([x[key] for x in obs_list], 0)
            yield cur_obs
    

    def __iter__(self):
        worker_info = th.utils.data.get_worker_info()
        if worker_info is None:
            return self.iterable_func(0)
        else:
            worker_id = worker_info.id
            return self.iterable_func(worker_id)
        # return self.iterable_func()

class MultipleProgramBCEnvDataset(BCEnvDataset):
    def __init__(self, bc_env, n_iters, latent_execution_rate, 
                 le_add_noise, le_only_origins=["WS", "NR"], fetch_reward=False, *args, **kwargs):
        super(MultipleProgramBCEnvDataset, self).__init__(bc_env, n_iters ,*args, **kwargs)
        self.program_list = None
        self.latent_execution_rate = latent_execution_rate
        self.le_add_noise = le_add_noise
        self.rand_counter = 0
        self.idx_counter = 0
        self.random_samples_len = int(1e5)
        self.le_only_origins = le_only_origins
        self.fetch_reward = fetch_reward
        

    def update_program_list(self, program_list):
        self.program_list =  [x.copy() for x  in program_list]
        random.shuffle(self.program_list)
        self.random_samples = np.random.uniform(size = self.random_samples_len)
        
    def __len__(self):
        # let it be the len of pr
        return self.n_iters
    
    def update_ler(self, new_rate):
        self.latent_execution_rate = new_rate

    def iterable_func(self, iter_start, iter_end, limit):
        counter = 0
        while(True):
            if self.idx_counter % len(self.program_list) == 0:
                random.shuffle(self.program_list)
            

            self.idx_counter = (self.idx_counter+1)
            if self.idx_counter == iter_end:
                self.idx_counter = iter_start

            cur_prog_dict = self.program_list[self.idx_counter]


            origin = cur_prog_dict['origin']
            if origin in self.le_only_origins:
                latent_execution = True
            else:
                rand_sample = self.random_samples[self.rand_counter]
                self.rand_counter = (self.rand_counter + 1) % self.random_samples_len
                if rand_sample < self.latent_execution_rate:
                    latent_execution = True
                else:
                    latent_execution = False

            output = dataloader_format(cur_prog_dict, self.env.envs[0], latent_execution, self.le_add_noise, self.fetch_reward)
            if counter < limit:
                counter += 1
                yield output
            else:
                return output

    def __iter__(self):
        worker_info = th.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.program_list)
            limit = self.n_iters
        else:
            per_worker = int(math.ceil((len(self.program_list)) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.program_list))
            self.idx_counter = 0
            limit = self.n_iters / worker_info.num_workers
        return self.iterable_func(iter_start, iter_end, limit)
    
    def __getitem__(self, idx):
        
        if idx % len(self.program_list) == 0:
            random.shuffle(self.program_list)

        idx = idx % len(self.program_list)
        cur_prog_dict = self.program_list[idx]


        origin = cur_prog_dict['origin']
        if origin in self.le_only_origins:
            latent_execution = True
        else:
            rand_sample = self.random_samples[self.rand_counter]
            self.rand_counter = (self.rand_counter + 1) % self.random_samples_len
            if rand_sample < self.latent_execution_rate:
                latent_execution = True
            else:
                latent_execution = False

        output = dataloader_format(cur_prog_dict, self.env.envs[0], latent_execution, self.le_add_noise)

        return output
    

