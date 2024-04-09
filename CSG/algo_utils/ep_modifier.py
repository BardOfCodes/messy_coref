import numpy as np
from CSG.env.modified_env import ModifierCSG
from CSG.env.cad_env import CADCSG
import cv2
# from CSG.env.csg2d.differentiable_stack import DifferentiableStack, OccupancyStack, myround, BatchDifferentiableStack, BatchOccupancyStack
from CSG.env.reward_function import chamfer
# from CSG.env.csg2d.parsers import Parser
from tensorboard import program
import torch
import time
import _pickle as cPickle
from CSG.utils.metrics import MetricObj, DiffOptMetricExtractor, DIFF_OPT_METRICS, BASE_METRICS, DefaultMetricExtractor
from CSG.utils.beam_utils import batch_parallel_beam_search
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method
from .ep_helper import run_parallel, get_opt_programs_with_perturb, get_opt_programs_with_perturb_occupancy, get_opt_programs
from collections import defaultdict
import os
from pathlib import Path


def return_zero(*args):
    return 0
try:
     set_start_method('spawn')
except RuntimeError:
    pass



class NoModification():
    
    def __init__(self, config) -> None:
        # self.mod_env = ModifierCSG(config=config.CONFIG, phase_config=config.PHASE_CONFIG, 
        #                            seed=0, n_proc=1, proc_id=0)
        pass
    
    def fetch_unmodified(self, rollout_buffer, info, env_id, episode_start, episode_end,
               *args, **kwargs):
                   
        
        episode_length = episode_end - episode_start
        
        episode_dict = {'observations': dict(),
                        'actions': None,
                        'rewards' : None,
                        'target_id': info['target_id'],
                        'slot_id': info['slot_id'],
                        'length': episode_length
                        }
        # Observations
        # for key, value in rollout_buffer.observations.items():
        #     episode_dict['observations'][key] = value[episode_start: episode_end, env_id].copy()
          
        # Actions:
        episode_dict['actions'] = rollout_buffer.actions[episode_start: episode_end, env_id].astype(np.int32).copy()
        # Rewards:
        episode_dict['rewards'] = rollout_buffer.rewards[episode_start: episode_end, env_id].copy()
        
        return episode_dict
    
    
    def modify(self, rollout_buffer, info, env_id, episode_start, episode_end,
               *args, **kwargs):
        """
        Basic Implementation: No Modification
        """
        episode_dict = self.fetch_unmodified(rollout_buffer, info, env_id, episode_start, episode_end, *args, **kwargs)
        return [episode_dict]
    def bulk_modify(self, *args, **kwargs):
        return [], {}
    
class HindsightModifier(NoModification):

    def __init__(self, config) -> None:
        self.collect_base = config.COLLECT_BASE
    
    
    def modify(self, rollout_buffer, info, env_id, episode_start, episode_end,
               *args, **kwargs):
        
        new_target = info['predicted_canvas']
        episode_length = episode_end - episode_start
        
        episode_dict = {'observations': dict(),
                        'actions': None,
                        'rewards' : None,
                        'length': episode_length
                        }
        
        for key, value in rollout_buffer.observations.items():
            episode_dict['observations'][key] = value[episode_start: episode_end, env_id].copy()
            if key == 'obs':
                # Also replace target:
                episode_dict['observations'][key][:, :1] = new_target.copy()
            
        # Actions:
        episode_dict['actions'] = rollout_buffer.actions[episode_start: episode_end, env_id].copy()
        # Rewards:
        episode_dict['rewards'] = np.zeros((episode_length))
        episode_dict['rewards'][-1] = 1.0
        ep_list = [episode_dict]
        
        if self.collect_base:
            base_episode = super(HindsightModifier, self).fetch_unmodified(rollout_buffer, env_id, episode_start, episode_end,)
            ep_list.append(base_episode)
        return ep_list
            

class RefactorModifier(NoModification):
    
    """
    Perform refactoring of predicted program on the fly. 
    """
    def __init__(self, config) -> None:
        self.mod_env = ModifierCSG(config=config.CONFIG, phase_config=config.PHASE_CONFIG, 
                                   seed=0, n_proc=1, proc_id=0)
        self.mod_env.reset()
        self.collect_base = config.COLLECT_BASE
        
        

    def modify(self, rollout_buffer, info, env_id, episode_start, episode_end,
               *args, **kwargs):
        
        target_canvas = info['target_canvas']
        pred_expression = info['predicted_expression']
        new_expression, refactored_observations, \
                refactored_actions, refactored_rewards = self.mod_env.get_refactored_experience(pred_expression, target_canvas)
        
        ep_list = []
        if new_expression:
            if len(new_expression) < len(pred_expression):
                episode_length = len(new_expression)
                
                episode_dict = {'observations': dict(),
                                'actions': None,
                                'rewards' : None,
                                'length': episode_length
                                }
                
                for key, value in refactored_observations.items():
                    episode_dict['observations'][key] = value.copy()
                    
                # Actions:
                episode_dict['actions'] = refactored_actions.copy()
                # Rewards:
                episode_dict['rewards'] = refactored_rewards
                ep_list.append(episode_dict)
                
            else:
                base_episode = super(RefactorModifier, self).fetch_unmodified(rollout_buffer, env_id, episode_start, episode_end,)
                ep_list.append(base_episode)
            
        else:
            base_episode = super(RefactorModifier, self).fetch_unmodified(rollout_buffer, env_id, episode_start, episode_end,)
            ep_list.append(base_episode)
                
        if self.collect_base:
            base_episode = super(RefactorModifier, self).fetch_unmodified(rollout_buffer, env_id, episode_start, episode_end,)
            ep_list.append(base_episode)
        return ep_list
    

class DiffOptModifier(RefactorModifier):
    
    def __init__(self, config) -> None:
        # self.mod_env = ModifierCSG(config=config.CONFIG, phase_config=config.PHASE_CONFIG, 
        #                            seed=0, n_proc=1, proc_id=0)
        ## Just need the stack optimizers 
        # self.batch_diff_stack = BatchDifferentiableStack(max_len=7, canvas_shape=[64, 64])
        self.batch_diff_stack = BatchOccupancyStack(max_len=7, canvas_shape=[64, 64])
        
        self.parser = Parser()
        
        self.N_STEPS = 200
        self.LR = 0.75
        self.SDF_CAP = 0.001
        self.batch_size = 150
        
        self.n_proc = 8
        self.save_loc = config.SAVE_LOCATION
        Path(config.SAVE_LOCATION).mkdir(parents=True, exist_ok=True)

        self.mod_env = ModifierCSG(config=config.CONFIG, phase_config=config.PHASE_CONFIG, 
                                   seed=0, n_proc=1, proc_id=0)
        self.mod_env.reset()
        self.collect_base = config.COLLECT_BASE
        
        
    def optimize_episode(self, express, target_data):
        
        express = "".join(express)
        new_prog = self.parser.parse(express) 
        
        ## For input
        thresholded_target = target_data[0].astype(np.uint8)
        target_distances = cv2.distanceTransform(1 - thresholded_target, cv2.DIST_L2, maskSize=0)
        target_distances = target_distances / np.sqrt(64 * 64)
        
        self.diff_stack.generate_stack(new_prog)
        # diff_init_output = self.diff_stack.get_top() # stack.items[0]
        # diff_init_output = diff_init_output.detach().numpy()
        # draw_init = diff_init_output<=self.SDF_CAP
        optimizer = torch.optim.Adam(self.diff_stack.variables, self.LR)
        
        distances_th = torch.from_numpy(target_distances)
        for i in range(self.N_STEPS):
            
            self.diff_stack.recompute_stack()
            output = self.diff_stack.get_top()
            distances_th = torch.from_numpy(target_distances).cuda()
            loss = torch.nn.functional.mse_loss(output, distances_th)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        self.diff_stack.recompute_stack()
        for idx, token in enumerate(self.diff_stack.variables):
            if idx % 2 == 0:
                # Its location
                token = myround(token, clamp_min=8, clamp_max=56)
            else:
                token = myround(token, base=4, clamp_min=4, clamp_max=32)
            self.diff_stack.variables[idx] = token
            
        final_program = self.diff_stack.get_expression()
        final_expression = self.parser.prog_to_exp(final_program)
        return final_expression
    
    
    def bulk_modify(self, collected_episodes, *args, **kwargs):
        ## Spwan multiple batch_diff stacks with different set of programs:
        start_time = time.time()
        n_episodes = len(collected_episodes)
        target_episodes = [[] for j in range(self.n_proc)]
        
        for i in range(n_episodes):
            target_episodes[i % self.n_proc].append(collected_episodes[i])
        processes = []
        metric_obj = MetricObj(DIFF_OPT_METRICS, DiffOptMetricExtractor, self.mod_env, 
                            gt_program=False, 
                            save_predictions=True)
        optimization_function =  get_opt_programs
        for proc_id in range(self.n_proc):
            p = mp.Process(target=run_parallel, args=(proc_id, target_episodes[proc_id], 
                                                      self.batch_diff_stack, self.parser, self.mod_env, self.LR, self.N_STEPS, self.save_loc,
                                                      self.batch_size, metric_obj, optimization_function))
            p.start()
            processes.append(p) 
        
        for p in processes:
            p.join()
        ep_list = []
        reward_ratio = []
        ep_accept_rate = []
        for i in range(self.n_proc):
            file_name = self.save_loc + "/_%d.pkl" % i
            with open(file_name, 'rb') as f:
                temp_ep_list, temp_reward_ratio, temp_ep_accept, temp_metric_obj= cPickle.load(f)
                ep_list.extend(temp_ep_list)
                reward_ratio.extend(temp_reward_ratio)
                ep_accept_rate.extend(temp_ep_accept)
                metric_obj.fuse(temp_metric_obj)
            ## Delete them:
            os.system("rm %s" % (file_name))
        
        end_time = time.time()
        print("Total_time ", end_time - start_time)
        print("Accepted episodes ", len(ep_list), "from ", len(collected_episodes))
        stats = dict(reward_ratio=np.nanmedian(reward_ratio), total_time=end_time - start_time,
                     ep_accept_rate=np.nanmean(ep_accept_rate))
        final_metrics, _, _ = metric_obj.return_metrics()
        for key,value in final_metrics.items():
            stats["diffOpt_%s" % key] = value
        return ep_list, stats
        
        
    def modify(self, rollout_buffer, info, env_id, episode_start, episode_end,
               *args, **kwargs):
        
        target_canvas = info['target_canvas']
        pred_expression = info['predicted_expression']
        pred_program = self.parser.parse("".join(pred_expression))
        thresholded_target = target_canvas[0].astype(np.uint8)
        target_distances = cv2.distanceTransform(1 - thresholded_target, cv2.DIST_L2, maskSize=0)
        target_distances = target_distances / np.sqrt(64 * 64)
        # distances_th = torch.from_numpy(target_distances).cuda()
        rewards = rollout_buffer.rewards[episode_start: episode_end, env_id].copy()
        
        episode_length = len(pred_expression)
        episode_dict = {'observations': dict(),
                        'target_canvas': target_canvas,
                        'pred_expression': pred_expression,
                        'pred_programs': pred_program,
                        # 'distances': target_distances,
                        'distances': target_canvas[0],
                        'length': episode_length,
                        'reward': rewards[-1],
                        'target_id': info['target_id'],
                        'slot_id': info['slot_id'],
                        }
        
        return [episode_dict]

    def modify_singular(self, rollout_buffer, info, env_id, episode_start, episode_end,
               *args, **kwargs):
        
        if not info['target_expression']:
            # TODO: Get proper error - this is not triggered.
            raise ValueError("Oracle Modifier requires GT program sequence")
        
        ep_list = []
        
        target_canvas = info['target_canvas']
        pred_expression = info['predicted_expression']
        new_expression = self.optimize_episode(pred_expression, target_canvas)
        token_invalid = [x not in self.mod_env.unique_draw for x in new_expression]
        if (any(token_invalid)):
            print("rejected due to action space restriction")
            return ep_list
        refactored_observations, refactored_actions, refactored_rewards = self.mod_env.generate_experience(new_expression, target_canvas)
        
        episode_length = len(new_expression)
        
        episode_dict = {'observations': dict(),
                        'actions': None,
                        'rewards' : None,
                        'length': episode_length
                        }
        
        for key, value in refactored_observations.items():
            episode_dict['observations'][key] = value.copy()
            
        # Actions:
        episode_dict['actions'] = refactored_actions.copy()
        # Rewards:
        episode_dict['rewards'] = refactored_rewards
        ep_list.append(episode_dict)
                
                
        if self.collect_base:
            base_episode = super(RefactorModifier, self).fetch_unmodified(rollout_buffer, env_id, episode_start, episode_end)
            ep_list.append(base_episode)
            
        return ep_list
    
class PerturbAndDiffOptModifier(DiffOptModifier):
    ## Perturn the input and optimize - Avoid local min:
    
    def __init__(self, config) -> None:
        super(PerturbAndDiffOptModifier, self).__init__(config)
        
        self.n_rounds = 2
        self.DEBUG = True 
        
    def bulk_modify(self, collected_episodes, *args, **kwargs):
        ## Spwan multiple batch_diff stacks with different set of programs:
        start_time = time.time()
        n_episodes = len(collected_episodes)
        target_episodes = [[] for j in range(self.n_proc)]
        
        for i in range(n_episodes):
            target_episodes[i % self.n_proc].append(collected_episodes[i])
        processes = []
        metric_obj = MetricObj(DIFF_OPT_METRICS, DiffOptMetricExtractor, self.mod_env, 
                            gt_program=False, 
                            save_predictions=True)
        ep_list = []
        reward_ratio = []
        ep_accept_rate = []
        for round in range(self.n_rounds):
            if not self.DEBUG: 
                for proc_id in range(self.n_proc):
                # batch_diff_stack = BatchDifferentiableStack(max_len=7, canvas_shape=[64, 64])
                    _metric_obj = MetricObj(DIFF_OPT_METRICS, DiffOptMetricExtractor, self.mod_env, 
                                        gt_program=False, 
                                save_predictions=True)
                    optimization_function =  get_opt_programs_with_perturb_occupancy
                
                    p = mp.Process(target=run_parallel, args=(proc_id, target_episodes[proc_id], 
                                                            self.batch_diff_stack, self.parser, self.mod_env, self.LR, self.N_STEPS, self.save_loc,
                                                            self.batch_size, _metric_obj, optimization_function, round))
                    p.start()
                    processes.append(p) 
            
                for p in processes:
                    p.join()
                    
            for i in range(self.n_proc):
                file_name = self.save_loc + "/round_%d_process_%d.pkl" % (round, i)
                with open(file_name, 'rb') as f:
                    temp_ep_list, temp_reward_ratio, temp_ep_accept, temp_metric_obj= cPickle.load(f)
                    ep_list.extend(temp_ep_list)
                    reward_ratio.extend(temp_reward_ratio)
                    ep_accept_rate.extend(temp_ep_accept)
                    metric_obj.fuse(temp_metric_obj)
                ## Delete them:
                # os.system("rm %s" % (file_name))
        
        end_time = time.time()
        print("Total_time ", end_time - start_time)
        print("Accepted episodes ", len(ep_list), "from ", len(collected_episodes))
        stats = dict(reward_ratio=np.nanmedian(reward_ratio), total_time=end_time - start_time,
                     ep_accept_rate=np.nanmean(ep_accept_rate))
        final_metrics, _, _ = metric_obj.return_metrics()
        for key,value in final_metrics.items():
            stats["diffOpt_%s" % key] = value
        return ep_list, stats
        
    
    
class BeamSearchModifier(DiffOptModifier):
    
    def __init__(self, config) -> None:
        ## Just need the stack optimizers 
        
        self.beam_size = 10
        self.batch_size = 250
        self.n_proc = 4
        self.beam_selector = "rewards"
        self.save_loc = config.SAVE_LOCATION
        
        self.mod_env = ModifierCSG(config=config.CONFIG, phase_config=config.PHASE_CONFIG, 
                                   seed=0, n_proc=1, proc_id=0)
        self.mod_env.reset()
        self.parser = Parser()
        
        self.collect_base = config.COLLECT_BASE
        self.keep_obs = False
    
    def modify(self, rollout_buffer, info, env_id, episode_start, episode_end,
               *args, **kwargs):
        
        target_canvas = info['target_canvas']
        pred_expression = info['predicted_expression']
        pred_program = self.parser.parse("".join(pred_expression))
        # distances_th = torch.from_numpy(target_distances).cuda()
        rewards = rollout_buffer.rewards[episode_start: episode_end, env_id].copy()
        
        episode_length = len(pred_expression)
        episode_dict = {'observations': dict(),
                        'target_id': info['target_id'],
                        'slot_id': info['slot_id'],
                        'pred_expression': pred_expression,
                        'pred_programs': pred_program,
                        'length': 0,
                        'reward': rewards[-1]
                        }
        
        return [episode_dict]
    
    def get_final_episodes(self,metric_obj, initial_rewards):
        ep_list = []
        reward_ratio = []
        
        for i, item in enumerate(metric_obj.predictions):
            final_expression, slot_id, target_id = item
            refactored_actions = np.array([self.mod_env.unique_draw.index(x) for x in final_expression])[:, None]
            cur_reward = metric_obj.metric_dict['reward'][i]
            refactored_rewards = np.zeros(refactored_actions.shape[:1])
            refactored_rewards[-1] = cur_reward
            episode_dict = {'observations': dict(),
                            'actions': refactored_actions.copy(),
                            'rewards' : refactored_rewards,
                            'length': len(final_expression),
                            'slot_id': slot_id,
                            'target_id': target_id,
                            }
            ep_list.append(episode_dict)
            reward_ratio.append(cur_reward/(initial_rewards[i] + 1e-9)) 
        print("final reward ratio", np.median(reward_ratio))
        end_time = time.time()
        stats = dict(reward_ratio=np.median(reward_ratio), num_episodes=len(ep_list))
        
        final_metrics, _, _ = metric_obj.return_metrics()
        for key,value in final_metrics.items():
            stats["beam_search_%s" % key] = value
            
        return ep_list, stats
    
    def bulk_modify(self, collected_episodes, policy):
        ## Spwan multiple batch_diff stacks with different set of programs:
        start_time = time.time()
        n_episodes = len(collected_episodes)
        print("performing beam search for %d episodes" % n_episodes)
        target_episodes = [[] for j in range(self.n_proc)]
        target_slots = [[] for j in range(self.n_proc)]
        initial_rewards = [collected_episodes[x]['reward'] for x in range(n_episodes)]
        reward_seq = [[] for j in range(self.n_proc)]
        for i in range(n_episodes):
            target_episodes[i % self.n_proc].append(collected_episodes[i]['target_id'])
            # ind = self.mod_env.program_generator.get_next_slot(0)
            target_slots[i % self.n_proc].append(collected_episodes[i]['slot_id'])
            reward_seq[i % self.n_proc].append(initial_rewards[i])
        initial_rewards = []
        for r_seq in reward_seq:
            initial_rewards.extend(r_seq)
        
            
        target_lengths = [len(target_slots[j]) for j in range(self.n_proc)]
        processes = []
        metric_obj = MetricObj(BASE_METRICS, DefaultMetricExtractor, self.mod_env, 
                            gt_program=False, 
                            save_predictions=True)
        policy.eval()
        policy.set_training_mode(False)
        policy.action_dist.proba_distribution(torch.zeros([1, 400]).cuda())
        policy.masked_action_dist.proba_distribution(torch.zeros([1, 400]).cuda())
        policy.masked_action_dist.requires_grad = False
        policy.share_memory()
        processes = []
        mp.reductions
        for proc_id in range(self.n_proc):
            p = mp.Process(target=batch_parallel_beam_search, args=(proc_id, policy, self.mod_env, target_slots[proc_id],
                                                            target_episodes[proc_id], metric_obj,
                                                            self.beam_size, self.beam_size, self.beam_selector, self.save_loc, self.batch_size))
            p.start()
            processes.append(p) 
        
        for p in processes:
            p.join()
            
        for i in range(self.n_proc):
            
            with open(self.save_loc + "/_%d.pkl" % i, 'rb') as f:
                temp_metric_obj = cPickle.load(f)
                metric_obj.fuse(temp_metric_obj)
        ### Convert the pred episodes to selected episodes: 
        ep_list, stats = self.get_final_episodes(metric_obj, initial_rewards)
        end_time = time.time()
        stats['total_time']=end_time - start_time
        return ep_list, stats
        
class OracleModifier(RefactorModifier):
    
    """
    Return GT Program for the given problem directly. 
    """
        
    def modify(self, rollout_buffer, info, env_id, episode_start, episode_end,
               *args, **kwargs):
        
        if not info['target_expression']:
            # TODO: Get proper error - this is not triggered.
            raise ValueError("Oracle Modifier requires GT program sequence")
        
        target_canvas = info['target_canvas']
        pred_expression = info['predicted_expression']
        new_expression = info['target_expression']
        refactored_observations, refactored_actions, refactored_rewards = self.mod_env.generate_experience(new_expression, target_canvas)
        
        ep_list = []
        episode_length = len(new_expression)
        
        episode_dict = {'observations': dict(),
                        'actions': None,
                        'rewards' : None,
                        'length': episode_length
                        }
        
        for key, value in refactored_observations.items():
            episode_dict['observations'][key] = value.copy()
            
        # Actions:
        episode_dict['actions'] = refactored_actions.copy()
        # Rewards:
        episode_dict['rewards'] = refactored_rewards
        ep_list.append(episode_dict)
                
                
        if self.collect_base:
            base_episode = super(RefactorModifier, self).fetch_unmodified(rollout_buffer, env_id, episode_start, episode_end)
            ep_list.append(base_episode)
            
        return ep_list


    
    
EP_MODIFIER = {
    'NoModification': NoModification,
    'HindsightModifier': HindsightModifier,
    'RefactorModifier': RefactorModifier,
    "OracleModifier": OracleModifier,
    "DiffOptModifier": DiffOptModifier,
    'BeamSearchModifier': BeamSearchModifier,
    'PerturbAndDiffOptModifier': PerturbAndDiffOptModifier
}  