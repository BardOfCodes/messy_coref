
import time
import torch.multiprocessing as mp
import random
import _pickle as cPickle
from torch.multiprocessing import Pool, Process, set_start_method
import os
import numpy as np
import torch as th
from .utils import get_rand_sample_list
from .diff_opt_helper import parallel_do
from .graph_sweep_helper import parallel_gs
from .code_splice_helper import parallel_code_splice
from .noisy_rewrite_helper import parallel_nr

# from .pc_diff_opt_helper import parallel_do
# from .pc_graph_sweep_helper import parallel_gs
# from .pc_code_splice_helper import parallel_code_splice
from CSG.utils.train_utils import load_all_weights

from .subexpr_cache import FaissIVFCache, MergeSplicerCache
from .sa_subexpr_cache import SAFaissIVFCache
from CSG.env.csg3d.sub_parsers import RotationFCSG3DParser, FCSG3DParser
from CSG.env.csg2d.sub_parsers import RotationFCSG2DParser, FCSG2DParser
from .sa_gs_helper import parallel_sa_gs
from .sa_cs_helper import parallel_sa_code_splice

class DifferentiableOptimizer():

    def __init__(self, config, max_length, save_dir, logger, length_alpha):

        # DIFFERENTIABLE OPTIMIZATION
        self.enable = config.ENABLE
        self.select_random = config.SELECT_RANDOM
        self.do_n_proc = config.N_PROC
        self.do_n_steps = config.N_STEPS
        self.do_lr = config.LR
        self.do_sample_count = config.SAMPLE_COUNT
        self.do_exhaustive = config.EXHAUSTIVE
        self.language_name = config.LANGUAGE_NAME
        self.length_alpha = length_alpha
        if config.INPUT_GATING:
            self.valid_input_origins = ["BS"]
        else:
            self.valid_input_origins = ["BS", "GS", "CS"]
            # self.valid_input_origins = ["BS", "GS",]

        self.common_setting(max_length, save_dir, logger)

    def common_setting(self, max_length, save_dir, logger):
        self.max_length = max_length
        self.save_dir = save_dir
        self.logger = logger

    def rewrite_programs(self, temp_env, best_program_dict, tensorboard_step, quantize, *args, **kwargs):

        target_slots, target_ids, mini_prog_lists = get_rand_sample_list(best_program_dict, self.do_sample_count,
                                                                         self.do_n_proc,
                                                                         exhaustive=self.do_exhaustive,
                                                                         select_random=self.select_random,
                                                                         valid_origin=self.valid_input_origins,
                                                                         n_prog_per_item=1, remove_do_fail=True)
        processes = []
        temp_env.program_generator.set_execution_mode(
            th.device("cpu"), th.float32)
        start_time = time.time()

        for proc_id in range(self.do_n_proc):
            print("starting process %d of %d" % (proc_id, self.do_n_proc))
            temp_env.program_generator.load_specific_data(
                target_slots[proc_id], target_ids[proc_id])
            p = mp.Process(target=parallel_do, args=(proc_id, mini_prog_lists[proc_id], temp_env,
                                                     self.save_dir, self.do_n_steps, self.do_lr,
                                                     quantize, self.length_alpha))
            p.start()
            processes.append(p)
        temp_env.program_generator.reload_data()

        for p in processes:
            p.join()

        process_time = time.time() - start_time
        all_prog_objs, failed_keys = self.do_log_and_retrieve_results(
            process_time, tensorboard_step)

        return all_prog_objs, failed_keys

    def do_log_and_retrieve_results(self, process_time, tensorboard_step):

        all_updated_rewards = []
        all_previous_rewards = []
        all_prog_objs = []
        for i in range(self.do_n_proc):
            file_name = self.save_dir + "/do_%d.pkl" % i
            with open(file_name, 'rb') as f:
                prog_obj, updated_rewards, previous_rewards, failed_keys = cPickle.load(
                    f)
                all_updated_rewards.extend(updated_rewards)
                all_previous_rewards.extend(previous_rewards)
                all_prog_objs.extend(prog_obj)
                os.system("rm %s" % (file_name))

        all_previous_rewards = np.array(all_previous_rewards)
        all_updated_rewards = np.array(all_updated_rewards)
        reward_delta = all_updated_rewards - all_previous_rewards
        increment_counter = reward_delta > 0
        pc_improvement = reward_delta / (all_previous_rewards + 1e-9)
        pc_improvement = pc_improvement[increment_counter]

        self.logger.record("search/DO_updated reward",
                           np.nanmean(all_updated_rewards))
        self.logger.record("search/DO_previous reward",
                           np.nanmean(all_previous_rewards))
        self.logger.record("search/DO Avg Reward Delta",
                           np.sum(reward_delta)/(np.sum(increment_counter) + 1e-9))
        self.logger.record("search/DO_pc_improvement",
                           np.nanmedian(pc_improvement))
        self.logger.record("search/DO_pc_programs_improved",
                           np.nanmean(increment_counter))
        self.logger.record("search/DO_new_progs",
                           len(all_prog_objs))
        self.logger.record("Training data/DO time", process_time)
        self.logger.dump(tensorboard_step)

        return all_prog_objs, failed_keys


class GraphSweeper(DifferentiableOptimizer):

    def __init__(self, config, max_length, save_dir, logger, length_alpha):

        # DEAD CODE REMOVAL
        self.enable = config.ENABLE
        self.select_random = config.SELECT_RANDOM
        self.gs_n_proc = config.N_PROC
        self.gs_sample_count = config.SAMPLE_COUNT
        self.gs_exhaustive = config.EXHAUSTIVE
        self.td_mode = config.TD_MODE
        self.reward_threshold = config.REWARD_THRESHOLD
        self.length_alpha = length_alpha
        if config.INPUT_GATING:
            self.valid_input_origins = ["BS"]
        else:
            self.valid_input_origins = ["BS", "DO", "CS"]
        self.language_name = config.LANGUAGE_NAME
        if "SA" in self.language_name:
            self.graph_sweep_function = parallel_sa_gs
        else:
            self.graph_sweep_function = parallel_gs
        self.common_setting(max_length, save_dir, logger)

    def rewrite_programs(self, temp_env, best_program_dict, tensorboard_step, *args, **kwargs):

        target_slots, target_ids, mini_prog_lists = get_rand_sample_list(best_program_dict, self.gs_sample_count,
                                                                         self.gs_n_proc,
                                                                         exhaustive=True,
                                                                         select_random=self.select_random,
                                                                         valid_origin=self.valid_input_origins,
                                                                         n_prog_per_item=1)
        processes = []
        temp_env.program_generator.set_execution_mode(
            th.device("cpu"), th.float16)
        start_time = time.time()
        if self.gs_exhaustive:
            count_limit = len(target_slots[0])
        else:
            count_limit = self.gs_sample_count//self.gs_n_proc
        for proc_id in range(self.gs_n_proc):
            temp_env.program_generator.load_specific_data(
                target_slots[proc_id], target_ids[proc_id])
            p = mp.Process(target=self.graph_sweep_function, args=(proc_id, mini_prog_lists[proc_id], temp_env, self.save_dir,
                                                                   self.td_mode, self.reward_threshold, self.length_alpha, count_limit))
            p.start()
            processes.append(p)
        temp_env.program_generator.reload_data()

        for p in processes:
            p.join()
        process_time = time.time() - start_time
        all_prog_objs = self.gs_log_and_retrieve_results(
            process_time, tensorboard_step)
        failed_keys = None

        return all_prog_objs, failed_keys

    def gs_log_and_retrieve_results(self, process_time, tensorboard_step, tag=""):

        all_prog_objs = []
        all_previous_lengths = []
        all_updated_lengths = []
        all_updated_rewards = []
        all_previous_rewards = []
        all_td_delta = []
        all_bu_delta = []

        for i in range(self.gs_n_proc):
            file_name = self.save_dir + "/gs_%d.pkl" % i
            with open(file_name, 'rb') as f:
                prog_objs, updated_rewards, previous_rewards, updated_lengths, previous_lengths, td_delta, bu_delta = cPickle.load(
                    f)
                all_previous_lengths.extend(previous_lengths)
                all_updated_lengths.extend(updated_lengths)
                all_previous_rewards.extend(previous_rewards)
                all_updated_rewards.extend(updated_rewards)
                all_td_delta.extend(td_delta)
                all_bu_delta.extend(bu_delta)
                all_prog_objs.extend(prog_objs)
                os.system("rm %s" % (file_name))

        all_td_delta = np.array(all_td_delta)
        all_bu_delta = np.array(all_bu_delta)
        td_counter = all_td_delta > 0
        bu_counter = all_bu_delta > 0

        all_previous_lengths = np.array(all_previous_lengths)
        all_updated_lengths = np.array(all_updated_lengths)
        length_delta = all_previous_lengths - all_updated_lengths
        increment_counter = length_delta > 0
        pc_improvement = length_delta / (all_previous_lengths + 1e-9)
        pc_improvement = pc_improvement[increment_counter]

        self.logger.record("search/%sGS_prev Length" %
                           tag, np.nanmean(all_previous_lengths))
        self.logger.record("search/%sGS_new Length" %
                           tag, np.nanmean(all_updated_lengths))
        self.logger.record("search/%sGS_prev reward" %
                           tag, np.nanmean(all_previous_rewards))
        self.logger.record("search/%sGS_new reward" %
                           tag, np.nanmean(all_updated_rewards))
        self.logger.record("search/%sGS_TD pc_programs_improved" %
                           tag, np.nanmean(td_counter))
        self.logger.record("search/%sGS_BU pc_programs_improved" %
                           tag, np.nanmean(bu_counter))
        self.logger.record("search/%sGS_pc_improvement" %
                           tag, np.nanmedian(pc_improvement))
        self.logger.record("search/%sGS_pc_programs_improved" %
                           tag, np.nanmean(increment_counter))
        self.logger.record("search/GS_new_progs",
                           len(all_prog_objs))
        self.logger.record("Training data/%sGS time" % tag, process_time)
        self.logger.dump(tensorboard_step)

        return all_prog_objs


class CodeSplicer(DifferentiableOptimizer):

    def __init__(self, config, max_length, save_dir, logger, model_info, device, init_model_path, length_alpha, *args, **kwargs):

        # DEAD CODE REMOVAL
        self.enable = config.ENABLE
        self.select_random = config.SELECT_RANDOM
        self.cs_n_proc = config.N_PROC
        self.cs_sample_count = config.SAMPLE_COUNT
        self.cs_top_k = config.TOP_K
        self.cs_max_bool_count = config.MAX_BOOL_COUNT
        self.cs_rewrite_limit = config.REWRITE_LIMIT
        self.cs_exhaustive = config.EXHAUSTIVE
        self.cs_node_masking_req = config.NODE_MASKING_REQ
        self.cs_dummy_node = config.DUMMY_NODE
        self.cs_run_gs = config.RUN_GS
        self.cs_return_top_k = config.RETURN_TOP_K
        self.cs_use_canonical = config.USE_CANONICAL
        # For log probability
        self.use_probs = config.USE_PROBS
        self.logprob_threshold = config.LOGPROB_THRESHOLD
        self.reward_based_thresh = config.REWARD_BASED_THRESH
        self.valid_nodes = config.VALID_NODES
        self.higher_language = config.HIGHER_LANGUAGE
        self.fcsg_mode = config.FCSG_MODE
        # For loading models
        self.device = device
        self.model_info = model_info
        self.init_model_path = init_model_path
        # Explicitly set for eval
        self.eval_mode = False
        self.merge_splice = config.MERGE_SPLICE.ENABLE
        if self.merge_splice:
            self.merge_splice_config = config.MERGE_SPLICE
        else:
            self.merge_splice_config = None

        self.cache_config = config.CACHE_CONFIG
        self.length_alpha = length_alpha
        if config.INPUT_GATING:
            self.valid_input_origins = ["BS"]
        else:
            # self.valid_input_origins = ["BS"]
            self.valid_input_origins = ["BS", "DO", "GS", "CS"]
            # self.valid_input_origins = ["BS", "GS", "DO"]
        self.cs_counter = 0

        self.language_name = config.LANGUAGE_NAME
        if "SA" in self.language_name:
            self.code_splice_function = parallel_sa_code_splice
        else:
            self.code_splice_function = parallel_code_splice

        self.common_setting(max_length, save_dir, logger)

    def rewrite_programs(self, temp_env, best_program_dict, tensorboard_step, quantize, epoch, save_path, *args, **kwrags):

        start_time = time.time()

        # Get the different parser
        if self.fcsg_mode:
            if '3D' in temp_env.language_name:
                temp_env.program_generator.parser = RotationFCSG3DParser(
                    temp_env.program_generator.parser.module_path, temp_env.program_generator.parser.device)
            else:
                temp_env.program_generator.parser = RotationFCSG2DParser(
                    temp_env.program_generator.parser.module_path, temp_env.program_generator.parser.device)

        temp_env.program_generator.set_execution_mode(
            th.device("cuda"), th.float16)
        if "SA" in self.language_name:
            subexpr_cache = SAFaissIVFCache(self.save_dir, self.cache_config, self.merge_splice_config,
                                            eval_mode=self.eval_mode, language_name=self.language_name)
        else:
            if self.merge_splice:
                subexpr_cache = MergeSplicerCache(
                    self.save_dir, self.cache_config, self.merge_splice_config, eval_mode=self.eval_mode, language_name=self.language_name)
            else:
                subexpr_cache = FaissIVFCache(
                    self.save_dir, self.cache_config, eval_mode=self.eval_mode, language_name=self.language_name)

        if epoch > 0:
            subexpr_cache.load_previous_subexpr_cache = True

        subexpr_cache.generate_cache_and_index(
            best_program_dict, temp_env, self.cs_use_canonical)
        if self.merge_splice:
            ms_prog_objs = subexpr_cache.get_merge_spliced_expressions(
                temp_env, self.higher_language,  logger=self.logger, tensorboard_step=tensorboard_step, quantize_expr=quantize, length_alpha=self.length_alpha)

        for key, value in subexpr_cache.stats.items():
            self.logger.record("subexpr_health/%s" % key, value)
        target_slots, target_ids, mini_prog_lists = get_rand_sample_list(best_program_dict, self.cs_sample_count,
                                                                         self.cs_n_proc,
                                                                         exhaustive=self.cs_exhaustive,
                                                                         select_random=self.select_random,
                                                                         valid_origin=self.valid_input_origins,
                                                                         n_prog_per_item=1, remove_cs_fail=True,
                                                                         )
        #  n_prog_per_item=1, remove_cs_fail=False)

        temp_env.program_generator.set_execution_mode(
            th.device("cpu"), th.float16)

        if self.cs_counter == 0:
            save_path = self.init_model_path
        if self.use_probs:
            policy, _, _, _ = load_all_weights(save_path, temp_env, instantiate_model=True,
                                               model_info=self.model_info, device=self.device)
            # Important
            policy.training = False
            policy.eval()
            policy.set_training_mode(False)
            policy.disable_mask()
            policy.share_memory()
        else:
            policy = None

        th.cuda.empty_cache()
        processes = []
        for proc_id in range(self.cs_n_proc):
            print("Starting process %d" % proc_id)
            temp_env.program_generator.load_specific_data(
                target_slots[proc_id], target_ids[proc_id])
            p = mp.Process(target=self.code_splice_function, args=(proc_id, mini_prog_lists[proc_id],
                                                                   temp_env, subexpr_cache, self.save_dir,
                                                                   self.cs_top_k, self.cs_max_bool_count, self.cs_rewrite_limit,
                                                                   self.cs_node_masking_req, self.cs_dummy_node, self.cs_run_gs,
                                                                   self.cs_return_top_k, quantize, self.cs_use_canonical,
                                                                   policy, self.use_probs, self.logprob_threshold, self.reward_based_thresh,
                                                                   self.valid_nodes, self.higher_language, self.max_length, temp_env.max_expression_complexity,
                                                                   self.length_alpha))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        if self.use_probs:
            policy.cpu()
            del policy, _
        del subexpr_cache

        temp_env.program_generator.reload_data()

        process_time = time.time() - start_time

        all_prog_objs, failed_keys = self.cs_log_and_retrieve_results(
            process_time, tensorboard_step)

        self.cs_counter += 1

        if self.merge_splice:
            all_prog_objs.extend(ms_prog_objs)
        # Get the different parser
        if self.fcsg_mode:
            if '3D' in temp_env.language_name:
                temp_env.program_generator.parser = FCSG3DParser(
                    temp_env.program_generator.parser.module_path, temp_env.program_generator.parser.device)
            else:
                temp_env.program_generator.parser = FCSG2DParser(
                    temp_env.program_generator.parser.module_path, temp_env.program_generator.parser.device)

        return all_prog_objs, failed_keys

    def cs_log_and_retrieve_results(self, process_time, tensorboard_step):

        all_previous_lengths = []
        all_updated_lengths = []
        all_updated_rewards = []
        all_previous_rewards = []
        all_num_rewrites = []
        all_prog_objs = []
        all_previous_logprob = []
        all_updated_logprob = []
        for i in range(self.cs_n_proc):
            file_name = self.save_dir + "/cs_%d.pkl" % i
            with open(file_name, 'rb') as f:
                prog_objs, updated_rewards, previous_rewards, updated_lengths, previous_lengths, \
                    updated_logprobs, previous_logprobs, num_rewrites, failed_keys = cPickle.load(
                        f)
                all_previous_lengths.extend(previous_lengths)
                all_updated_lengths.extend(updated_lengths)
                all_updated_rewards.extend(updated_rewards)
                all_previous_rewards.extend(previous_rewards)
                all_updated_logprob.extend(updated_logprobs)
                all_previous_logprob.extend(previous_logprobs)
                all_num_rewrites.extend(num_rewrites)
                all_prog_objs.extend(prog_objs)
                os.system("rm %s" % (file_name))

        all_previous_rewards = np.array(all_previous_rewards)
        all_updated_rewards = np.array(all_updated_rewards)
        reward_delta = all_updated_rewards - all_previous_rewards
        increment_counter = reward_delta > 0
        pc_improvement = reward_delta / (all_previous_rewards + 1e-9)
        pc_improvement = pc_improvement[increment_counter]

        self.logger.record("search/CS_updated Length",
                           np.nanmean(all_updated_lengths))
        self.logger.record("search/CS_previous Length",
                           np.nanmean(all_previous_lengths))
        self.logger.record("search/CS_updated logprob",
                           np.nanmean(all_updated_logprob))
        self.logger.record("search/CS_previous logprob",
                           np.nanmean(all_previous_logprob))
        self.logger.record("search/CS_updated reward",
                           np.nanmean(all_updated_rewards))
        self.logger.record("search/CS_previous reward",
                           np.nanmean(all_previous_rewards))
        self.logger.record("search/CS Avg Reward Delta",
                           np.sum(reward_delta)/(np.sum(increment_counter) + 1e-9))
        self.logger.record("search/CS_pc_improvement",
                           np.nanmedian(pc_improvement))
        self.logger.record("search/CS_pc_programs_improved",
                           np.nanmean(increment_counter))
        self.logger.record("search/CS_avg_num_rewrites",
                           np.nanmean(all_num_rewrites))
        self.logger.record("search/CS_new_progs",
                           len(all_prog_objs))
        self.logger.record("Training data/CS time", process_time)
        self.logger.dump(tensorboard_step)

        return all_prog_objs, failed_keys


class NoisyRewriter(DifferentiableOptimizer):

    def __init__(self, config, max_length, save_dir, logger, length_alpha):

        # DIFFERENTIABLE OPTIMIZATION
        self.enable = config.ENABLE
        self.select_random = config.SELECT_RANDOM
        self.n_proc = config.N_PROC
        self.n_augs = config.N_AUGS
        self.noise_rate = config.NOISE_RATE
        self.reward_threshold_pow = config.POWER

        self.numeric_sigma = config.NUMERIC_SIGMA
        self.discrete_delta = config.DISCRETE_DELTA
        self.max_augs_per_sample = config.MAX_AUGS_PER_SAMPLE

        self.language_name = config.LANGUAGE_NAME
        self.length_alpha = length_alpha
        if config.INPUT_GATING:
            self.valid_input_origins = ["BS"]
        else:
            self.valid_input_origins = ["BS", "DO", "GS", "CS"]

        self.common_setting(max_length, save_dir, logger)

    def rewrite_programs(self, temp_env, program_list, tensorboard_step, quantize, *args, **kwargs):

        # How to convert it?

        target_slots = [[] for j in range(self.n_proc)]
        target_ids = [[] for j in range(self.n_proc)]
        mini_prog_lists = [[] for x in range(self.n_proc)]

        for ind, program in enumerate(program_list):
            index = ind % self.n_proc
            cur_slot_id = program['slot_id']
            cur_target_id = program['target_id']
            origin = program['origin']
            mini_prog_lists[index].append(
                ((cur_slot_id, cur_target_id, origin), program))
            target_slots[index].append(cur_slot_id)
            target_ids[index].append(cur_target_id)

        processes = []
        temp_env.program_generator.set_execution_mode(
            th.device("cpu"), th.float32)
        start_time = time.time()

        for proc_id in range(self.n_proc):
            print("starting process %d of %d" % (proc_id, self.n_proc))
            temp_env.program_generator.load_specific_data(
                target_slots[proc_id], target_ids[proc_id])
            p = mp.Process(target=parallel_nr, args=(proc_id, mini_prog_lists[proc_id], temp_env,
                                                     self.save_dir, self.n_augs, self.noise_rate, self.reward_threshold_pow,
                                                     self.numeric_sigma, self.discrete_delta, self.max_augs_per_sample,
                                                     self.length_alpha))
            p.start()
            processes.append(p)
        temp_env.program_generator.reload_data()

        for p in processes:
            p.join()

        process_time = time.time() - start_time
        all_prog_objs, failed_keys = self.log_and_retrieve_results(
            process_time, tensorboard_step)

        return all_prog_objs, failed_keys

    def log_and_retrieve_results(self, process_time, tensorboard_step):

        all_updated_rewards = []
        all_previous_rewards = []
        all_prog_objs = []
        for i in range(self.n_proc):
            file_name = self.save_dir + "/nr_%d.pkl" % i
            with open(file_name, 'rb') as f:
                prog_obj, updated_rewards, previous_rewards, failed_keys = cPickle.load(
                    f)
                all_updated_rewards.extend(updated_rewards)
                all_previous_rewards.extend(previous_rewards)
                all_prog_objs.extend(prog_obj)
                os.system("rm %s" % (file_name))

        all_previous_rewards = np.array(all_previous_rewards)
        all_updated_rewards = np.array(all_updated_rewards)
        reward_delta = all_updated_rewards - all_previous_rewards
        increment_counter = reward_delta > 0
        pc_improvement = reward_delta / (all_previous_rewards + 1e-9)
        pc_improvement = pc_improvement[increment_counter]

        self.logger.record("search/NR_updated reward",
                           np.nanmean(all_updated_rewards))
        self.logger.record("search/NR_previous reward",
                           np.nanmean(all_previous_rewards))
        self.logger.record("search/NR Avg Reward Delta",
                           np.sum(reward_delta)/(np.sum(increment_counter) + 1e-9))
        self.logger.record("search/NR_pc_improvement",
                           np.nanmedian(pc_improvement))
        self.logger.record("search/NR_pc_programs_improved",
                           np.nanmean(increment_counter))
        self.logger.record("Training data/NR time", process_time)
        self.logger.record("search/NR_new_progs",
                           len(all_prog_objs))
        self.logger.dump(tensorboard_step)

        return all_prog_objs, failed_keys
