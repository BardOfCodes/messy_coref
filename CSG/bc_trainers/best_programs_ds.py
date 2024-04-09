from collections import defaultdict
import numpy as np
import torch as th
from .rewrite_engines.sa_utils import sa_convert_hcsg_commands_to_cpu
from .rewrite_engines.rewriters import DifferentiableOptimizer, CodeSplicer
from .rewrite_engines.utils import ALL_ORIGINS, get_program_list, MERGABLE_SANS_BS_TYPES, probabilistic_program_list



class SRTBestProgramsDS:
    
    def __init__(self, config, logger, rl_mode, rl_moving_baseline_alpha, max_length):
        
        self.rl_mode = rl_mode
        self.rl_moving_baseline_alpha = rl_moving_baseline_alpha
        self.mean_reward = 0
        self.bpd = defaultdict(list)
        self.logger = logger
        self.training_data_selection = config.TRAINING_DATA_SELECTION
        self.best_prog_count = config.BEST_PROG_COUNT
        self.store_single = config.STORE_SINGLE
        self.gen_sample_rate = config.GEN_SAMPLE_RATE
        self.n_samples_per_item = config.N_SAMPLES_PER_ITEM
        self.data_sampling_temperature = config.DATA_SAMPLING_TEMPERATURE
        self.multiply_inputs = config.MULTIPLY_INPUT
        self.max_length = max_length
    
    def initialize_bpd(self):
        
        self.bpd = defaultdict(list)

        if self.rl_mode:
            # Check this
            new_bpd = defaultdict(list)
            for key, value in self.bpd.items():
                origin = key[2]
                if not origin == "BS":
                    bs_key = (key[0], key[1], "BS")
                    if bs_key in  self.bpd.keys():
                        if value[0]['reward'] >= self.bpd[bs_key][0]['reward']:
                            new_bpd[key] = value 
            self.bpd = new_bpd
        
    def set_best_programs(self, prog_objs, temp_env):
        # if self.rl_mode:
        temp_env.program_generator.set_execution_mode(th.device("cuda"), th.float16)
        for obj in prog_objs:
            expression = obj['expression']
            # expression.append("$")
            slot_id = obj['slot_id']
            target_id = obj['target_id']
            reward = obj['reward']
            log_prob = obj['log_prob']
            origin = obj['origin']
            do_fail = obj['do_fail']
            cs_fail = obj['cs_fail']
            # ep_length = all_metrics['ep_length'][idx]
            main_actions = temp_env.action_space.expression_to_action(
                expression)
            # expression_t = temp_env.action_space.action_to_expression(main_actions)
            ep_length = main_actions.shape[0]
            actions = np.zeros(self.max_length, dtype=np.int64)
            if ep_length > self.max_length:
                continue
            actions[:ep_length] = main_actions
            # execute and load?

            if origin in ["BS", "WS"]:
                bs_reward = reward
            else:
                bs_reward = self.bpd[(slot_id, target_id, "BS")][-1]['reward']
                
            program_dict = dict(expression=expression, actions=actions,
                                seq_length=ep_length, reward=np.float32(reward), 
                                bs_reward = np.float32(bs_reward),
                                log_prob=log_prob,do_fail=do_fail, cs_fail=cs_fail)
                
            if "SA" in temp_env.language_name:
                command_list = temp_env.program_generator.parser.parse(expression)
                temp_env.program_generator.compiler._compile(command_list)
                csg_commands = sa_convert_hcsg_commands_to_cpu(temp_env.program_generator.compiler.hcsg_program)
                temp_env.program_generator.compiler.reset()
                program_dict["csg_expression"] = csg_commands
            
            program_dict.update({
                "slot_id": slot_id,
                "target_id": target_id,
                "origin": origin
            })
            if self.rl_mode:
                # Only latest
                self.bpd[(slot_id, target_id, origin)]= [program_dict]
            else:
                self.bpd[(slot_id, target_id, origin)].append(
                    program_dict)
                
                self.bpd[(slot_id, target_id, origin)].sort(
                    key=lambda x: x['reward'], reverse=True)
                if self.store_single:
                    self.bpd[(slot_id, target_id, origin)] = self.bpd[(
                        slot_id, target_id, origin)][:1]
                else:
                    if origin == "BS":
                        self.bpd[(slot_id, target_id, origin)] = self.bpd[(
                            slot_id, target_id, origin)][:1]

        rewards = []
        for key, value in self.bpd.items():
            rewards.extend([x['reward'] for x in value])
        if self.rl_mode:
            self.mean_reward = self.mean_reward * self.rl_moving_baseline_alpha +  (1-self.rl_moving_baseline_alpha) * np.nanmean(rewards)
        else:
            self.mean_reward = np.nanmean(rewards)

        # Load weights
    def mark_failure_cases(self, failed_keys, rewriter):
        do_fail, do_fail = False, False
        if isinstance(rewriter, DifferentiableOptimizer):
            do_fail = True
        if isinstance(rewriter, CodeSplicer):
            do_fail = True
        for key in failed_keys:
            self.bpd[key][0]['do_fail'] = do_fail
            self.bpd[key][0]['cs_fail'] = do_fail
    
    def remove_selected_programs(self, remove_origin):
        new_best_progs = defaultdict(list)
        for key, value in self.bpd.items():
            origin = key[2] 
            if origin != remove_origin:
                new_best_progs[key] = value
        
        self.bpd = new_best_progs
        
        
    def log_training_data_details(self, train_state):
        # The percentage of each used for sampling
        seperate_lists = defaultdict(list)

        
        for key, value in self.bpd.items():
            origin = key[2]
            seperate_lists[origin].extend(value)
        
        all_prog_lens = []
        all_prog_rewards = []
        tot_programs = 0
        for key in seperate_lists.keys():
            progs = seperate_lists[key]
            self.logger.record("Training data/%s N Progs" % key, len(progs))
            tot_programs += len(progs)
            cur_lens = [len(x['expression']) for x in progs]
            self.logger.record("Training data/%s AVG length" % key, np.nanmean(cur_lens))
            rewards = [x['reward'] for x in progs]
            self.logger.record("Training data/%s AVG reward" % key, np.nanmean(rewards))
            logprobs = [x['log_prob'] for x in progs]
            self.logger.record("Training data/%s AVG log_prob" % key, np.nanmean(logprobs))
            if not key in ['WS']:
                all_prog_lens.extend(cur_lens)
                all_prog_rewards.extend(rewards)
        
        program_list = self.construct_training_data(noisy_rewriter=None, temp_env=None, train_state=train_state)
        rewards = [x['reward'] for x in program_list if x['origin'] != "WS"]
        lens = [len(x['expression']) for x in program_list]
        n_programs = len(program_list)
        for n_origin in ALL_ORIGINS:
            count = len([x for x in program_list if x['origin'] == n_origin])
            self.logger.record("Training data/Training %s Progs" % n_origin, count)

        self.logger.record("Training data/Training N Progs", n_programs)
        self.logger.record("Training data/Training AVG length", np.nanmean(lens))
        self.logger.record("Training data/Training AVG reward", np.nanmean(rewards))
        
        self.logger.dump(train_state.tensorboard_step)
        
    def construct_training_data(self, noisy_rewriter, temp_env, train_state, tag="BS", remove_generative=False):
        # Convert the 
        if remove_generative:
            gen_sample_rate = 0
        else:
            gen_sample_rate = self.gen_sample_rate
            
        if self.training_data_selection == "BEST":
            program_list = get_program_list(self.bpd, n_prog_per_item=self.best_prog_count)
        elif self.training_data_selection == "BS+BEST":
            program_list = get_program_list(self.bpd, merge_origins=MERGABLE_SANS_BS_TYPES, BS_gated=True, n_prog_per_item=self.best_prog_count)
            
        elif self.training_data_selection == "BS+BEST+PROB-ONE":
            # Set probability as, P(BS) = max(1/N, 0.25)
            
            # P(Others) = softmx(delta_reward) * (1 - P(B))
            # create and return the data list
            program_list = probabilistic_program_list(self.bpd, probability_type=1, n_rewrites_per_item=self.best_prog_count,
                                                      temperature=self.data_sampling_temperature,
                                                      gen_sample_rate=gen_sample_rate, n_samples_per_item=self.n_samples_per_item,
                                                      multiply_input=self.multiply_inputs)
            
        elif self.training_data_selection == "BS+BEST+PROB-TWO":
            # Set probability by simply softmx(reward)
            program_list = probabilistic_program_list(self.bpd, probability_type=2, n_rewrites_per_item=self.best_prog_count,
                                                      temperature=self.data_sampling_temperature,
                                                      gen_sample_rate=gen_sample_rate, n_samples_per_item=self.n_samples_per_item,
                                                      multiply_input=self.multiply_inputs)
            
        elif self.training_data_selection == "BS+BEST+PROB-THREE":
            program_list = probabilistic_program_list(self.bpd, probability_type=3, n_rewrites_per_item=self.best_prog_count,
                                                      temperature=self.data_sampling_temperature,
                                                      gen_sample_rate=gen_sample_rate, n_samples_per_item=self.n_samples_per_item,
                                                      multiply_input=self.multiply_inputs)
            
        # Here you want to create augmentations
        if noisy_rewriter:
            if noisy_rewriter.enable:
                augmented_prog_objs, failed_keys = noisy_rewriter.rewrite_programs(temp_env, program_list, 
                                                                    train_state.tensorboard_step, quantize=True)
                augmented_programs = self.nr_prog_obj_converter(augmented_prog_objs, temp_env)
                
                program_list.extend(augmented_programs)
        
            
        print("Training with %d programs" % len(program_list))
        self.logger.record("Training data/ %s N Training progs" % tag, len(program_list))
        self.logger.dump(train_state.tensorboard_step)
        return program_list

    def nr_prog_obj_converter(self, prog_objs, temp_env):
        # if self.rl_mode:
        program_list = []
        temp_env.program_generator.set_execution_mode(th.device("cuda"), th.float16)
        for obj in prog_objs:
            expression = obj['expression']
            # expression.append("$")
            slot_id = obj['slot_id']
            target_id = obj['target_id']
            reward = obj['reward']
            log_prob = obj['log_prob']
            origin = obj['origin']
            do_fail = obj['do_fail']
            cs_fail = obj['cs_fail']
            # ep_length = all_metrics['ep_length'][idx]
            main_actions = temp_env.action_space.expression_to_action(
                expression)
            # expression_t = temp_env.action_space.action_to_expression(main_actions)
            ep_length = main_actions.shape[0]
            actions = np.zeros(self.max_length, dtype=np.int64)
            if ep_length > self.max_length:
                continue
            actions[:ep_length] = main_actions
            # execute and load?

            if origin == "BS":
                bs_reward = reward
            else:
                bs_reward = self.bpd[(slot_id, target_id, "BS")][-1]['reward']
                
            program_dict = dict(expression=expression, actions=actions,
                                seq_length=ep_length, reward=np.float32(reward), 
                                bs_reward = np.float32(bs_reward),
                                log_prob=log_prob,do_fail=do_fail, cs_fail=cs_fail)
                
            if "SA" in temp_env.language_name:
                command_list = temp_env.program_generator.parser.parse(expression)
                temp_env.program_generator.compiler._compile(command_list)
                csg_commands = sa_convert_hcsg_commands_to_cpu(temp_env.program_generator.compiler.hcsg_program)
                temp_env.program_generator.compiler.reset()
                program_dict["csg_expression"] = csg_commands
            
            program_dict.update({
                "slot_id": slot_id,
                "target_id": target_id,
                "origin": origin
            })
            program_list.append(program_dict)
        return program_list