import copy
from .subexpr_cache import FaissIVFCache
from CSG.env.shape_assembly.graph_compiler import GraphicalSACompiler
import time
import torch as th
import faiss
import random
import numpy as np
from .code_splice_utils import get_scores
from .sa_utils import sa_convert_hcsg_commands_to_cpu, get_master_program_struct

class SAFaissIVFCache(FaissIVFCache):

    def __init__(self, save_dir, cache_config, merge_splice_config, eval_mode, language_name):
        super(SAFaissIVFCache, self).__init__(save_dir, cache_config, eval_mode, language_name)    
        self.ms_remove_keys = ["slot_id", "target_id", "reward", "cube_name", 
                               "total_expression_length", "master_program_struct"]
        self.min_delta = 1
        
        if merge_splice_config:

            if merge_splice_config.INPUT_GATING:
                self.valid_input_origins = ["BS"]
            else:
            # self.valid_input_origins = ["BS"]
                self.valid_input_origins = ["BS", "DO", "GS", "CS"] # This is for MS
            self.min_delta = merge_splice_config.MIN_DELTA
        else:
            self.min_delta = 0
            self.valid_input_origins = ["BS"]
        self.MS = True

    def accept_node(self, node):
        # TODO: Make it configurable
        type_condition = node['type'] in ["D"]
        if type_condition:
            has_subpr = node['has_subprogram']
        # iou_condition = node['subexpr_info']['masked_iou'] >= self.min_masked_iou_rate
        overall_condition = type_condition and has_subpr
        return overall_condition
        

    def generate_subexpr_cache(self, best_program_dict, temp_env, use_canonical=True):
        print("Collecting Subexpressions")
        # Check it to the new ones:
        # Replace with temp_env.parser
        base_parser = temp_env.program_generator.parser
        compiler = temp_env.program_generator.compiler
        graph_compiler = GraphicalSACompiler(resolution=temp_env.program_generator.compiler.resolution,
                                    scale=temp_env.program_generator.compiler.scale,
                                    draw_mode=temp_env.program_generator.compiler.draw.mode)

        graph_compiler.set_to_cuda()
        graph_compiler.set_to_half()
        graph_compiler.reset()   
        compiler.set_to_cuda()
        compiler.set_to_half()
        compiler.reset()   

        base_parser.set_device("cuda")
        base_parser.set_tensor_type(th.float16)             

        max_subexpr_cuboid_count = temp_env.state_machine.sub_max_prim
        counter = 0
        st = time.time()
        subexpr_cache = []
        # random sample 10k expressions:
        keys = list(best_program_dict.keys())
        keys = [x for x in keys if x[2] in self.valid_input_origins]

        rand_keys = random.sample(keys, min(len(keys), self.n_program_to_sample))

        for iteration, key in enumerate(rand_keys):
            value = best_program_dict[key][:1]
            if iteration % 500 == 0:
                print("cur iteration %d. Cur Time %f" % (iteration, time.time() - st))

            for cur_value in value:
                expression = cur_value['expression']
                total_expression_length = temp_env.action_space.expression_to_action(expression).shape[0]
                with th.no_grad():
                    # with th.cuda.amp.autocast():
                        master_program_struct = get_master_program_struct(temp_env, base_parser, compiler, graph_compiler, expression)
                # Collect all the subprograms:
                canonical_reprs = {}
                for name, cube_struct in master_program_struct['cube_dict'].items():
                    if "canonical_shape" in cube_struct.keys():
                        canonical_reprs[name] = cube_struct.pop("canonical_shape")

                for name, cube_struct in master_program_struct['cube_dict'].items():
                    if cube_struct["has_subprogram"]:
                        node_item = {
                            'canonical_shape': canonical_reprs[name],
                            'subpr_sa_commands': cube_struct['subpr_sa_commands'],
                            'subpr_hcsg_commands': cube_struct['subpr_hcsg_commands'],
                            'sa_action_length': cube_struct["sa_action_length"],
                        }
                    
                        if self.MS:
                            more_items = {
                                'slot_id': cur_value['slot_id'],
                                'target_id': cur_value['target_id'],
                                'reward': cur_value['reward'],
                                'cube_name': name,
                                'total_expression_length': total_expression_length,
                                "master_program_struct": master_program_struct,
                            }
                            node_item.update(more_items)

                        subexpr_cache.append(node_item)

                # Also add the parent expression:
                subprogram_dict = base_parser.get_all_subprograms(expression)
                master_program = subprogram_dict[0]
                subpr = master_program# ['subpr_sa_commands']# .copy()
                subpr[0] = "bbox = cuboid(1.0, 1.0, 1.0, 0)"
                cuboid_count = len([x for x in subpr if 'cuboid(' in x]) - 1
                if cuboid_count <= max_subexpr_cuboid_count:
                    sub_command_list = base_parser.parse(subpr)
                    compiler._compile(sub_command_list)
                    output_shape = (compiler._output <0)
                    hcsg_commands = sa_convert_hcsg_commands_to_cpu(compiler.hcsg_program)
                    action_len = temp_env.action_space.expression_to_action(subpr).shape[0]
                    master_item = {
                        'canonical_shape': output_shape.clone(),
                        'subpr_sa_commands': subpr,
                        'subpr_hcsg_commands': hcsg_commands,
                        'sa_action_length': action_len,
                    }
                    subexpr_cache.append(master_item)



                counter += 1
        et = time.time()
        print("Subexpr Discovery Time", et - st)
        print("found %d sub-expressions" % counter)
        return subexpr_cache


    def merge_cache(self, subexpr_cache):
        # Now from this cache create unique:
        merge_spliced_commands = []

        avg_length = np.mean([x['sa_action_length'] for x in subexpr_cache])
        print("Starting merge with  %d sub-expressions with avg. action length %f" % (len(subexpr_cache), avg_length))
        
        st = time.time()
        cached_expression_shapes = [x['canonical_shape'].reshape(-1) for x in subexpr_cache]
        cached_expression_shapes = th.stack(cached_expression_shapes, 0)
        # cached_expression_shapes.shape
        cached_np = cached_expression_shapes.cpu().data.numpy()
        chached_np_packed = np.packbits(cached_np,axis=-1,bitorder="little")

        self.cache_d = cached_expression_shapes.shape[1]
        merge_nb = cached_expression_shapes.shape[0]
        # Initializing index.
        quantizer = faiss.IndexBinaryFlat(self.cache_d)  # the other index

        index = faiss.IndexBinaryIVF(quantizer, self.cache_d, self.merge_nlist)
        assert not index.is_trained
        index.train(chached_np_packed)
        assert index.is_trained
        index.add(chached_np_packed)
        index.nprobe = self.merge_nprobe
        lims, D, I = index.range_search(chached_np_packed, self.merge_bit_distance)
        lims_shifted = np.zeros(lims.shape)
        lims_shifted[1:] = lims[:-1]
        
        all_indexes = set(list(range(merge_nb)))
        counter = 0
        selected_subexprs = []
        while(len(all_indexes)> 0):
            cur_ind = next(iter(all_indexes))
            sel_lims = (lims[cur_ind], lims[cur_ind+1])
            selected_ids = I[sel_lims[0]:sel_lims[1]]
            sel_exprs = [subexpr_cache[x] for x in selected_ids]
            min_len = np.inf
            for ind, expr in enumerate(sel_exprs):
                cur_len = expr['sa_action_length']
                if cur_len < min_len:
                    min_len = cur_len
                    min_ind = ind
            # Now for the rest create a new expression with the replacement
            expr_to_splice = sel_exprs[min_ind]
            for ind, expr in enumerate(sel_exprs):
                if ind != min_ind:
                    if (min_len + self.min_delta) < expr['sa_action_length'] and "cube_name" in expr.keys():

                        # create new expression
                        replacement_sa_program = expr_to_splice["subpr_sa_commands"]
                        replacement_hcsg_commands = expr_to_splice['subpr_hcsg_commands']
                        cube_name = expr['cube_name']
                        new_master_struct = copy.deepcopy(expr["master_program_struct"])
                        new_master_struct['cube_dict'][cube_name]["subpr_sa_commands"] = replacement_sa_program.copy()
                        new_master_struct['cube_dict'][cube_name]["subpr_hcsg_commands"] = replacement_hcsg_commands.copy()
                        merge_spliced_commands.append([new_master_struct, expr['slot_id'], expr['target_id'],
                                                       expr['reward'], expr['total_expression_length']])

            compressed_expr = {x:y for x, y in expr_to_splice.items()}
            selected_subexprs.append(compressed_expr)
            for ind in selected_ids:
                if ind in all_indexes:
                    all_indexes.remove(ind)

        avg_length = np.mean([x['sa_action_length'] for x in subexpr_cache])
        print("found %d unique sub-expressions with avg. length %f" % (len(selected_subexprs), avg_length))
        et = time.time()
        print("Merge Process Time", et - st)
        self.merge_spliced_commands = merge_spliced_commands
        return selected_subexprs


    def get_merge_spliced_expressions(self, temp_env, higher_language=None, quantize_expr=True, logger=None, tensorboard_step=0,
                                     reward_threshold=0.005, length_alpha=0):

        st = time.time()

        resolution = temp_env.action_space.resolution

        device, dtype = th.device("cuda"), th.float16
        temp_env.program_generator.set_execution_mode(th.device("cuda"), dtype)
        base_parser = temp_env.program_generator.parser
        base_compiler = temp_env.program_generator.compiler

        graph_compiler = GraphicalSACompiler(resolution=temp_env.program_generator.compiler.resolution,
                                    scale=temp_env.program_generator.compiler.scale,
                                    draw_mode=temp_env.program_generator.compiler.draw.mode)

        graph_compiler.set_to_cuda()
        graph_compiler.set_to_half()
        graph_compiler.reset()   
        base_compiler.set_to_cuda()
        base_compiler.set_to_half()
        base_compiler.reset()   

        updated_reward = []
        previous_reward = []

        previous_length = []
        updated_length = []

        prog_objs = []
        for item in self.merge_spliced_commands:
            new_master_struct = item[0]
            cur_slot = item[1]
            cur_target = item[2]
            original_reward = item[3]
            original_len = item[4]

            new_sa_expression, new_hcsg_commands = graph_compiler.master_struct_to_sa_and_hcsg(new_master_struct)
            new_command_list = base_parser.parse(new_sa_expression)

            complexity = base_compiler._get_complexity(new_command_list)
            if complexity > temp_env.max_expression_complexity:
                # reject
                continue
            # base_compiler._compile(new_command_list)
            base_compiler._csg_compile(new_hcsg_commands)
            output = base_compiler._output.detach().clone()
            output = (output <= 0)
            target_np, _ = temp_env.program_generator.get_executed_program(cur_slot, cur_target)
            target = th.from_numpy(target_np).cuda().bool()
            new_reward = get_scores(output, target).item()
            new_reward = new_reward + length_alpha * len(new_sa_expression)

            if new_reward >= (original_reward - reward_threshold):

                prog_obj = dict(expression=new_sa_expression.copy(),
                                slot_id=cur_slot,
                                target_id=cur_target,
                                reward=new_reward,
                                origin="CS",
                                do_fail=False,
                                cs_fail=False,
                                log_prob=0)
                prog_objs.append(prog_obj)
                updated_reward.append(new_reward)
                action_len = temp_env.action_space.expression_to_action(new_sa_expression).shape[0]
                updated_length.append(action_len)
            else:
                updated_reward.append(original_reward)
                updated_length.append(original_len)
            previous_reward.append(original_reward)
            previous_length.append(original_len)

        et = time.time()
        process_time = et - st
        self.log_merge_splice_info(logger, tensorboard_step, updated_length, previous_length, 
                                   updated_reward, previous_reward, process_time)
        self.merge_spliced_commands = None
        return prog_objs


    def log_merge_splice_info(self, logger, tensorboard_step, updated_length, previous_length, 
                              updated_reward, previous_reward, process_time):

        previous_length = np.array(previous_length)
        updated_length = np.array(updated_length)
        length_delta = previous_length - updated_length
        increment_counter = length_delta > 0
        pc_improvement = length_delta / (previous_length + 1e-9)
        pc_improvement = pc_improvement[increment_counter]

        logger.record("search/MS_updated Length",
                           np.nanmean(updated_length))
        logger.record("search/MS_previous Length",
                           np.nanmean(previous_length))
        logger.record("search/MS_updated reward",
                           np.nanmean(updated_reward))
        logger.record("search/MS_previous reward",
                           np.nanmean(previous_reward))
        logger.record("search/MS Avg Length Delta",
                           np.sum(length_delta)/(np.sum(increment_counter) + 1e-9))
        logger.record("search/MS_pc_improvement",
                           np.nanmedian(pc_improvement))
        logger.record("search/MS_pc_programs_improved",
                           np.nanmean(increment_counter))
        logger.record("search/MS_new_programs",
                           np.sum(increment_counter))
        logger.record("search/MS_tot_programs",
                           previous_length.shape[0])
        logger.record("Training data/MS time", process_time)
        logger.dump(tensorboard_step)

    def get_candidate_format(self, cube_name, idx, node_cur_maxes, sel_item):
        candidate = {'cube_name': cube_name, 
                    'masked_iou': node_cur_maxes[idx],
                    'subpr_sa_commands': sel_item['subpr_sa_commands'], 
                    'sa_action_length': sel_item['sa_action_length'], 
                    'subpr_hcsg_commands': sel_item['subpr_hcsg_commands']}     
        return candidate