from collections import defaultdict
from email.policy import default
import time
import torch as th
from .code_splice_utils import match_to_cache, get_masked_scores, bool_count, batched_masked_scores
import _pickle as cPickle
import os
import numpy as np
import random
from CSG.env.csg3d.languages import GraphicalMCSG3DCompiler, MCSG3DParser, PCSG3DParser 
from CSG.env.csg2d.languages import GraphicalMCSG2DCompiler, MCSG2DParser, PCSG2DParser 
from CSG.env.csg3d.parser_utils import pcsg_to_ntcsg, ntcsg_to_pcsg
from CSG.utils.profile_utils import profileit
from itertools import cycle
import faiss
import cProfile
import torch
from .code_splice_utils import get_new_command, distill_transform_chains, get_scores
from CSG.env.reward_function import chamfer

MAX_SIZE = int(100000)

def return_zero(*args):
    return 0
class BaselineCache():
    
    def __init__(self, save_dir, cache_config, language_name):
        self.cache = []
        self.search_space = None
        self.max_masking_rate = cache_config.MAX_MASKING_RATE
        self.save_dir = save_dir
        self.cache_size = cache_config.CACHE_SIZE
        self.replacement_idx = random.sample(range(self.cache_size), self.cache_size)
        self.replacement_iterator = cycle(self.replacement_idx)
        self.MS = False
        self.language_name = language_name
        self.ms_remove_keys = ["slot_id", "target_id", "command_ids", "all_commands", "reward", "expression"]
        
    def accept_node(self, node):
        # TODO: Make it configurable
        type_condition = node['type'] in ["B", "M",]
        # iou_condition = node['subexpr_info']['masked_iou'] >= self.min_masked_iou_rate
        overall_condition = type_condition
        return overall_condition
        
    
class FaissIVFCache(BaselineCache):
    
    def __init__(self, save_dir, cache_config, eval_mode, language_name):
        super(FaissIVFCache, self).__init__(save_dir, cache_config, language_name)    
        self.replacement_idx = None
        self.replacement_iterator = None
        # Important parameters: 
        self.merge_nlist = cache_config.MERGE_NLIST
        self.merge_nprobe = cache_config.MERGE_NPROBE
        self.merge_bit_distance = cache_config.MERGE_BIT_DISTANCE
        #
        self.search_nlist = cache_config.SEARCH_NLIST
        self.search_nprobe = cache_config.SEARCH_NPROBE
        self.n_program_to_sample = cache_config.N_PROGRAM_FOR_MERGE
        self.load_previous_subexpr_cache = False
        self.eval_mode = eval_mode
        if eval_mode:
            self.subexpr_load_path = cache_config.SUBEXPR_LOAD_PATH
        
        self.stats = {}
        

    def generate_cache_and_index(self, best_program_dict, temp_env, use_canonical=True):
        
        main_st = time.time()
        if self.eval_mode: 
            subexpr_cache = []
        else:
            subexpr_cache = self.generate_subexpr_cache(best_program_dict, temp_env, use_canonical)
        # cPickle.dump(subexpr_cache, open("step_1.pkl" ,"wb"))
        # subexpr_cache = cPickle.load(open("step_1.pkl" ,"rb"))
        self.stats['n_new_expr'] = len(subexpr_cache)
        if len(subexpr_cache) > MAX_SIZE:
            subexpr_cache = random.sample(subexpr_cache, MAX_SIZE)

        # Setting some primitives
        # if we have subexpr_cache from before, add it here:
        if self.eval_mode:
            all_previous_exprs = self.load_all_exprs(self.subexpr_load_path)
            subexpr_cache.extend(all_previous_exprs)
        else:
            if self.load_previous_subexpr_cache:
                st=time.time()
                all_previous_exprs = self.load_all_exprs()
                if all_previous_exprs:
                    subexpr_cache.extend(all_previous_exprs)
                print("time for loading data", time.time() - st)
        print("Number of subexpressions in cache = %d" % len(subexpr_cache))

        if len(subexpr_cache) > MAX_SIZE:
            # Need better mechanism here
            # Store the ones used more often
            subexpr_cache = random.sample(subexpr_cache, MAX_SIZE)
            
        
        self.stats['n_loaded_expr'] = len(subexpr_cache)
        th.cuda.empty_cache()
        subexpr_cache = self.merge_cache(subexpr_cache)
        subexpr_cache = [x for x in subexpr_cache if isinstance(x['canonical_shape'], th.Tensor)]

        self.stats['n_merged_expr'] = len(subexpr_cache)
        # Other stats:
        # expression volume
        # ratio
        if "CSG_T" in self.language_name:
            command_ratios = defaultdict(return_zero)
            volume_list = []
            for subexpr in subexpr_cache:
                command_syms = [x['symbol'] for x in subexpr['commands']]
                for c_sym in command_syms:
                    command_ratios[c_sym]  += 1
                bbox = 1/subexpr['canonical_commands'][0]['param']
                original_volume = bbox[0] * bbox[0] * bbox[0]
                volume_list.append(original_volume)
                    
            total_tokens = np.sum(list(command_ratios.values()))
            for key, value in command_ratios.items():
                self.stats['%s_ratio' % key] = value/total_tokens
            self.stats["avg_volume"] = np.mean(volume_list)
        elif "SA" in self.language_name:
            pass
        
        
        st=time.time()
        # self.save_all_exprs(subexpr_cache)
        if self.MS:
            # Reduce the expressions
            for subexpr in subexpr_cache:
                for key in self.ms_remove_keys:
                    if key in subexpr.keys():
                        subexpr.pop(key)
        print("time for saving data", time.time() - st)

        node_item = subexpr_cache[0]
        self.tensor_shape = node_item['canonical_shape'].shape
        self.tensor_device = node_item['canonical_shape'].device
        self.cache_d = node_item['canonical_shape'].reshape(-1).shape[0]
        if "CSG" in self.language_name:
            self.empty_item = {'canonical_shape': 0,
                        'commands': [],
                        'canonical_commands': node_item['canonical_commands']
                      }
        else:
            self.empty_item = {'canonical_shape': 0,
                        'subpr_sa_commands': [],
                        "sa_action_length": 192,
                        "subpr_hcsg_commands": [],}
            

        self.centroids, self.invlist_list, self.invlist_lookup_list = self.create_final_index(subexpr_cache)
        lookup_index_list = []
        for idx, invlist in enumerate(self.invlist_list):
            cur_lookup = [(idx, x) for x in range(len(invlist))]
            lookup_index_list.append(cur_lookup)
        self.lookup_index_list = lookup_index_list
        et = time.time()
        print("Overall Time", et - main_st)

        
    def create_final_index(self, selected_subexprs):

        # Now we need to create the final indexes:
        if "CSG" in self.language_name:
            subselected_subexpr = []
            for subexpr in selected_subexprs:
                new_subexpr = dict(commands=subexpr['commands'],
                                   canonical_commands=subexpr['canonical_commands'])
                subselected_subexpr.append(new_subexpr)
        elif "SA" in self.language_name:
            subselected_subexpr = []
            for subexpr in selected_subexprs:
                new_subexpr = dict(subpr_sa_commands=subexpr['subpr_sa_commands'], 
                                   subpr_hcsg_commands=subexpr['subpr_hcsg_commands'],
                                   sa_action_length=subexpr['sa_action_length'])
                subselected_subexpr.append(new_subexpr)
            
        st = time.time()
        cached_expression_shapes = [x['canonical_shape'].reshape(-1) for x in selected_subexprs]
        cached_expression_shapes = th.stack(cached_expression_shapes, 0)
        # cached_expression_shapes.shape
        cached_np = cached_expression_shapes.cpu().data.numpy()
        chached_np_packed = np.packbits(cached_np, axis=-1, bitorder="little")

        self.cache_d = cached_expression_shapes.shape[1]
        merge_nb = cached_expression_shapes.shape[0]
        # Initializing index.
        invl_sizes = [0]
        count = 0
        while(np.min(invl_sizes) <= 0):
            count += 1
            print("Trying to make a non-empty index for the %d th time." % count)
            quantizer = faiss.IndexBinaryFlat(self.cache_d)  # the other index
            index = faiss.IndexBinaryIVF(quantizer, self.cache_d, self.search_nlist)
            index.nprobe = self.search_nprobe
            assert not index.is_trained
            index.train(chached_np_packed)
            index.add(chached_np_packed)
            invl_sizes = np.array([index.invlists.list_size(i) for i in range(self.search_nlist)])
            print("Minimum list size", np.min(invl_sizes))
            if np.min(invl_sizes) <= 0:
                #should update the sizes:
                self.search_nlist = int(self.search_nlist * 0.75)
                print("setting search_nlist to %d" % self.search_nlist)
        # fetch the different centroids, and create the different lists:
        centroid_list = []
        invlist_list = []
        inv_lookup_list = []
        # Now we need to keep it within the index size:
        # Option 1: within each invlist, keep %n random shapes where %n = min(avg(n), l)
        
        cur_cache_size = np.sum(invl_sizes)
        if self.cache_size < cur_cache_size:
            avg_size = self.cache_size / self.search_nlist
            keep_as_is = invl_sizes <= avg_size
            new_size = int(avg_size) # (self.cache_size - np.sum(invl_sizes[keep_as_is])) / (self.search_nlist - np.sum(keep_as_is) + 1e-9)
            # new_size = int(new_size)
        else:
            avg_size = int(self.cache_size / self.search_nlist)
            new_size = np.max(invl_sizes)
            new_size = min(avg_size, new_size)

        for ind in range(self.search_nlist):
            centroid_list.append(quantizer.reconstruct(ind))
            invlist_ids = self.get_invlist_idx(index.invlists, ind)
            # now create the NN-Lookup array:
            invlist_ids = list(invlist_ids)
            if len(invlist_ids) > new_size:
                invlist_ids  = random.sample(invlist_ids, new_size)
                pad = False
            else:
                pad = True
            invlist = [subselected_subexpr[x] for x in invlist_ids]
            shapelist = [selected_subexprs[x] for x in invlist_ids]
            cached_expression_shapes = [x['canonical_shape'].reshape(-1) for x in shapelist]
            # How to fix if this is empty?

            inv_lookup = th.stack(cached_expression_shapes, 0)
            if pad:
                pad_size = new_size - len(invlist_ids)
                pad_invlist = [self.empty_item.copy() for i in range(pad_size)]
                invlist.extend(pad_invlist)
                pad_invlookup = torch.zeros((1, self.cache_d), dtype=th.bool).to(self.tensor_device)
                pad_invlookup = pad_invlookup.expand(pad_size, -1)
                inv_lookup = torch.cat([inv_lookup, pad_invlookup], 0)
            invlist_list.append(invlist)
            inv_lookup_list.append(inv_lookup)
        
        centroid_list = np.stack(centroid_list, 0)
        centroids = self.packed_np_to_tensor(centroid_list)
        et = time.time()
        print("Final Index Creation Time", et - st)

        return centroids, invlist_list, inv_lookup_list

    def get_invlist_idx(self, invlists, ind):
        ls = invlists.list_size(ind)
        list_ids = np.zeros(ls, dtype='int64')
        x = invlists.get_ids(ind)
        faiss.memcpy(faiss.swig_ptr(list_ids), x, list_ids.nbytes)
        invlists.release_ids(ind, x)
        return list_ids

    def packed_np_to_tensor(self, array):
        unpacked_np = np.unpackbits(array, axis=-1, bitorder="little")
        tensor = torch.from_numpy(unpacked_np)
        # tensor = tensor.reshape(tensor.shape[0], self.tensor_shape[0], self.tensor_shape[1], self.tensor_shape[2])

        tensor = tensor.to(self.tensor_device)
        return tensor

    def generate_subexpr_cache(self, best_program_dict, temp_env, use_canonical=True):
        print("Collecting Subexpressions")
        # Check it to the new ones:
        # Replace with temp_env.parser
        base_parser = temp_env.program_generator.parser
        if "3D" in temp_env.language_name:
            graph_compiler = GraphicalMCSG3DCompiler(resolution=temp_env.program_generator.compiler.resolution,
                                        scale=temp_env.program_generator.compiler.scale,
                                        draw_mode=temp_env.program_generator.compiler.draw.mode)
        else:
            graph_compiler = GraphicalMCSG2DCompiler(resolution=temp_env.program_generator.compiler.resolution,
                                        scale=temp_env.program_generator.compiler.scale,
                                        draw_mode=temp_env.program_generator.compiler.draw.mode)

        graph_compiler.set_to_cuda()
        graph_compiler.set_to_half()
        graph_compiler.reset()   

        base_parser.set_device("cuda")
        base_parser.set_tensor_type(th.float32)             

        counter = 0
        st = time.time()
        subexpr_cache = []
        # random sample 10k expressions:
        keys = list(best_program_dict.keys())
        rand_keys = random.sample(keys, min(len(keys), self.n_program_to_sample))

        for iteration, key in enumerate(rand_keys):
            value = best_program_dict[key][:1]
            if iteration % 500 == 0:
                print("cur iteration %d. Cur Time %f" % (iteration, time.time() - st))
                if len(subexpr_cache) > MAX_SIZE:
                    print("subexpr cache is full", len(subexpr_cache), ". Sampling from it")
                    subexpr_cache = random.sample(subexpr_cache, MAX_SIZE)
            
            for cur_value in value:
                expression = cur_value['expression']
                with th.no_grad():
                    # with th.cuda.amp.autocast():
                        command_list = base_parser.parse(expression)
                        graph = graph_compiler.command_tree(command_list, None, enable_subexpr_targets=False, add_splicing_info=True)
                graph_nodes = [graph.nodes[i] for i in graph.nodes]
                for node in graph_nodes:
                    if iteration == 1:
                        # This is to include the leaf commands
                        add = node['type'] in ["B", "D", "M", "RD"]
                    else:
                        add = self.accept_node(node)
                    if add:
                        if use_canonical:
                            shape = node['subexpr_info']['canonical_shape']
                        else:
                            shape = node['subexpr_info']['expr_shape']
                        node_item = {'canonical_shape': shape,
                                        'commands': node['subexpr_info']['commands'],
                                        'canonical_commands': node['subexpr_info']['canonical_commands']}
                        subexpr_cache.append(node_item)

                        counter += 1
        et = time.time()
        print("Subexpr Discovery Time", et - st)
        print("found %d sub-expressions" % counter)
        return subexpr_cache

    def merge_cache(self, subexpr_cache):
        # Now from this cache create unique:
        
        avg_length = np.mean([len(x['commands']) for x in subexpr_cache])
        print("Starting merge with  %d sub-expressions with avg. length %f" % (len(subexpr_cache), avg_length))
        
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
                cur_len = len(expr['commands'])
                if cur_len < min_len:
                    min_len = cur_len
                    min_ind = ind
            
            selected_subexprs.append(sel_exprs[min_ind])
            for ind in selected_ids:
                if ind in all_indexes:
                    all_indexes.remove(ind)

        avg_length = np.mean([len(x['commands']) for x in selected_subexprs])
        print("found %d unique sub-expressions with avg. length %f" % (len(selected_subexprs), avg_length))
        et = time.time()
        print("Merge Process Time", et - st)

        return selected_subexprs

    def get_candidates(self, targets, masks, node_ids, k, min_per_probe=0):
        candidate_list = []
        

        # find masked iou with centroids
        if len(self.centroids.shape) == 2:
            self.centroids = self.centroids.unsqueeze(0)
        # Given the targets, find the k nearest for each target:
        targets = targets.unsqueeze(1)
        masks = masks.unsqueeze(1)

        scores = batched_masked_scores(self.centroids, targets, masks)
        maxes, arg_maxes = th.topk(scores, dim=1, k=self.search_nprobe)
        
        arg_maxes = arg_maxes.cpu().numpy()
        # each lookup will return k, we can create a list of k
        all_lookup_lists = []
        all_scores = []
        
        for ind, node_id in enumerate(node_ids):
            cur_target = targets[ind, :]
            cur_mask = masks[ind, :]
            sel_invlist_index = list(arg_maxes[ind])
            
            lookup_lists = []
            for idx in sel_invlist_index:
                lookup_lists.extend(self.lookup_index_list[idx])
            all_lookup_lists.append(lookup_lists)
            
            inv_lookups = [self.invlist_lookup_list[idx] for idx in sel_invlist_index]
            inv_lookups = torch.cat(inv_lookups, 0)
            scores = batched_masked_scores(inv_lookups, cur_target, cur_mask)
            all_scores.append(scores)
            
        all_scores = torch.stack(all_scores, 0)
        cur_maxes, cur_arg_maxes = th.topk(all_scores, dim=1, k=k)
        cur_arg_maxes = list(cur_arg_maxes.cpu().numpy())
        
        for ind, node_id in enumerate(node_ids):
            node_arg_maxes = cur_arg_maxes[ind]
            node_lookup_lists = all_lookup_lists[ind]
            node_cur_maxes = cur_maxes[ind]
            for idx, sel_ind in enumerate(node_arg_maxes):
                sel_indexes = node_lookup_lists[sel_ind]
                inv_list_id = sel_indexes[0]# sel_invlist_index[nprobe_id]
                list_position = sel_indexes[1]# sel_ind % invlist_size
                sel_item = self.invlist_list[inv_list_id][list_position]
                candidate = self.get_candidate_format(node_id, idx, node_cur_maxes, sel_item)
                candidate_list.append(candidate)
        return candidate_list

    def get_candidate_format(self, node_id, idx, node_cur_maxes, sel_item):
        candidate = {'node_id': node_id, 
                            'masked_iou': node_cur_maxes[idx],
                            'bool_count': bool_count(sel_item),
                            'commands': sel_item['commands'], 
                            'canonical_commands': sel_item['canonical_commands']}
                    
        return candidate

    
    def save_all_exprs(self, subexpr_cache):
        file_name = self.save_dir + "/all_subexpr.pkl"
        print("Saving all expressions at %s" % file_name)
        cPickle.dump(subexpr_cache, open(file_name, "wb"))
        
    def load_all_exprs(self, file_name=None):
        if file_name is None:
            file_name = self.save_dir + "/all_subexpr.pkl"
        print("Loading all expressions at %s" % file_name)
        if os.path.exists(file_name):   
            subexpr_cache = cPickle.load(open(file_name, "rb"))
            return subexpr_cache
        else:
            print("file %s does not exist. No data Loading" % file_name)
            return None

    def save_data(self, clear_memory=True):
        file_name = self.save_dir + "/subexpr_cache.pkl"
        data = [item for sublist in self.invlist_list for item in sublist]
        data = [x for x in data if isinstance(x['canonical_shape'], th.Tensor)]
        cPickle.dump(data, open(file_name, "wb"))
        if clear_memory:
            del self.centroids, self.invlist_list, self.invlist_lookup_list
        self.load_previous_subexpr_cache = True
    
    def load_data(self,):
        file_name = self.save_dir + "/subexpr_cache.pkl"
        if os.path.exists(file_name):    
            self.previous_subexpr_cache = cPickle.load(open(file_name, "rb"))
            self.load_previous_subexpr_cache = True
            print("loaded Previous data with %d expressions!" % len(self.previous_subexpr_cache))
            
        else:
            print("file %s does not exist. No data Loading" % file_name)


class MergeSplicerCache(FaissIVFCache):
    """
    When merging different programs, replace them in some of the other programs as well.
    """
    
    def __init__(self, save_dir, cache_config, merge_splice_config, eval_mode, language_name):
        super(MergeSplicerCache, self).__init__(save_dir, cache_config, eval_mode, language_name)    

        if merge_splice_config.INPUT_GATING:
            self.valid_input_origins = ["BS"]
        else:
        # self.valid_input_origins = ["BS"]
            self.valid_input_origins = ["BS", "DO", "GS", "CS"] # This is for MS
        self.min_delta = merge_splice_config.MIN_DELTA
        self.merge_splice_sample_count = merge_splice_config.SAMPLE_COUNT
        self.MS = True

    def generate_subexpr_cache(self, best_program_dict, temp_env, use_canonical=True):
        print("Collecting Subexpressions")
        # Check it to the new ones:
        # Replace with temp_env.parser
        base_parser = temp_env.program_generator.parser
        if "3D" in temp_env.language_name:
            graph_compiler = GraphicalMCSG3DCompiler(resolution=temp_env.program_generator.compiler.resolution,
                                        scale=temp_env.program_generator.compiler.scale,
                                        draw_mode=temp_env.program_generator.compiler.draw.mode)
        else:
            graph_compiler = GraphicalMCSG2DCompiler(resolution=temp_env.program_generator.compiler.resolution,
                                        scale=temp_env.program_generator.compiler.scale,
                                        draw_mode=temp_env.program_generator.compiler.draw.mode)

        graph_compiler.set_to_cuda()
        graph_compiler.set_to_half()
        graph_compiler.reset()   

        base_parser.set_device("cuda")
        base_parser.set_tensor_type(th.float32)             

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
                with th.no_grad():
                    # with th.cuda.amp.autocast():
                        command_list = base_parser.parse(expression)
                        graph = graph_compiler.command_tree(command_list, None, enable_subexpr_targets=False, add_splicing_info=True)
                graph_nodes = [graph.nodes[i] for i in graph.nodes]
                for node in graph_nodes:
                    if iteration == 1:
                        # This is to include the leaf commands
                        add = node['type'] in ["B", "D", "M", "RD"]
                    else:
                        add = self.accept_node(node)
                    if add:
                        if use_canonical:
                            shape = node['subexpr_info']['canonical_shape']
                        else:
                            shape = node['subexpr_info']['expr_shape']
                        node_item = {'canonical_shape': shape,
                                        'commands': node['subexpr_info']['commands'],
                                        'canonical_commands': node['subexpr_info']['canonical_commands'],
                                        'slot_id': cur_value['slot_id'],
                                        'target_id': cur_value['target_id'],
                                        'command_ids': node['subexpr_info']['command_ids'],
                                        'all_commands': command_list, 
                                        'reward': cur_value['reward'],
                                        'expression': expression}
                        subexpr_cache.append(node_item)

                        counter += 1
        et = time.time()
        print("Subexpr Discovery Time", et - st)
        print("found %d sub-expressions" % counter)
        return subexpr_cache

    def merge_cache(self, subexpr_cache):
        # Now from this cache create unique:
        merge_spliced_commands = []

        avg_length = np.mean([len(x['commands']) for x in subexpr_cache])
        print("Starting merge with  %d sub-expressions with avg. length %f" % (len(subexpr_cache), avg_length))
        
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
                cur_len = len(expr['commands'])
                if cur_len < min_len:
                    min_len = cur_len
                    min_ind = ind
            # Now for the rest create a new expression with the replacement
            expr_to_splice = sel_exprs[min_ind]
            for ind, expr in enumerate(sel_exprs):
                if ind != min_ind and "all_commands" in expr.keys():
                    if (min_len + self.min_delta) < len(expr['commands']):
                        old_command_list = expr['all_commands']
                        old_command_ids = expr['command_ids']
                        old_canonical_commands = expr['canonical_commands']
                        old_reward = expr['reward']
                        old_len = len(expr['expression'])
                        new_command_list = get_new_command(old_command_list, old_command_ids, old_canonical_commands,
                                        expr_to_splice, use_canonical=True)
                        merge_spliced_commands.append([new_command_list, expr['slot_id'], expr['target_id'],
                                                       old_reward, old_len])

            compressed_expr = {x:y for x, y in expr_to_splice.items()}
            # {'canonical_shape': expr_to_splice['canonical_shape'],
            #             'commands': expr_to_splice['commands'],
            #             'canonical_commands': expr_to_splice['canonical_commands']}
            selected_subexprs.append(compressed_expr)
            for ind in selected_ids:
                if ind in all_indexes:
                    all_indexes.remove(ind)

        avg_length = np.mean([len(x['commands']) for x in selected_subexprs])
        print("found %d unique sub-expressions with avg. length %f" % (len(selected_subexprs), avg_length))
        et = time.time()
        print("Merge Process Time", et - st)
        self.merge_spliced_commands = merge_spliced_commands
        return selected_subexprs


    def get_merge_spliced_expressions(self, temp_env, higher_language, quantize_expr=True, logger=None, tensorboard_step=0,
                                     reward_threshold=0.005, length_alpha=0):


        # action_space = temp_env.action_space
        # zero_actions = np.zeros(temp_env.perm_max_len)
        # action_list = []
        # action_lens = []
        st = time.time()

        resolution = temp_env.action_space.resolution

        device, dtype = th.device("cuda"), th.float16
        temp_env.program_generator.set_execution_mode(th.device("cuda"), dtype)
        base_parser = temp_env.program_generator.parser
        base_compiler = temp_env.program_generator.compiler

        updated_reward = []
        previous_reward = []

        previous_length = []
        updated_length = []

        prog_objs = []
        for item in self.merge_spliced_commands:
            new_command_list = item[0]
            cur_slot = item[1]
            cur_target = item[2]
            original_reward = item[3]
            original_len = item[4]
            if higher_language:
                if '3D' in temp_env.language_name:
                    new_command_list = distill_transform_chains(new_command_list, device, dtype)
                else:
                    new_command_list = distill_transform_chains(new_command_list, device, dtype, mode="2D")
                complexity = base_compiler._get_complexity(new_command_list)
                if complexity > temp_env.max_expression_complexity:
                    # reject
                    continue
            base_expr = base_parser.get_expression(new_command_list, clip=True,
                                                quantize=quantize_expr, resolution=resolution)

            temp_command_list = base_parser.parse(base_expr)
            base_compiler._compile(temp_command_list)
            output = base_compiler._output.detach().clone()
            output = (output <= 0)
            
            target_np, _ = temp_env.program_generator.get_executed_program(cur_slot, cur_target)
            target = th.from_numpy(target_np).cuda().bool()
            if '3D' in temp_env.language_name:
                new_reward = get_scores(output, target).item()
                # Create a function to take in points and get the output. 
            else:
                output_np = output.data.cpu().numpy()
                new_reward = 100 - chamfer(output_np[None, :, :], target_np[None, :, :])[0]
                
            new_reward = new_reward + length_alpha * len(base_expr)
            if new_reward >= (original_reward - reward_threshold):

                prog_obj = dict(expression=base_expr.copy(),
                                slot_id=cur_slot,
                                target_id=cur_target,
                                reward=new_reward,
                                origin="CS",
                                do_fail=False,
                                cs_fail=False,
                                log_prob=0)
                prog_objs.append(prog_obj)
                updated_reward.append(new_reward)
                updated_length.append(len(base_expr))
            else:
                updated_reward.append(original_reward)
                updated_length.append(original_len)
            previous_reward.append(original_reward)
            previous_length.append(original_len)


        if len(prog_objs) > self.merge_splice_sample_count:
            prog_objs.sort(key=lambda x: x['reward'], reverse=True)
            prog_objs = prog_objs[:self.merge_splice_sample_count]
            



        et = time.time()
        process_time = et - st
        self.log_merge_splice_info(logger, tensorboard_step, updated_length, previous_length, 
                                   updated_reward, previous_reward, process_time, prog_objs)
        self.merge_spliced_commands = None



        return prog_objs


    def log_merge_splice_info(self, logger, tensorboard_step, updated_length, previous_length, 
                              updated_reward, previous_reward, process_time, all_prog_objs):

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
        logger.record("search/MS_new_progs",
                           len(all_prog_objs))
        logger.record("Training data/MS time", process_time)
        logger.dump(tensorboard_step)