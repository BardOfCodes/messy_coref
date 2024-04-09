import networkx as nx
import torch as th
import numpy as np
from .compiler import MCSG3DCompiler
from .compiler_utils import get_reward
import cc3d

class GraphicalMCSG3DCompiler(MCSG3DCompiler):
    """For doing the things that are related to program tree construction.
    Program Tree is innately tied to the compile process. 
    """
    def __init__(self, *args, **kwargs):
        self.denoising_filter = None
        super(GraphicalMCSG3DCompiler, self).__init__(*args, **kwargs)
        # For Tree Crops
        self.threshold = 0.001
        self.threshold_diff = 0.005
        self.mode = "3D"
        
        self.denoise = True
        filter_size = 5
        self.denoisizing_threshold = 8
        self.connected_components = True
        self.cc_limit = 100
        self.denoising_filter = th.ones([1, 1, filter_size, filter_size, filter_size],
                                        device=self.draw.device, dtype=th.float32)

    def set_to_half(self):
        super(GraphicalMCSG3DCompiler, self).set_to_half()
        if not self.denoising_filter is None:
            self.denoising_filter = self.denoising_filter.half()

    def set_to_full(self):
        super(GraphicalMCSG3DCompiler, self).set_to_full()
        if not self.denoising_filter is None:
            self.denoising_filter = self.denoising_filter.float()
    
    def set_to_cuda(self):
        super(GraphicalMCSG3DCompiler, self).set_to_cuda()
        if not self.denoising_filter is None:
            self.denoising_filter = self.denoising_filter.to(self.device)
        
    def set_to_cpu(self):
        super(GraphicalMCSG3DCompiler, self).set_to_cpu()
        if not self.denoising_filter is None:
            self.denoising_filter = self.denoising_filter.to(self.device)
        
    def command_tree(self, command_list, target=None, reset=True, 
                              enable_subexpr_targets=False, add_sweeping_info=False, add_splicing_info=False):
        # Generate the rewards if target is not none
        # Rewards are generated at Boolean and draw operations
        # For transforms
        #  Annotate Graph with Subexpr info

        if reset:
            self.reset()

        graph = nx.DiGraph()
        counter = 0
        default_info = dict(type="ROOT", symbol="START", parent=None, children=[])
        if add_splicing_info:
            default_info["subexpr_info"] = dict()
        graph.add_node(counter, node_id=counter, **default_info)
        cur_parent_ind = counter
        self.non_terminal_expr_start_idx = []


        for ind, command in enumerate(command_list):
            c_type = command['type']
            c_symbol = command['symbol']
            counter +=1
            if c_type == "B":
                self.apply_boolean(boolean_command=self.boolean_to_execute[c_symbol], 
                                   add_splicing_info=add_splicing_info)
                self.draw_node(graph, cur_parent_ind, counter, command)
                cur_parent_ind = counter
                self.non_terminal_expr_start_idx.append(ind)
                
            elif c_type == "T":
                # print("creating Transform", command)
                self.apply_transform(transform_command=self.transform_to_execute[c_symbol], param=command['param'])
                self.draw_node(graph, cur_parent_ind, counter, command)
                cur_parent_ind = counter
                
            elif c_type == "D":
                # print("creating Draw", command)
                self.apply_draw(draw_command=self.draw_to_execute[c_symbol])
                # TODO: Whats the right choice here?
                self.draw_node(graph, cur_parent_ind, counter, command)
                cur_parent_ind = self.get_next_parent_after_draw(graph, cur_parent_ind)
                # Subexpr Stuff:
                if add_splicing_info:
                    cur_expr_canvas = self.draw.base_coords.clone()
                    new_canvas = self.draw_to_execute[c_symbol](coords=cur_expr_canvas)
                    self.update_subexpr_info(graph, new_canvas, ind, ind + 1, command_list, counter)
                elif add_sweeping_info:
                    new_canvas_set = self.canvas_stack[-1]
                    reward = self.get_reward(new_canvas_set, target)
                    new_canvas = new_canvas_set[0]
                    graph.nodes[counter]['reward'] = reward
                    graph.nodes[counter]['validty'] = None
                    graph.nodes[counter]['expression_length'] = 1
                    
            elif c_type == "RD":
                # print("creating Draw", command)
                draw_symbol = c_symbol.split("_")[1]
                self.apply_rotate_draw(draw_command=self.draw_to_execute[draw_symbol], param=command['param'])
                # TODO: Whats the right choice here?
                self.draw_node(graph, cur_parent_ind, counter, command)
                cur_parent_ind = self.get_next_parent_after_draw(graph, cur_parent_ind)
                # Subexpr Stuff:
                if add_splicing_info:
                    cur_expr_canvas = self.draw.base_coords.clone()
                    new_canvas = self.draw_to_execute[draw_symbol](coords=cur_expr_canvas)
                    self.update_subexpr_info(graph, new_canvas, ind, ind + 1, command_list, counter)
                elif add_sweeping_info:
                    new_canvas_set = self.canvas_stack[-1]
                    reward = self.get_reward(new_canvas_set, target)
                    new_canvas = new_canvas_set[0]
                    graph.nodes[counter]['reward'] = reward
                    graph.nodes[counter]['validty'] = None
                    graph.nodes[counter]['expression_length'] = 1

            elif c_type == "M":
                # take the set of coords and mirror it
                self.apply_mirror(param=command['param'], 
                                   add_splicing_info=add_splicing_info)
                # For Tree
                self.draw_node(graph, cur_parent_ind, counter, command)
                cur_parent_ind = counter
                self.non_terminal_expr_start_idx.append(ind)
            
            elif c_type == "DUMMY":
                assert add_splicing_info, "Only for Splicing"
                latest_coords_set = self.transformed_coords.pop()
                # Make a blank canvas
                new_canvas_set = [x.norm(dim=-1) + 1e-9 for x in latest_coords_set]
                self.canvas_stack.append(new_canvas_set)
                self.draw_node(graph, cur_parent_ind, counter, command)
                cur_parent_ind = self.get_next_parent_after_draw(graph, cur_parent_ind)

                cur_expr_canvas = self.draw.base_coords.clone()
                new_canvas = cur_expr_canvas.norm(dim=-1)+ 1e-9 
                self.update_subexpr_info(graph, new_canvas, ind, ind + 1, command_list, counter, dummy=True)
                

            self.tree_mirror_merge(graph, target, command_list, ind, add_sweeping_info, add_splicing_info)
            ## Housekeeping:
            if len(self.transformed_coords) == 0:
                self.transformed_coords.append([self.draw.base_coords.clone()])
            
            ## Apply Boolean if possible - multiple times: 
            self.tree_boolean_resolve(graph, target, command_list, ind, enable_subexpr_targets, add_sweeping_info, add_splicing_info)
            
        self.check_errors(command_list)
        
        self._output = self.canvas_stack.pop()[0]
        if reset:
            self.reset()
        if enable_subexpr_targets:
        # Now annotate the targets and target masks:
            self.create_subexpr_targets(graph, target)
            
        if reset:
            self.reset()
        return graph
        

    def create_subexpr_targets(self, graph, target, add_subexpr_stats=True):
        state_list = [graph.nodes[0]]
        target = th.stack([target, th.ones(target.shape, dtype=th.bool).to(target.device)], -1)
        # target = target.bool()
        self.target_stack = [target.clone()]
        while(state_list):
            cur_node = state_list[0]
            state_list = state_list[1:]
            cur_target = self.target_stack.pop()
            # Perform processing according to the node type
            node_type = cur_node['type']
            cur_node['subexpr_info']['expr_target'] = cur_target.clone().detach()
            

            if node_type == "B":
                c_symbol = cur_node['symbol']
                # Save the node as the target:
                # perform inverse boolean on each and add to stack.
                child_A_canvas = cur_node['subexpr_info']['child_A']
                child_B_canvas = cur_node['subexpr_info']['child_B']
                target_A = self.inverse_boolean[c_symbol](cur_target, child_B_canvas, 0)
                target_B = self.inverse_boolean[c_symbol](cur_target, child_A_canvas, 1)
                self.target_stack.append(target_B)
                self.target_stack.append(target_A)
                # Add the statistics:
                bbox = cur_node['subexpr_info']['bbox']
                canonical_shape, commands = self.get_canonical_shape_and_commands(cur_target.clone(), bbox)
                cur_node['subexpr_info']['canonical_target'] = canonical_shape
                cur_node['subexpr_info']['canonical_target_commands'] = commands
                if add_subexpr_stats:
                    self.calculate_subexpr_stats(cur_node)
                    self.calculate_canonical_subexpr_stats(cur_node)

            elif node_type == "T":
                # Apply inverse transform on target
                param = cur_node['param']
                c_symbol = cur_node['symbol']
                param = self.invert_transform_param(param, c_symbol)
                cur_target = self.inverse_transform[c_symbol](param=param, input_shape=cur_target)
                self.target_stack.append(cur_target)
                
            elif node_type == "M":
                # eventually add it here as well.
                bbox = cur_node['subexpr_info']['bbox']
                canonical_shape, commands = self.get_canonical_shape_and_commands(cur_target.clone(), bbox)
                cur_node['subexpr_info']['canonical_target'] = canonical_shape
                cur_node['subexpr_info']['canonical_target_commands'] = commands
                if add_subexpr_stats:
                    self.calculate_subexpr_stats(cur_node)
                    self.calculate_canonical_subexpr_stats(cur_node)
                param = cur_node['param']
                # c_symbol = cur_node['symbol']
                # param = self.invert_transform_param(param, c_symbol)
                # cur_target = self.inverse_transform[c_symbol](param=param, input_shape=cur_target)
                self.target_stack.append(cur_target)

            elif node_type == "D":
                # Add the statistics:
                bbox = cur_node['subexpr_info']['bbox']
                canonical_shape, commands = self.get_canonical_shape_and_commands(cur_target.clone(), bbox)
                cur_node['subexpr_info']['canonical_target'] = canonical_shape
                cur_node['subexpr_info']['canonical_target_commands'] = commands
                if add_subexpr_stats:
                    self.calculate_subexpr_stats(cur_node)
                    self.calculate_canonical_subexpr_stats(cur_node)
                    
            elif node_type == "RD":
                # Add the statistics:
                bbox = cur_node['subexpr_info']['bbox']
                canonical_shape, commands = self.get_canonical_shape_and_commands(cur_target.clone(), bbox)
                cur_node['subexpr_info']['canonical_target'] = canonical_shape
                cur_node['subexpr_info']['canonical_target_commands'] = commands
                if add_subexpr_stats:
                    self.calculate_subexpr_stats(cur_node)
                    self.calculate_canonical_subexpr_stats(cur_node)
                    
                ## LET GO of the target.
                # self.target_stack.append(cur_target)
            elif node_type == "ROOT":
                self.target_stack.append(cur_target)
            
            elif node_type == "DUMMY":
                # Get the bbox of the target!
                if self.mode == "3D":
                    masked_target = th.logical_and(cur_target[:,:,:,0], cur_target[:,:,:,1]).float()
                else:
                    masked_target = th.logical_and(cur_target[:,:,0], cur_target[:,:,1]).float()
                # do noise processing here:
                if self.denoise:
                    noise_count_tensor = masked_target.float().unsqueeze(0).unsqueeze(0)
                    nhbd_count = th.nn.functional.conv3d(noise_count_tensor, self.denoising_filter, padding=2)[0][0]
                    removal_mask = nhbd_count < self.denoisizing_threshold
                    # print(f"Removing {(removal_mask.float() * masked_target.float()).sum()} voxels")
                    masked_target[removal_mask] = 0
                if self.connected_components:
                    masked_target = self.maximal_connected_components(masked_target)
                
                floating_target_pseudo_sdf = -(masked_target-0.1)
                
                bbox = self.draw.return_bounding_box(floating_target_pseudo_sdf, normalized=False)
                cur_node['subexpr_info']['bbox'] = bbox
                canonical_shape, commands = self.get_canonical_shape_and_commands(cur_node['subexpr_info']['expr_shape'], bbox)
                cur_node['subexpr_info']['canonical_shape'] = canonical_shape
                cur_node['subexpr_info']['canonical_commands'] = commands
                canonical_shape, commands = self.get_canonical_shape_and_commands(cur_target.clone(), bbox)
                cur_node['subexpr_info']['canonical_target'] = canonical_shape
                cur_node['subexpr_info']['canonical_target_commands'] = commands
                if add_subexpr_stats:
                    self.calculate_subexpr_stats(cur_node)
                    self.calculate_canonical_subexpr_stats(cur_node)
                
            # Next set of children:
            cur_children_id = cur_node['children']
            for child_id in cur_children_id[::-1]:
                child = graph.nodes[child_id]
                state_list.insert(0, child)
    def maximal_connected_components(self, target):
        seg_voxels, n = cc3d.connected_components(target.cpu().numpy().astype(np.int32), return_N=True)
        if n >= 1:
            sizes = np.bincount(seg_voxels.ravel())[1:]
            ind = sizes.argmax()
            output = seg_voxels == ind + 1
            output = th.from_numpy(output.astype(np.float32))
            if sizes[ind] < self.cc_limit:
                output = target
        else:
            output = target
        return output
        
    def tree_boolean_resolve(self, graph, target, command_list, ind, enable_subexpr_targets=False,
                        add_sweeping_info=False, add_splicing_info=False):
        # assert not (add_sweeping_info and add_splicing_info), "CANNOT MAKE GRAPH FOR SWEEPING AND SPLICING TOGETHER."
        ## Apply Boolean if possible - multiple times: 
        if self.boolean_stack:
            # print("checking Boolean stack")
            while(self.canvas_pointer >= self.boolean_start_index[-1] + 2):
                # We can apply the operation!
                boolean_op = self.boolean_stack.pop()
                _ = self.boolean_start_index.pop()
                c_2_set = self.canvas_stack.pop()
                c_1_set = self.canvas_stack.pop()
                new_canvas_set = [boolean_op(x, y) for x, y in zip(c_1_set, c_2_set)]
                self.canvas_stack.append(new_canvas_set)
                # For tree:
                # Update the node with information
                boolean_command_idx = self.non_terminal_expr_start_idx.pop()
                boolean_graph_idx = boolean_command_idx + 1
                if add_sweeping_info:
                    reward = self.get_reward(new_canvas_set, target)
                    validity_status = self.get_validity(new_canvas_set, c_1_set, c_2_set)
                    graph.nodes[boolean_graph_idx]['reward'] = reward
                    graph.nodes[boolean_graph_idx]['validity'] = validity_status
                    graph.nodes[boolean_graph_idx]['expression_length'] = (ind + 1) - boolean_command_idx
                if add_splicing_info:
                    cur_bool_canvas = new_canvas_set.pop() 
                    expr_start = boolean_command_idx
                    expr_end = ind +1
                    self.update_subexpr_info(graph, cur_bool_canvas, expr_start, expr_end, command_list, boolean_graph_idx)
                    # cur_bool_canvas, expr_start, expr_end, command_list, graph_id
                    if enable_subexpr_targets:
                        graph.nodes[boolean_graph_idx]['subexpr_info']['child_A'] = (c_1_set.pop() <= 0)
                        graph.nodes[boolean_graph_idx]['subexpr_info']['child_B'] = (c_2_set.pop() <= 0)

                ### EXTRA CAN BE REMOVED
                self.tree_mirror_merge(graph, target, command_list, ind, add_sweeping_info, add_splicing_info)

                if not self.boolean_start_index:
                    break
        # Formatting:
    
    def tree_mirror_merge(self, graph, target, command_list, ind, add_sweeping_info=False, add_splicing_info=False):

        ## Resolve all the mirrors: 
        if self.mirror_start_boolean_stack:
            cur_mirror_bool_req = self.mirror_start_boolean_stack[-1]
            cur_mirror_canvs_req = self.mirror_start_canvas_stack[-1]
            while len(self.boolean_stack) == cur_mirror_bool_req and self.canvas_pointer == (cur_mirror_canvs_req +1):
                cur_mirror_bool_req = self.mirror_start_boolean_stack.pop()
                cur_mirror_canvs_req = self.mirror_start_canvas_stack.pop()
                previous_parallel_size = self.mirror_init_size.pop()
                # print("Reflect Merging!", previous_parallel_size, cur_mirror_canvs_req, cur_mirror_bool_req)
                c_1_set = self.canvas_stack.pop()
                set_a = c_1_set[:previous_parallel_size]
                set_b = c_1_set[previous_parallel_size:]
                new_canvas_set = [self.draw.union(x, y) for x, y in zip(set_a, set_b)]
                self.canvas_stack.append(new_canvas_set)
                boolean_command_idx = self.non_terminal_expr_start_idx.pop()
                boolean_graph_idx = boolean_command_idx + 1
                if add_sweeping_info:
                    validity_status = self.get_validity(new_canvas_set, set_a, set_b)
                    reward = self.get_reward(new_canvas_set, target)
                    graph.nodes[boolean_graph_idx]['reward'] = reward
                    graph.nodes[boolean_graph_idx]['validity'] = validity_status
                    graph.nodes[boolean_graph_idx]['expression_length'] = (ind + 1) - boolean_command_idx
                if add_splicing_info:
                    cur_bool_canvas = new_canvas_set.pop() 
                    expr_start = boolean_command_idx
                    expr_end = ind +1
                    self.update_subexpr_info(graph, cur_bool_canvas, expr_start, expr_end, command_list, boolean_graph_idx)
                
                if not self.mirror_start_boolean_stack:
                    break
                else:
                    cur_mirror_bool_req = self.mirror_start_boolean_stack[-1]
                    cur_mirror_canvs_req = self.mirror_start_canvas_stack[-1]


    def update_subexpr_info(self, graph, cur_canvas, expr_start, expr_end, command_list, graph_id, dummy=False):

        
        cur_commands = command_list[expr_start:expr_end]
        if dummy:
            res = self.draw.resolution
            if self.mode == "3D":
                bbox = np.array([[0, 0, 0], [res, res, res]])
            else:
                bbox = np.array([[0, 0], [res, res]])
        else:
            bbox = self.draw.return_bounding_box(cur_canvas, normalized=False)
        current_shape = (cur_canvas < 0)
        
        graph.nodes[graph_id]['subexpr_info']['expr_shape'] = current_shape
        graph.nodes[graph_id]['subexpr_info']['bbox'] = bbox
        graph.nodes[graph_id]['subexpr_info']['commands'] = cur_commands
        graph.nodes[graph_id]['subexpr_info']['command_ids'] = (expr_start, expr_end)
        graph.nodes[graph_id]['subexpr_info']['command_length'] = len(cur_commands) 
        canonical_shape, commands = self.get_canonical_shape_and_commands(current_shape, bbox)
        graph.nodes[graph_id]['subexpr_info']['canonical_shape'] = canonical_shape
        graph.nodes[graph_id]['subexpr_info']['canonical_commands'] = commands


    def get_canonical_shape_and_commands(self, target, bbox):
        # Pretend its a doublet
        reduce_dim = False
        if self.mode == "3D":
            if len(target.shape) == 3:
                target = th.stack([target,  th.ones(target.shape, dtype=th.bool).to(target.device)] , -1)
                reduce_dim = True
        else:
            if len(target.shape) == 2:
                target = th.stack([target,  th.ones(target.shape, dtype=th.bool).to(target.device)] , -1)
                reduce_dim = True

        cur_bbox = bbox.copy()
        normalized_bbox = (-1 + (cur_bbox + 0.5)/self.draw.grid_divider)
        center = normalized_bbox.mean(0)
        cur_bbox[1] += 1
        normalized_bbox = (-1 + (cur_bbox + 0.5)/self.draw.grid_divider)
        size = normalized_bbox[1] - normalized_bbox[0]
        inverse_translate_params = -center
        inverse_size_params = (2 - 1/self.draw.resolution)/(size + 1e-9)
        
        ## HACK For inverted skeps
        if self.draw.mode == "inverted":
            inverse_translate_params[[0, 1, 2]] = inverse_translate_params[[1, 0, 2]]
            inverse_size_params[[0, 1, 2]] = inverse_size_params[[1, 0, 2]]
        trans_comm = 'translate'
        target = self.inverse_transform[trans_comm](param=inverse_translate_params, input_shape=target)
        scale_comm = 'scale'
        target = self.inverse_transform[scale_comm](param=inverse_size_params, input_shape=target)
        
        command_list = [{'type': "T", "symbol": scale_comm, "param": inverse_size_params},
                        {'type': "T", "symbol": trans_comm, "param": inverse_translate_params}]
        
        if reduce_dim:
            if self.mode == "3D":
                target = target[:,:,:,0]
            else:
                target = target[:,:,0]


        return target, command_list


    def calculate_subexpr_stats(self, cur_node):
        
        pred_shape = cur_node['subexpr_info']['expr_shape']
        target = cur_node['subexpr_info']['expr_target']
        target_mask = target[:,:,:,1]
        target_shape = target[:,:,:,0]

        # use Bounding box
        bbox = cur_node['subexpr_info']['bbox']
        min_x, min_y, min_z = bbox[0]
        max_x, max_y, max_z = bbox[1] + 1

        pred_shape = pred_shape[min_x:max_x, min_y:max_y, min_z:max_z]
        target_shape = target_shape[min_x:max_x, min_y:max_y, min_z:max_z]
        target_mask = target_mask[min_x:max_x, min_y:max_y, min_z:max_z]
        
        ## Increase the mask, and also calculate the score in canonical form.

        R = th.sum(th.logical_and(th.logical_and(pred_shape, target_shape), target_mask)) / \
                (th.sum(th.logical_and(th.logical_or(pred_shape, target_shape), target_mask)) + 1e-6)
        cur_node['subexpr_info']['masked_iou'] = R
        cur_node['subexpr_info']['masking_rate'] = 1 - target_mask.float().mean()
        cur_node['subexpr_info']['masked_matching'] = th.logical_and((pred_shape == target_shape), target_mask).sum() / target_mask.sum()
        ### TODO: Add Hierarchy Volume and Expression length
        cur_node['subexpr_info']['sa_sweep_metric'] = th.logical_and(th.logical_and(pred_shape, target_shape), target_mask).sum()

    def calculate_canonical_subexpr_stats(self, cur_node):
        
        pred_shape = cur_node['subexpr_info']['canonical_shape']
        target = cur_node['subexpr_info']['canonical_target']
        
        if self.mode == "3D":
            target_mask = target[:,:,:,1]
            target_shape = target[:,:,:,0]
        else:
            target_mask = target[:,:,1]
            target_shape = target[:,:,0]
        
        # use Bounding box

        R = th.sum(th.logical_and(th.logical_and(pred_shape, target_shape), target_mask)) / \
                (th.sum(th.logical_and(th.logical_or(pred_shape, target_shape), target_mask)) + 1e-6)
        cur_node['subexpr_info']['canonical_masked_iou'] = R
        cur_node['subexpr_info']['canonical_masking_rate'] = 1 - target_mask.float().mean()
        cur_node['subexpr_info']['canonical_masked_matching'] = th.logical_and((pred_shape == target_shape), target_mask).sum() / target_mask.sum()

    def invert_transform_param(self, param, c_symbol):
        if c_symbol in ["translate", "rotate", "mirror"]:
            param = -param
        # elif c_symbol == "rotate_with_matrix":
        #     param = param.T
        elif c_symbol == "scale":
            param = 1/(param + 1e-9)
        
        return param
    
    def get_reward(self, new_canvas_set, target):
        # Cheat and add it to only one:
        selected_shape = new_canvas_set[0]

        R = get_reward(selected_shape, target)
        return R

    def get_validity(self, output, top, bottom):
        # Each canvas set has to be valid.
        num_canvas = len(output)
        global_validity = [True, True, True]
        for ind in range(num_canvas):
            selected_out = output[ind]
            selected_top = top[ind]
            selected_bottom = bottom[ind]
            validity = self.get_canvas_validity(selected_out, selected_top, selected_bottom)
            global_validity = [x and validity[ind] for ind, x in enumerate(global_validity)]
        return global_validity
    
    def get_canvas_validity(self, output, top, bottom):
        total = output.nelement()
        output_shape = (output <= 0)
        top_shape = (top <= 0)
        bottom_shape = (bottom <= 0)
        val_1 = th.sum(output_shape) / total
        val_2 = th.sum(output_shape ^ bottom_shape) / th.sum(output_shape)
        val_3 = th.sum(output_shape ^ top_shape) / th.sum(output_shape)
        cond_1 = val_1 > self.threshold # < 40*40
        # print(val_1, val_2, val_3 )
        # Also not too small
        # print("\% occupied after operation", val_1)
        # print("\% difference", val_2, val_3)
        cond_2 = val_2 > self.threshold_diff
        cond_3 = val_3 > self.threshold_diff
        output = [cond_1,  cond_2, cond_3]
        # output = [cond_1.item(),  cond_2.item(), cond_3.item()]
        # print("Validity", output)
        return output
    
    def get_masking_based_validity(self, graph):

        # Now annotate the targets and target masks:
        state_list = [graph.nodes[0]]
        while(state_list):
            cur_node = state_list[0]
            state_list = state_list[1:]
            # Perform processing according to the node type
            node_type = cur_node['type']
            
            if node_type == "B":
                execution = cur_node['subexpr_info']['expr_shape']
                if self.mode == "3D":
                    execution_mask = cur_node['subexpr_info']['expr_target'][:,:,:,1]
                else:
                    execution_mask = cur_node['subexpr_info']['expr_target'][:,:,1]

                valid = th.any(th.logical_and(execution, execution_mask))
                if not valid:
                    cur_node['validity'][0] = False

            cur_children_id = cur_node['children']
            for child_id in cur_children_id[::-1]:
                child = graph.nodes[child_id]
                state_list.insert(0, child)
        return graph
    
    def tree_to_command(self, graph, cur_node):
        # Treat the tree node as starting point. 
        command_list = []
        state_list = [cur_node]
        while(state_list):
            selected_node = state_list[0]
            state_list = state_list[1:]
            command = self.get_command_from_tree_node(selected_node)
            command_list.append(command)
            children_id = selected_node['children']
            # children_id.sort()
            for child_id in children_id[::-1]:
                new_child = graph.nodes[child_id]
                state_list.insert(0, new_child)

        return command_list
    
    def tree_to_expression(self, graph, cur_node):
        # Treat the tree node as starting point. 
        expression_list = []
        state_list = [cur_node]
        while(state_list):
            selected_node = state_list[0]
            state_list = state_list[1:]
            # command = self.get_command_from_tree_node(selected_node)
            node_data = selected_node['node_data']
            if 'true_expression' in node_data.keys():
                expression = node_data['true_expression']
                expression_list.append(expression)
            children_id = selected_node['children']
            # children_id.sort()
            for child_id in children_id[::-1]:
                new_child = graph.nodes[child_id]
                state_list.insert(0, new_child)

        return expression_list
    
    def cmd_to_expression(self, commands):
        expression_list = []
        for cmd in commands:
            if 'true_expression' in cmd.keys():
                expression = cmd['true_expression']
                expression_list.append(expression)
        return expression_list
        
    
    def get_command_from_tree_node(self, selected_node):
        c_type = selected_node['type']
        c_symbol = selected_node['symbol']
        new_command = dict(type=c_type, symbol=c_symbol)
        if 'param' in selected_node.keys():
            new_command['param'] = selected_node['param']
        return new_command
        
    def draw_node(self, graph, cur_parent_ind, counter, command):

        node_data = {x:y for x, y in command.items()}
        # if 'param' in node_data.keys():
        #     node_data['param'] = np.array(node_data['param'])
        node_data['parent'] = graph.nodes[cur_parent_ind]['node_id']
        node_data['children'] = []
        graph.add_node(counter, node_id=counter, subexpr_info=dict(), **node_data)
        # print("Added Node %d with %s" % (counter, c_symbol))
        graph.nodes[cur_parent_ind]['children'].append(counter)
        graph.add_edge(cur_parent_ind, counter)
        # print("Connecting %d to %d" % (cur_parent_ind, counter))

    def get_next_parent_after_draw(self, graph, cur_parent_ind):

        cur_parent = graph.nodes[cur_parent_ind]
        # print("starting draw search with %d" % cur_parent_ind)
        while(True):
            cur_parent_ind = cur_parent['node_id']
            cur_parent_type = cur_parent['type']
            # print("cur parent Type %s" % cur_parent_type, cur_parent_ind)
            cur_siblings = cur_parent['children']
            if cur_parent_type == "B":
                # may do something:
                n_sibling = len(cur_siblings)
                # print("n_siblings", n_sibling)
                if n_sibling == 1:
                    # print("Boolean has 1 child now. This works!")
                    break
                elif n_sibling == 2:
                    # print("Boolean has 2 children now; Going further up.")
                    pass
                else:
                    raise ValueError("Found a Boolean parent with more than 2 children (or 0).")
            elif cur_parent_type == "T":
                # simply go up:
                pass
            elif cur_parent_type == "R":
                # simply go up:
                # SINGLE Child node
                pass
            elif cur_parent_type == ["D", "DUMMY"]:
                raise ValueError(cur_parent_type, " as a child of a Draw - ERROR!")

            next_parent_ind = cur_parent['parent']
            if  next_parent_ind is None:
                # print("Reached Root!!")
                cur_parent_ind = None
                break
            else:
                cur_parent = graph.nodes[next_parent_ind]
                
        return cur_parent_ind