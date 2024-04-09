from .draw import DiffDraw3D
import os
import torch as th
import numpy as np
from .compiler_utils import convert_sdf_samples_to_ply, convert_stl_to_gltf
import cv2
from CSG.utils.visualization import render_stl

class MCSG3DCompiler():

    def __init__(self, resolution=64, scale=64, scad_resolution=30, device="cpu", draw_mode="inverted", *args, **kwargs):

        self.draw = DiffDraw3D(resolution=resolution, device=device, mode=draw_mode)

        self.scad_resolution = scad_resolution
        self.space_scale = scale
        self.resolution = resolution
        self.scale = scale
        self.draw_mode = draw_mode

        self.mirror_start_boolean_stack = []
        self.mirror_start_canvas_stack = []
        self.mirror_init_size = []

        self.device = device
        self.set_to_full()

        self.reset()
        self.set_init_mapping()

    def set_init_mapping(self):
        self.boolean_to_execute = {
            'union': self.draw.union,
            'intersection': self.draw.intersection,
            'difference': self.draw.difference,
        }
        self.transform_to_execute = {
            'translate': self.draw.translate,
            'rotate': self.draw.rotate,
            'scale': self.draw.scale,
            'quat_rotate': self.draw.quat_rotate,
        }
        self.draw_to_execute = {
            'sphere': self.draw.draw_sphere,
            'cuboid': self.draw.draw_cuboid,
            'cylinder': self.draw.draw_cylinder,
            "ellipsoid" : self.draw.draw_ellipsoid,
            'infinite_cylinder': self.draw.draw_infinite_cylinder,
            'infinite_cone': self.draw.draw_infinite_cone,
        }

        self.inverse_boolean = {
            'union': self.draw.union_invert,
            'intersection': self.draw.intersection_invert,
            'difference': self.draw.difference_invert
        }

    
        self.inverse_transform = {
            'translate': self.draw.shape_translate,
            'rotate': self.draw.shape_rotate,
            'scale': self.draw.shape_scale,
            'mirror': self.draw.shape_mirror,
        }
        
    @property
    def canvas_pointer(self):
        return len(self.canvas_stack)
        
    def adjust_scale(self, param):
        param = [x * self.space_scale for x in param]
        return param

    def reset(self):
        self.boolean_stack = []
        self.boolean_start_index = []
        self.canvas_stack = []
        self.transformed_coords = [[self.draw.base_coords.clone()]]
        self.mirror_start_boolean_stack = []
        self.mirror_start_canvas_stack = []
        self.mirror_init_size = []
        
    def _compile(self, command_list, reset=True):
        
        if reset:
            self.reset()

        for command in command_list:
            c_type = command['type']
            c_symbol = command['symbol']
            if c_type == "B":
                # print("creating Boolean", command)
                self.apply_boolean(boolean_command=self.boolean_to_execute[c_symbol])
            elif c_type == "T":
                # print("creating Transform", command)
                self.apply_transform(transform_command=self.transform_to_execute[c_symbol], param=command['param'])
            elif c_type == "D":
                # print("creating Draw", command)
                self.apply_draw(draw_command=self.draw_to_execute[c_symbol])
            elif c_type == "M":
                # take the set of coords and mirror it
                self.apply_mirror(param=command['param'])
            elif c_type == "RD":
                # For FCSG Code Splice
                # print("creating Draw", command)
                draw_symbol = c_symbol.split("_")[1]
                self.apply_rotate_draw(draw_command=self.draw_to_execute[draw_symbol], param=command['param'])

            self.mirror_merge()
            ## Housekeeping:
            if len(self.transformed_coords) == 0:
                self.transformed_coords.append([self.draw.base_coords.clone()])
            ## Apply Boolean if possible - multiple times: 
            self.boolean_resolve()
            
        self.check_errors(command_list)

        self._output = self.canvas_stack.pop()[0]
        if reset:
            self.reset()

    def set_device(self, device):
        self.device = device
        self.draw.device = device

    def set_tensor_type(self, tensor_type):
        self.tensor_type = tensor_type
        self.draw.tensor_type = tensor_type
    
    def set_to_half(self):
        self.tensor_type = th.float16
        self.draw.set_to_half()

    def set_to_full(self):
        self.tensor_type = th.float32
        self.draw.set_to_full()
    
    def set_to_cuda(self):
        self.draw.set_to_cuda()
        self.device = th.device("cuda")
        
    def set_to_cpu(self):
        self.draw.set_to_cpu()
        self.device = th.device("cpu")
        

    def _get_complexity(self, command_list, reset=True):
        max_complexity = 2
        if reset:
            self.reset()
        self.transformed_coords = [[0]]
        for command in command_list:
            c_type = command['type']
            c_symbol = command['symbol']
            if c_type == "B":
                # print("creating Boolean", command)
                self.boolean_stack.append(command)
                self.boolean_start_index.append(self.canvas_pointer)
                # Also Clone the top of the transform codes:
                latest_coords_set = self.transformed_coords[-1]
                cloned_coords_set = [0 for x in latest_coords_set]
                self.transformed_coords.append(cloned_coords_set)
            elif c_type == "T":
                # print("creating Transform", command)
                pass
            elif c_type == "D":
                # print("creating Draw", command)
                latest_coords_set = self.transformed_coords.pop()
                new_canvas_set = [0 for x in latest_coords_set]
                self.canvas_stack.append(new_canvas_set)
            elif c_type == "M":
                # take the set of coords and mirror it
                latest_coords_set = self.transformed_coords.pop()
                previous_parallel_size = len(latest_coords_set)
                mirrored_coord_set = [0 for x in latest_coords_set]
                latest_coords_set.extend(mirrored_coord_set)
                self.transformed_coords.append(latest_coords_set)
                self.mirror_start_boolean_stack.append(len(self.boolean_stack))
                self.mirror_start_canvas_stack.append(self.canvas_pointer)
                self.mirror_init_size.append(previous_parallel_size)
            elif c_type == "RD":
                # For FCSG Code Splice
                # print("creating Draw", command)
                latest_coords_set = self.transformed_coords.pop()
                latest_coords_set = [0 for x in latest_coords_set]
                new_canvas_set = [0 for x in latest_coords_set]
                self.canvas_stack.append(new_canvas_set)

            # Measure Complexity here:
            n_canvas = np.sum([len(x) for x in self.transformed_coords])
            ## Resolve all the mirrors: 
            self.complexity_mirror_resolve()
            ## Housekeeping:
            if len(self.transformed_coords) == 0:
                self.transformed_coords.append([0])
            ## Apply Boolean if possible - multiple times: 
            self.complexity_boolean_resolve()
            n_canvas_2 = np.sum([len(x) for x in self.transformed_coords])
            complexity = max(n_canvas, n_canvas_2)
            max_complexity = max(max_complexity, complexity)
            
        self.check_errors(command_list)

        if reset:
            self.reset()
        return max_complexity

    def get_node_commands(self, graph, cur_node):
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
    
    def check_errors(self, command_list):

        if len(self.canvas_stack) > 1:
            print("canvas stack has %d items instead 0f 1" % len(self.canvas_stack))
            print("Commands", command_list)
            raise ValueError
        if len(self.boolean_stack) > 0:
            print("boolean stack has %s items instead of 0" % len(self.boolean_stack))
            print("Commands", command_list)
            raise ValueError

    def complexity_mirror_resolve(self, ):

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
                new_canvas_set = [0 for x, y in zip(set_a, set_b)]
                self.canvas_stack.append(new_canvas_set)
                if not self.mirror_start_boolean_stack:
                    break
                else:
                    cur_mirror_bool_req = self.mirror_start_boolean_stack[-1]
                    cur_mirror_canvs_req = self.mirror_start_canvas_stack[-1]

    def complexity_boolean_resolve(self):
        if self.boolean_stack:
            # print("checking Boolean stack")
            while(self.canvas_pointer >= self.boolean_start_index[-1] + 2):
                # We can apply the operation!
                boolean_op = self.boolean_stack.pop()
                _ = self.boolean_start_index.pop()
                c_2_set = self.canvas_stack.pop()
                c_1_set = self.canvas_stack.pop()
                new_canvas_set = [0 for x, y in zip(c_1_set, c_2_set)]
                # print("applying", boolean_op)
                self.canvas_stack.append(new_canvas_set)
                self.complexity_mirror_resolve()
                if not self.boolean_start_index:
                    break

    def boolean_resolve(self):
        if self.boolean_stack:
            # print("checking Boolean stack")
            while(self.canvas_pointer >= self.boolean_start_index[-1] + 2):
                # We can apply the operation!
                boolean_op = self.boolean_stack.pop()
                _ = self.boolean_start_index.pop()
                c_2_set = self.canvas_stack.pop()
                c_1_set = self.canvas_stack.pop()
                new_canvas_set = [boolean_op(x, y) for x, y in zip(c_1_set, c_2_set)]
                # print("applying", boolean_op)
                self.canvas_stack.append(new_canvas_set)
                self.mirror_merge()
                if not self.boolean_start_index:
                    break

    def mirror_merge(self):

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
                if not self.mirror_start_boolean_stack:
                    break
                else:
                    cur_mirror_bool_req = self.mirror_start_boolean_stack[-1]
                    cur_mirror_canvs_req = self.mirror_start_canvas_stack[-1]

    def apply_boolean(self,boolean_command, add_splicing_info=False):
        self.boolean_stack.append(boolean_command)
        self.boolean_start_index.append(self.canvas_pointer)
        # Also Clone the top of the transform codes:
        latest_coords_set = self.transformed_coords[-1]
        if add_splicing_info:
            cur_expr_canvas = self.draw.base_coords.clone()
            latest_coords_set.append(cur_expr_canvas)
        cloned_coords_set = [x.clone() for x in latest_coords_set]
        self.transformed_coords.append(cloned_coords_set)
    
    def apply_transform(self, transform_command, param):
        latest_coords_set = self.transformed_coords.pop()
        latest_coords_set = [transform_command(param, coords=x) for x in latest_coords_set]
        self.transformed_coords.append(latest_coords_set)
    
    def apply_draw(self, draw_command):
        latest_coords_set = self.transformed_coords.pop()
        new_canvas_set = [draw_command(coords=x) for x in latest_coords_set]
        self.canvas_stack.append(new_canvas_set)

    def apply_rotate_draw(self, draw_command, param):
        latest_coords_set = self.transformed_coords.pop()
        latest_coords_set = [self.transform_to_execute["rotate"](param, coords=x) for x in latest_coords_set]
        new_canvas_set = [draw_command(coords=x) for x in latest_coords_set]
        self.canvas_stack.append(new_canvas_set)
    
    def apply_mirror(self, param, add_splicing_info=False):
        latest_coords_set = self.transformed_coords.pop()
        previous_parallel_size = len(latest_coords_set)
        if add_splicing_info:
                cur_expr_canvas = self.draw.base_coords.clone()
                latest_coords_set.append(cur_expr_canvas)
                previous_parallel_size += 1
        mirrored_coord_set = [self.draw.mirror_coords(param, coords=x) for x in latest_coords_set]
        latest_coords_set.extend(mirrored_coord_set)
        self.transformed_coords.append(latest_coords_set)
        self.mirror_start_boolean_stack.append(len(self.boolean_stack))
        self.mirror_start_canvas_stack.append(self.canvas_pointer)
        self.mirror_init_size.append(previous_parallel_size)

    def get_output_shape(self, return_bool=False):
        output = (self._output<=0)
        if not return_bool:
            output = output.to(self.tensor_type)
        return output

    def get_output_sdf(self):
        return self._output.clone()
    
    def is_valid_primitive(self, sdf):
        return self.draw.is_valid_primitive(sdf)

    def is_valid_sdf(self, sdf):
        return self.draw.is_valid_sdf(sdf)
        
    def get_point_cloud(self, sdf):
        return self.draw.return_inside_coords(sdf)
    
    ### Conversion Functions
    def _compile_to_scad(self, command_list):
        
        scad_list = ['$fn=%d;\n' % self.scad_resolution]
        ## Copy command:
        mirror_copy_command = "module copy_mirror(vec=[0,1,0])\n {\n children();\n mirror(vec) children();\n}"
        scad_list.append(mirror_copy_command)
        scad_list.append("rotate([0, 0, -90]){\n")
        stack_state = 1
        canvas_state = 0
        for command in command_list:
            c_type = command['type']
            c_symbol = command['symbol']
            
            if c_type == "B":
                # print("creating Boolean", command)
                scad_command = "%s%s(){\n" % ('\t' * stack_state, c_symbol)
                stack_state += 1
                self.boolean_stack.append(stack_state)
                self.boolean_start_index.append(canvas_state)
                scad_list.append(scad_command)
                
            elif c_type == "T":
                # print("creating Transform", command)
                param = command['param']
                if c_symbol == "translate":
                    param = self.adjust_scale(param)
                if c_symbol == "rotate":
                    param = [-x for x in param]
                scad_command = "%s%s([%f, %f, %f]){\n" % ('\t' * stack_state, 
                                                          c_symbol, param[0], param[1], param[2])
                stack_state += 1
                self.transformed_coords.append(stack_state)
                scad_list.append(scad_command)
            elif c_type == "M":
                param = command['param']
                if c_symbol == "mirror":
                    scad_func = "copy_mirror"
                scad_command = "%s%s([%f, %f, %f]){\n" % ('\t' * stack_state, 
                                                          scad_func, param[0], param[1], param[2])
                stack_state += 1
                self.transformed_coords.append(stack_state)
                scad_list.append(scad_command)
                
            elif c_type == "D":
                # print("creating Draw", command)
                scad_command = self.draw_command_to_scad(command, stack_state)                    
                canvas_state += 1
#                 stack_state -= 1
                ## Check if we can close:
                state_update = True
                boolean_update = True
                while(state_update or boolean_update):
                    state_update = False
                    boolean_update = False
                    if self.boolean_stack:
                        while(stack_state > self.boolean_stack[-1]):
                            stack_state -= 1
                            scad_command += "%s} \n" %('\t' * stack_state)
                            state_update = True
                        if (canvas_state >= self.boolean_start_index[-1] + 2):
                            canvas_state -=1
                            stack_state -= 1
                            scad_command += "%s} \n" %('\t' * stack_state)
                            self.boolean_start_index.pop()
                            self.boolean_stack.pop()
                            boolean_update = True
                    else:
                        while(stack_state > 0):
                            stack_state -= 1
                            scad_command += "%s} \n" %('\t' * stack_state)
                            state_update = True
                    
                scad_list.append(scad_command)
        self.reset()
        
        return scad_list

    def draw_command_to_scad(self, command, stack_state):
        c_symbol = command['symbol']
        param = [0.5, 0.5, 0.5] # command['param']
        param = self.adjust_scale(param)
        if c_symbol == "sphere":
            scad_command = "%s%s(r=%f, center=true);\n" % ('\t' * stack_state, c_symbol, param[0])
        elif c_symbol == "cylinder":
            scad_command = "%s%s(r=%f, h=%f, center=true);\n" % ('\t' * stack_state, c_symbol, param[0], param[1] * 2)
        elif c_symbol == "cuboid":
            scad_command = "%scube([%f, %f, %f], center=true);\n" % ('\t' * stack_state, param[0] * 2, 
                                                                    param[1] * 2, 
                                                                    param[2] * 2)
        elif c_symbol == "ellipsoid":
            ## A bit tricky
            radius = param[0]
            scad_command = "%s scale([%f, %f, %f]){\n" % ('\t' * stack_state, param[0]/float(radius), 
                                                        param[1]/float(radius),
                                                        param[2]/float(radius))
            scad_command += "%ssphere(r=%f, center=true);\n" % ('\t' * (stack_state +1), radius)
            scad_command += "%s} \n" %('\t' * stack_state)
        return scad_command
    

    def write_to_scad(self, command_list, file_name):
        commands = self._compile_to_scad(command_list)
        with open(file_name, 'w') as f:
            for command in commands:
                f.write(command)
    
    def write_to_stl(self, command_list, file_name):
        self.write_to_scad(command_list, file_name)
        scad_command = "openscad -o %s %s" %(file_name, file_name)
        # print("running", scad_command)
        os.system(scad_command)

    def get_rendered_image(self, command_list, file_name="tmp.scad"):
        self.write_to_scad(command_list, file_name)
        scad_command = "openscad -o %s --viewall %s" %("tmp.png", file_name)
        os.system(scad_command)
        image = cv2.imread("tmp.png")
        return image


    
    def march_to_ply(self, command_list, file_name):
        ## Since its all rotated: 
        rotate_command = {'type': "T", "symbol": "rotate", "param": [0, 0, 90]}
        command_list = [rotate_command] + command_list
        self._compile(command_list)
        sdf_values = self._output
        voxel_size = self.space_scale/float(self.draw.grid_shape[0])
        offset =  - self.space_scale  / 2.0  + 0.5 * voxel_size
        convert_sdf_samples_to_ply(pytorch_3d_sdf_tensor=sdf_values.cpu(), 
                                   voxel_grid_origin=[offset,offset,offset],
                                   voxel_size=voxel_size,
                                   ply_filename_out=file_name)
    
    def write_to_gltf(self, command_list, file_name):
        tmp_file = "tmp.stl"
        self.write_to_stl(command_list, tmp_file)
        convert_stl_to_gltf(tmp_file, file_name)
        os.system("rm %s" % tmp_file)
        
    
    def generate_images(self, graph):
        bbox_file = "bbox_file.stl"
        assert os.path.exists(bbox_file), "Please create bbox first"
        
        for node_id in range(len(graph.nodes)):
            cur_node = graph.nodes[node_id]
            if cur_node["subexpr_info"]:
                commands = cur_node["subexpr_info"]['commands']
                self.write_to_stl(commands, "tmp.stl")
                render_stl("tmp.stl", "tmp.png")
                image = cv2.imread("tmp.png")
                cur_node['image'] = image