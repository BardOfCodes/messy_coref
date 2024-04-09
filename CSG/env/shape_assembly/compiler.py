"""
The compiler will get a list of SA commands which on execution yield a modHCSG program -> only cuboids and unions.
"""
import time
from .draw import SADiffDraw3D
# from .compiler_utils import SACuboid, SAAttach, BooleanHolder
from CSG.env.csg3d.compiler import MCSG3DCompiler
from .sa_wrapper import SAProgram
import torch as th
import numpy as np
from .compiler_utils import _create_command_list_from_cuboid, get_params_from_sa
from collections import defaultdict

class SACompiler(MCSG3DCompiler):

    def __init__(self, resolution=64, scale=64, scad_resolution=30, device="cuda", varied_bbox_mode=False, *args, **kwargs):

        self.draw = SADiffDraw3D(resolution, device)

        self.resolution = resolution
        self.device = device

        self.scad_resolution = scad_resolution
        self.space_scale = scale
        self.resolution = resolution
        self.scale = scale
        self.tensor_type = th.float32

        self.reset()
        self.set_init_mapping()
        self.transform_to_execute['reflect'] = self.draw.mirror_coords
        self.inverse_transform['reflect'] = self.draw.shape_mirror

        self.transform_to_execute['rotate_with_matrix'] = self.draw.rotate_with_matrix
        self.inverse_transform['rotate_with_matrix'] = self.draw.shape_rotate_with_matrix

        self.hierarchical_state = []
        self.varied_bbox_mode = varied_bbox_mode


    def reset(self):
        self.boolean_stack = []
        self.boolean_start_index = []
        self.canvas_stack = []
        self.bool_canvas_req = []
        self.transformed_coords = [[self.draw.base_coords.clone()]]
        # For SA
        self.sa_executed_progs = {}
        self.sub_programs_replacement_dict = []
        self.hcsg_exrps = []
        self.hcsg_program = []
        self.cuboid_type_dicts = []
        self.final_hcsg_exprs = {}
        self._output = None
    
    def _compile(self, command_bundle, reset=True):
        # 
        if reset:
            self.reset()
        self._sa_compile(command_bundle)
        # csg_commands = self.convert_to_hcsg()
        self._csg_compile(self.hcsg_program, reset)

    def _sa_compile(self, sa_programs):
        hcsg_command_dicts = []
        for ind, cur_prog in enumerate(sa_programs):
            # Convert "cur program to form"
            TP = SAProgram(device=self.device, dtype=self.tensor_type)
            for cur_command in cur_prog:
                sa_command = cur_command['type']
                param = cur_command['param']
                # print("sa_command", sa_command, param)
                TP.execute_with_params(sa_command, param)
            self.sa_executed_progs[ind] = TP

            cur_hcsg_commands = []
            # also create the HCSG program and execute it
            for name, cuboid in TP.cuboids.items():
                if name == "bbox":
                    # Skip bbox
                    continue
                cube_param = cuboid.getParams()
                cuboid_type = cuboid.cuboid_type
                reflect = "_ref_" in name
                if reflect:
                    reflect_dir = name.split("_")[-2]
                else:
                    reflect_dir = None
                hcsg_cube_commands = _create_command_list_from_cuboid(cube_param, reflect, reflect_dir)
                cur_hcsg_commands.append((cuboid_type, hcsg_cube_commands))
            hcsg_command_dicts.append(cur_hcsg_commands)
        # print("program_over")

        # Now start completing the tree in reverse order
        reversed_ind = len(sa_programs) - 1
        for cur_hcsg_commands in hcsg_command_dicts[::-1]:
            # same algo -> convert
            cur_hcsg_program = []
            for cube_type, commands in cur_hcsg_commands:
                if cube_type == 0:
                    cur_hcsg_program.extend(commands)
                else:
                    # Plug in the "complete" program from hcsg
                    child_program = self.sa_executed_progs[cube_type]
                    hcsg_to_plug = self.final_hcsg_exprs[cube_type]
                    if self.varied_bbox_mode:
                        bbox_param = child_program.cuboids['bbox'].getParams()
                        t_p, s_p, r_p = get_params_from_sa(bbox_param)
                        translate_dict = dict(type="T", symbol="translate", param=-t_p)
                        scale_dict = dict(type="T", symbol="scale", param=1/s_p)
                        rotate_dict = dict(type="T", symbol="rotate_with_matrix", param=r_p.T)
                        conversion = [scale_dict, rotate_dict, translate_dict]
                    else:
                        conversion = []
                    commands = commands[:-1] + conversion + hcsg_to_plug
                    cur_hcsg_program.extend(commands)
            cur_hcsg_program.insert(0, {'type': "B", "symbol": "union", "canvas_count": len(cur_hcsg_commands)})
            self.final_hcsg_exprs[reversed_ind] = cur_hcsg_program
            reversed_ind -= 1
        cur_hcsg = self.final_hcsg_exprs[0]
        # cur_hcsg.insert(0, {'type': "T", "symbol": "rotate", "param": [0, 90, 0]})
        self.hcsg_program = cur_hcsg

    def _get_complexity(self, sa_programs, reset=True):
        self._sa_complexity(sa_programs)
        max_complexity = self._csg_complexity()
        # print("MAX Complexity", max_complexity)
        return max_complexity

    def _sa_complexity(self, sa_programs, reset=True):
        # First make the hcsg_program
        hcsg_command_dicts = []
        draw_dict = dict(type="D", symbol="cuboid")
        for ind, cur_prog in enumerate(sa_programs):
            cur_hcsg_commands = []
            cube_to_type = dict()
            for cur_command in cur_prog:
                sa_command = cur_command['type']
                if sa_command == "sa_cuboid":
                    # get cuboid type:
                    cube_type = cur_command['param'][-1]
                    cube_name = cur_command['param'][0]
                    cube_to_type[cube_name] = cube_type
                    cur_hcsg_commands.append((cube_type, [draw_dict]))
                elif sa_command == "sa_reflect":
                    cube_name = cur_command['param'][0]
                    cube_type = cube_to_type[cube_name]
                    cur_hcsg_commands.append((cube_type, [draw_dict]))
                    # get cuboid type:
                elif sa_command == "sa_translate":
                    # get cuboid type:
                    cube_name = cur_command['param'][0]
                    cube_type = cube_to_type[cube_name]
                    n = cur_command['param'][2]
                    for i in range(n):
                        cur_hcsg_commands.append((cube_type, [draw_dict]))
                else:
                    pass
            hcsg_command_dicts.append(cur_hcsg_commands)
        reversed_ind = len(sa_programs) - 1
        for cur_hcsg_commands in hcsg_command_dicts[::-1]:
            cur_hcsg_program = []
            for cube_type, commands in cur_hcsg_commands:
                if cube_type == 0:
                    cur_hcsg_program.extend(commands)
                else:
                    # Plug in the "complete" program from hcsg
                    hcsg_to_plug = self.final_hcsg_exprs[cube_type]
                    commands = hcsg_to_plug
                    cur_hcsg_program.extend(commands)
            cur_hcsg_program.insert(0, {'type': "B", "symbol": "union", "canvas_count": len(cur_hcsg_commands)})
            self.final_hcsg_exprs[reversed_ind] = cur_hcsg_program
            reversed_ind -= 1
        cur_hcsg = self.final_hcsg_exprs[0]
        # cur_hcsg.insert(0, {'type': "T", "symbol": "rotate", "param": [0, 90, 0]})
        self.hcsg_program = cur_hcsg

    def _csg_complexity(self, reset=True):

        max_complexity = 2
        self.transformed_coords = [[0]]
        for command in self.hcsg_program:
            c_type = command['type']
            c_symbol = command['symbol']
            if c_type == "B":
                # print("creating Boolean", command)
                canvas_count = command['canvas_count'] 
                # print("adding B  with count %d" % canvas_count)
                boolean_command = self.boolean_to_execute[c_symbol]
                self.boolean_stack.append(boolean_command)
                self.boolean_start_index.append(self.canvas_pointer)
                # Also Clone the top of the transform codes:
                latest_coords_set = self.transformed_coords[-1]
                for j in range(canvas_count-1):
                    cloned_coords_set = [0 for x in latest_coords_set]
                    self.transformed_coords.append(cloned_coords_set)
                self.bool_canvas_req.append(canvas_count)
            elif c_type == "T":
                # print("creating Transform", command)
                pass
            elif c_type == "D":
                # print("creating Draw", command)
                latest_coords_set = self.transformed_coords.pop()
                new_canvas_set = [0 for x in latest_coords_set]
                self.canvas_stack.append(new_canvas_set)
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
            ## Housekeeping:
            if len(self.transformed_coords) == 0:
                self.transformed_coords.append([0])
            ## Apply Boolean if possible - multiple times: 
            self.complexity_boolean_resolve()
            n_canvas_2 = np.sum([len(x) for x in self.transformed_coords])
            complexity = max(n_canvas, n_canvas_2)
            max_complexity = max(max_complexity, complexity)
            
        if reset:
            self.reset()
        return max_complexity
        

    def complexity_boolean_resolve(self):
        if self.boolean_stack:
            # print("checking Boolean stack")
            while(self.canvas_pointer >= self.boolean_start_index[-1] + self.bool_canvas_req[-1]):
                # print("state", self.canvas_pointer, self.boolean_start_index)
                cur_req = self.bool_canvas_req.pop()
                _ = self.boolean_start_index.pop()
                boolean_op = self.boolean_stack.pop()
                c_sets = []
                for j in range(cur_req):
                    c_sets.append(self.canvas_stack.pop())
                # This should be done in batch
                new_canvas_set = [0 for args in zip(*c_sets)]
                # print("applying", boolean_op)
                self.canvas_stack.append(new_canvas_set)
                if len(self.boolean_start_index) == 0:
                    # print("all bool reqs fulfilled")
                    break


    def _csg_compile(self, command_list, reset=True, add_splicing_info=False):
        # Here these are MCSG commands, except there is a bool end token.
        for command in command_list:
            c_type = command['type']
            c_symbol = command['symbol']
            if c_type == "B":
                canvas_count = command['canvas_count'] 
                # print("adding B  with count %d" % canvas_count)
                boolean_command = self.boolean_to_execute[c_symbol]
                self.boolean_stack.append(boolean_command)
                self.boolean_start_index.append(self.canvas_pointer)
                # Also Clone the top of the transform codes:
                latest_coords_set = self.transformed_coords[-1]
                for j in range(canvas_count-1):
                    cloned_coords_set = [x.clone() for x in latest_coords_set]
                    self.transformed_coords.append(cloned_coords_set)
                self.bool_canvas_req.append(canvas_count)
            elif c_type == "T":
                # print("adding T %s" % c_symbol, command['param'])
                self.apply_transform(transform_command=self.transform_to_execute[c_symbol], param=command['param'])
            elif c_type == "D":
                # print("adding D")
                self.apply_draw(draw_command=self.draw_to_execute[c_symbol])
            ## Housekeeping:
            if len(self.transformed_coords) == 0:
                self.transformed_coords.append([self.draw.base_coords.clone()])
            ## Apply Boolean if possible - multiple times: 
            self.boolean_resolve()
            
        self.check_errors(command_list)

        self._output = self.canvas_stack.pop()[0]


    def check_errors(self, command_list):

        if len(self.canvas_stack) > 1:
            print("canvas stack has %d items instead 0f 1" % len(self.canvas_stack))
            print("Commands", command_list)
            raise ValueError
        if len(self.boolean_stack) > 0:
            print("boolean stack has %s items instead of 0" % len(self.boolean_stack))
            print("Commands", command_list)
            raise ValueError

    def boolean_resolve(self):
        if self.boolean_stack:
            # print("checking Boolean stack")
            while(self.canvas_pointer >= self.boolean_start_index[-1] + self.bool_canvas_req[-1]):
                # print("state", self.canvas_pointer, self.boolean_start_index)
                cur_req = self.bool_canvas_req.pop()
                _ = self.boolean_start_index.pop()
                boolean_op = self.boolean_stack.pop()
                c_sets = []
                for j in range(cur_req):
                    c_sets.append(self.canvas_stack.pop())
                # This should be done in batch
                new_canvas_set = [boolean_op(*args) for args in zip(*c_sets)]
                # print("applying", boolean_op)
                self.canvas_stack.append(new_canvas_set)
                if len(self.boolean_start_index) == 0:
                    # print("all bool reqs fulfilled")
                    break

    def per_level_check(self, command_bundle, min_unique_voxles, reset=True):
        if reset:
            self.reset()
            self._sa_compile(command_bundle)
        else:
            if not self.hcsg_program:
                self.reset()
                self._sa_compile(command_bundle)
        # now check
        return self.fuse_check(self.hcsg_program, min_unique_voxles)
    
    def fuse_check(self, command_list, min_unique_voxles):
        valid = True
        for command in command_list:
            c_type = command['type']
            c_symbol = command['symbol']
            if c_type == "B":
                canvas_count = command['canvas_count'] 
                # print("adding B  with count %d" % canvas_count)
                boolean_command = self.boolean_to_execute[c_symbol]
                self.boolean_stack.append(boolean_command)
                self.boolean_start_index.append(self.canvas_pointer)
                # Also Clone the top of the transform codes:
                latest_coords_set = self.transformed_coords[-1]
                for j in range(canvas_count-1):
                    cloned_coords_set = [x.clone() for x in latest_coords_set]
                    self.transformed_coords.append(cloned_coords_set)
                self.bool_canvas_req.append(canvas_count)
            elif c_type == "T":
                # print("adding T %s" % c_symbol, command['param'])
                self.apply_transform(transform_command=self.transform_to_execute[c_symbol], param=command['param'])
            elif c_type == "D":
                # print("adding D")
                self.apply_draw(draw_command=self.draw_to_execute[c_symbol])
            ## Housekeeping:
            if len(self.transformed_coords) == 0:
                self.transformed_coords.append([self.draw.base_coords.clone()])
            ## Apply Boolean if possible - multiple times: 
            valid = self.fuse_check_boolean_resolve(min_unique_voxles)
            if not valid:
                break
        
        self.reset()
        return valid

    def fuse_check_boolean_resolve(self, min_unique_voxels):
        valid = True
        if self.boolean_stack:
            # print("checking Boolean stack")
            while(self.canvas_pointer >= self.boolean_start_index[-1] + self.bool_canvas_req[-1]):
                # print("state", self.canvas_pointer, self.boolean_start_index)
                cur_req = self.bool_canvas_req.pop()
                _ = self.boolean_start_index.pop()
                boolean_op = self.boolean_stack.pop()
                c_sets = []
                for j in range(cur_req):
                    c_sets.append(self.canvas_stack.pop())

                # This should be done in batch
                output = th.stack([x[0] for x in c_sets], -1)
                exp_voxels = (output <= 0.)
                flat_voxels = exp_voxels.float().sum(dim=-1).unsqueeze(-1)
                num = th.sum(((flat_voxels == 1.) & exp_voxels).float(), (0, 1, 2))
                num = num.min().item()
                if len(self.boolean_stack) == 1:
                    # its a subprogram:
                    cur_min_value = min_unique_voxels / 4
                else:
                    cur_min_value = min_unique_voxels
                valid = (num >= cur_min_value)

                new_canvas_set = [boolean_op(*args) for args in zip(*c_sets)]
                # print("applying", boolean_op)
                self.canvas_stack.append(new_canvas_set)
                if not valid:
                    # print("Found min unique voxel", num)
                    break
                if len(self.boolean_start_index) == 0:
                    # print("all bool reqs fulfilled")
                    break
        return valid


    def mirror_merge(self):
        raise ValueError("Not allowed in ShapeAssembly")

    def apply_rotate_draw(self, draw_command, param):
        raise ValueError("Not allowed in Shape Assembly")
    
    def apply_mirror(self, param, add_splicing_info=False):
        raise ValueError("Not allowed in Shape Assembly")

    def write_to_scad(self, command_list, file_name):
        self._sa_compile(command_list)
        command_list = self.final_hcsg_exprs[0]
        super(SACompiler, self).write_to_scad(command_list, file_name)
    
    def write_to_stl(self, command_list, file_name):
        self._sa_compile(command_list)
        command_list = self.final_hcsg_exprs[0]
        super(SACompiler, self).write_to_stl(command_list, file_name)
    
    def march_to_ply(self, command_list, file_name):
        ## Since its all rotated: 
        self._sa_compile(command_list)
        command_list = self.final_hcsg_exprs[0]
        super(SACompiler, self).march_to_ply(command_list, file_name)
    
    def write_to_gltf(self, command_list, file_name):
        self._sa_compile(command_list)
        command_list = self.final_hcsg_exprs[0]
        super(SACompiler, self).write_to_gltf(command_list, file_name)
        
    
