
import copy
from doctest import master
from turtle import update
from weakref import finalize
from CSG.env.csg3d.graph_compiler import GraphicalMCSG3DCompiler
from .draw import SADiffDraw3D
import torch as th
import networkx as nx
import numpy as np
from .sa_wrapper import SAProgram
from CSG.env.csg3d.compiler_utils import get_reward
import re
from .compiler_utils import _create_command_list_from_cuboid, get_params_from_sa

class GraphicalSACompiler(GraphicalMCSG3DCompiler):

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
        self.transform_to_execute['rotate_with_matrix'] = self.draw.rotate_with_matrix
        self.inverse_transform['rotate_with_matrix'] = self.draw.shape_rotate_with_matrix

        self.transform_to_execute['reflect'] = self.draw.mirror_coords
        self.inverse_transform['reflect'] = self.draw.shape_mirror

        self.hierarchical_state = []
        self.varied_bbox_mode = varied_bbox_mode

        self.threshold = 0.025
        self.threshold_diff = 0.05
        self.fcsg_mode = True
        self.mode = "3D"
        self.denoising_filter = None
    def command_tree(self, command_list, subprog_info=None, target=None, reset=True, 
                              enable_subexpr_targets=False, create_sweeping_graph=True, add_splicing_info=True, add_sa_subprogram=False):
        if create_sweeping_graph:
            hcsg_commands = self._sa_sweep_compile(command_list)
            out = self._hcsg_command_tree(hcsg_commands, target, reset, enable_subexpr_targets, add_splicing_info, add_sa_subprogram=add_sa_subprogram)
        elif add_sa_subprogram:
            hcsg_commands, master_program_struct = self._sa_splice_compile(command_list, subprog_info, include_canonical_shape=True)
            out = master_program_struct

        return out


    def _sa_splice_compile(self, sa_programs, subprog_info, include_canonical_shape=False):
        hcsg_command_dicts = []
        for ind, cur_prog in enumerate(sa_programs[:1]):
            # Convert "cur program to form"
            macroed_cuboids = []
            TP = SAProgram(device=self.device, dtype=self.tensor_type)
            for cur_command in cur_prog:
                sa_command = cur_command['type']
                param = cur_command['param']
                TP.execute_with_params(sa_command, param)
                if sa_command in ["sa_reflect", "sa_translate"]:
                    macroed_cuboids.append(param[0])
                if sa_command == "sa_cuboid":
                    if param[4] > 0:
                        sel_program = subprog_info[param[4]]
                        TP.cuboids[param[0]].has_subprogram = True
                        TP.cuboids[param[0]].subpr_sa_commands = sel_program['subpr_sa_commands'].copy()
                        TP.cuboids[param[0]].subpr_hcsg_commands = sel_program['subpr_hcsg_commands'].copy()
                        TP.cuboids[param[0]].sa_action_length = sel_program['sa_action_length']
                        TP.cuboids[param[0]].canonical_shape = sel_program['canonical_shape']
            self.sa_executed_progs[ind] = TP

            cur_hcsg_commands = []
            cube_struct = dict()
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
                # hcsg_cube_commands = _create_command_list_from_cuboid(cube_param)
                hcsg_cube_commands[-1]['cube_name'] = cuboid.name
                hcsg_cube_commands[-1]['has_subprogram'] = cuboid.has_subprogram
                    
                hcsg_cube_commands[-1]['splicable'] = True
                hcsg_cube_commands[-1]['parent'] = None
                hcsg_cube_commands[-1]['siblings'] = []
                for macroed_name in macroed_cuboids:
                    if macroed_name in name:
                        if not name == macroed_name:
                            hcsg_cube_commands[-1]['splicable'] = False
                            hcsg_cube_commands[-1]['parent'] = name
                            cube_struct[macroed_name]['siblings'].append(name)

                cur_hcsg_commands.append((cuboid_type, hcsg_cube_commands))


                cur_cube_struct = {
                    "cube_name": name,
                    "has_subprogram": cuboid.has_subprogram,
                    "subpr_sa_commands": cuboid.subpr_sa_commands,
                    "subpr_hcsg_commands": cuboid.subpr_hcsg_commands,
                    "sa_action_length": cuboid.sa_action_length,
                    "splicable": hcsg_cube_commands[-1]['splicable'],
                    "parent": hcsg_cube_commands[-1]['parent'],
                    "siblings": hcsg_cube_commands[-1]['siblings'],
                }
                if include_canonical_shape:
                    cur_cube_struct["canonical_shape"] = cuboid.canonical_shape

                cube_struct[name] = cur_cube_struct

                
                
            hcsg_command_dicts.append(cur_hcsg_commands)

        # Now start completing the tree in reverse order
        reversed_ind = 0
        for cur_hcsg_commands in hcsg_command_dicts[::-1]:
            # same algo -> convert
            cur_hcsg_program = []
            for cube_type, commands in cur_hcsg_commands:
                # if cube_type == 0:
                    cur_hcsg_program.extend(commands)
            cur_hcsg_program.insert(0, {'type': "B", "symbol": "union", "canvas_count": len(cur_hcsg_commands)})
            # cur_hcsg_program[0]['sa_subprogram'] = sa_programs[0]
            self.final_hcsg_exprs[reversed_ind] = cur_hcsg_program
            reversed_ind -= 1
        cur_hcsg = self.final_hcsg_exprs[0]
        # cur_hcsg.insert(0, {'type': "T", "symbol": "rotate", "param": [0, 90, 0]})
        self.hcsg_program = cur_hcsg
        master_program_struct = {
            "master_hcsg_commands": sa_convert_hcsg_commands_to_cpu(cur_hcsg),
            "cube_dict": cube_struct, # the list of program cs_helper structs,
            "master_sa_commands": subprog_info[0]["subpr_sa_commands"], # the sa_program structure (with parameters).  
            "master_sa_action_length": subprog_info[0]["master_sa_action_length"], # the sa_program structure (with parameters).  
        }

        return cur_hcsg, master_program_struct

    def hcsg_program_replace(self, master_hcsg_program, cube_name, hcsg_commands):
        for ind, command in enumerate(master_hcsg_program):
            if command['type'] == "D":
                if "cube_name" in command.keys():
                    if command['cube_name'] == cube_name:
                        break
        new_master_command = master_hcsg_program[:ind] + hcsg_commands + master_hcsg_program[ind+1:]
        return new_master_command

    def master_struct_to_sa_and_hcsg(self, master_struct):


        master_sa_program = master_struct['master_sa_commands']# .copy()
        master_hcsg_program = copy.deepcopy(master_struct['master_hcsg_commands'])
        cube_dict = master_struct['cube_dict']

        final_sa_program = []

        subpr_to_add = []

        for line in master_sa_program:
            if "cuboid(" in line:
                cube_name = line.split("=")[0].strip()
                if cube_name == "bbox":
                    final_sa_program.append(line)
                else:
                    cube_info = cube_dict[cube_name]
                    if cube_info['has_subprogram']:
                        sa_subprogram = cube_info["subpr_sa_commands"]
                        hcsg_commands = cube_info["subpr_hcsg_commands"]
                        hcsg_commands[0]['cube_name'] = cube_name 
                        master_hcsg_program = self.hcsg_program_replace(master_hcsg_program, cube_name, hcsg_commands)
                        subpr_to_add.append(sa_subprogram)

                        line = line[:-2] + "%d)" % (len(subpr_to_add))
                        final_sa_program.append(line)
                        # check if there are reflects:
                        siblings = cube_info["siblings"]
                        for cur_cube_name in siblings:
                            hcsg_commands = copy.deepcopy(cube_info["subpr_hcsg_commands"])
                            hcsg_commands[0]['cube_name'] = cur_cube_name
                            master_hcsg_program = self.hcsg_program_replace(master_hcsg_program, cur_cube_name, hcsg_commands)
                    else:
                        # Check if it is a reflect:
                        sa_subprogram, hcsg_commands = [], []
                        line = line[:-2] + "0)"
                        final_sa_program.append(line)
            else:
                final_sa_program.append(line)
        
        if subpr_to_add:
            for subpr in subpr_to_add:
                final_sa_program[-1] = "$"
                final_sa_program.extend(subpr)
        
        return final_sa_program, master_hcsg_program
        


    def _sa_sweep_compile(self, sa_programs):
        hcsg_command_dicts = []
        head_start = 0
        for program_ind, cur_prog in enumerate(sa_programs):
            # Convert "cur program to form"
            # print("starting program", program_ind)
            TP = SAProgram(device=self.device, dtype=self.tensor_type)
            start_index = 0
            end_index = start_index + 1
            prev_box_name = 'bbox'
            for ind, cur_command in enumerate(cur_prog):
                sa_command = cur_command['type']
                param = cur_command['param']
                TP.execute_with_params(sa_command, param)
                if sa_command == "sa_cuboid" and param[0] != 'bbox':
                    end_index = ind
                    if not prev_box_name == "bbox":
                        prev_cube = TP.cuboids[prev_box_name]
                        prev_cube.command_ids = (head_start + start_index, head_start + end_index)
                        prev_cube.program_ind = program_ind
                        # print("For cuboid %s setting inds" % prev_box_name, start_index, end_index)
                    start_index = ind
                    prev_box_name = param[0]
                elif sa_command == "sa_reflect":
                    # mark the reflected for only the corresponding line
                    prev_cube = TP.cuboids[prev_box_name]
                    # new_cube_name = prev_box_name + "_ref_%d" % (prev_cube.ref_counter - 1)
                    new_cube_name = prev_box_name + "_ref_%s_%d" % (param[1], prev_cube.ref_counter - 1)
                    new_cube = TP.cuboids[new_cube_name]
                    new_cube.command_ids = (head_start + ind, head_start + ind + 1)
                    new_cube.program_ind = program_ind
                    new_cube.macro_parent = "program_%d_%s" % (program_ind, prev_box_name)# 
                    prev_cube.macro_list.append(["program_%d_%s" % (program_ind, new_cube_name)])
                    # print("For cuboid %s setting inds" % new_cube_name, ind, ind + 1)
                elif sa_command == "sa_translate":
                    prev_cube = TP.cuboids[prev_box_name]
                    prev_cube.has_macro = True
                    N = param[2]
                    new_cube_list = []
                    for j in range(1, N+1):
                        new_cube_name = prev_box_name + "_trans_%d_%d" % (prev_cube.trans_counter - 1, j)
                        new_cube = TP.cuboids[new_cube_name]
                        new_cube.command_ids = (head_start + ind, head_start + ind + 1)
                        new_cube.macro_parent = "program_%d_%s" % (program_ind, prev_box_name)# 
                        new_cube.program_ind = program_ind
                        new_cube_list.append("program_%d_%s" % (program_ind, new_cube_name))
                    prev_cube.macro_list.append(new_cube_list)
                        # print("For cuboid %s setting inds" % new_cube_name, ind, ind + 1)
            end_index = np.sum([len(prog) for pind, prog in enumerate(sa_programs) if pind <= program_ind]) + (program_ind)
            # print("cur end Index", end_index)
            prev_cube = TP.cuboids[prev_box_name]
            prev_cube.command_ids = (head_start + start_index, end_index)
            prev_cube.program_ind = program_ind
            head_start = end_index + 1
            start_index = ind
            # print("For cuboid %s setting inds" % prev_box_name, start_index, end_index)
                        


            self.sa_executed_progs[program_ind] = TP

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
                hcsg_cube_commands[-1]['name'] = "program_%d_%s" % (cuboid.program_ind, name)
                hcsg_cube_commands[-1]['sa_command_ids'] = cuboid.command_ids

                move_atts = cuboid.move_atts
                atts = cuboid.attachments
                not_move = [x for x in atts if x not in move_atts]
                # hcsg_cube_commands[-1]['incoming_attachments'] = not_move
                hcsg_cube_commands[-1]['macro_parent'] = cuboid.macro_parent
                hcsg_cube_commands[-1]['macro_list'] = cuboid.macro_list
                if len(not_move) > 0:
                    hcsg_cube_commands[-1]['deletable'] = False
                else:
                    hcsg_cube_commands[-1]['deletable'] = True


                cur_hcsg_commands.append((cuboid_type, hcsg_cube_commands))

            hcsg_command_dicts.append(cur_hcsg_commands)

        # Now start completing the tree in reverse order
        reversed_ind = len(sa_programs) - 1
        for cur_hcsg_commands in hcsg_command_dicts[::-1]:
            # same algo -> convert
            cur_hcsg_program = []
            updated_commands = dict()
            for cube_type, commands in cur_hcsg_commands:
                draw_command = commands[-1]
                if cube_type != 0:
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
                    if draw_command['macro_list']:
                        for cmd in hcsg_to_plug:
                            # print("in hcsg to plug", cmd)
                            if cmd['type'] == "D":
                                cmd['deletable'] = False
                    commands = commands[:-1] + conversion + hcsg_to_plug
                updated_commands[draw_command['name']] = commands
                # Now for macros:
                
            while(len(updated_commands.values())>0):
                cur_program_name = list(updated_commands.keys())[0]
                cur_command = updated_commands[cur_program_name]

                n_draws = len([x for x in cur_command if x['type'] == "D"])
                if not n_draws == 1:
                    # this is a extended program - can't do anything
                    cur_hcsg_program.append(cur_command)
                    updated_commands.pop(cur_program_name)
                else:
                    draw_command = cur_command[-1] 
                    if draw_command['macro_list']:
                        # Has macros:
                        prior_delete_state = draw_command['deletable']
                        draw_command['deletable'] = False

                        new_commands = cur_command
                        macros = draw_command['macro_list']
                        non_extended_commands = {x:y for x, y in updated_commands.items() if len([c for c in y if c['type'] == "D"]) == 1}
                        for cur_macros in macros:
                            all_cmds = []
                            for cur_name in cur_macros:
                                macro_cmd = [y for x, y in non_extended_commands.items() if x == cur_name]
                                all_cmds.extend(macro_cmd)
                                updated_commands.pop(cur_name)
                            if len(all_cmds) == 1:
                                # just add the single
                                new_commands.extend(all_cmds[0])
                            else:
                                # its a translate: singular delete as impossible
                                unioned_cmd = []
                                for cmd in all_cmds:
                                    cmd[-1]['deletable'] = False
                                    unioned_cmd.extend(cmd)
                                union_command = dict(type="B", symbol="union", canvas_count=len(all_cmds))
                                union_command['deletable'] = prior_delete_state
                                union_command['sa_command_ids'] = all_cmds[0][-1]['sa_command_ids']
                                unioned_cmd.insert(0, union_command)
                                new_commands.extend(unioned_cmd)
                        union_command = dict(type="B", symbol="union", canvas_count=len(macros)+1)
                        union_command['deletable'] = prior_delete_state
                        union_command['sa_command_ids'] = draw_command['sa_command_ids']
                        new_commands.insert(0, union_command)
                        cur_hcsg_program.append(new_commands)
                        updated_commands.pop(cur_program_name)
                    else:
                        cur_hcsg_program.append(cur_command)
                        updated_commands.pop(cur_program_name)
                        # collect all with the same name 
            n_canvas = len(cur_hcsg_program)
            cur_hcsg_program = [item for sublist in cur_hcsg_program for item in sublist]
            cur_hcsg_program.insert(0, {'type': "B", "symbol": "union", "canvas_count": n_canvas, 'deletable': False})
            self.final_hcsg_exprs[reversed_ind] = cur_hcsg_program
            reversed_ind -= 1
        cur_hcsg = self.final_hcsg_exprs[0]
        # cur_hcsg.insert(0, {'type': "T", "symbol": "rotate", "param": [0, 90, 0]})
        self.hcsg_program = cur_hcsg
        return cur_hcsg



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

    def _hcsg_command_tree(self, command_list, target=None, reset=True, 
                              enable_subexpr_targets=False, add_splicing_info=False, add_sa_subprogram=False):

        self.bool_canvas_req = []

        graph = nx.DiGraph()
        counter = 0
        default_info = dict(type="ROOT", symbol="START", parent=None, children=[])
        if add_splicing_info:
            default_info["subexpr_info"] = dict()
        graph.add_node(counter, node_id=counter, **default_info)
        cur_parent_ind = counter
        self.non_terminal_expr_start_idx = []
        self.bool_command_stack = []

        for ind, command in enumerate(command_list):
            c_type = command['type']
            c_symbol = command['symbol']
            counter +=1
            if c_type == "B":
                canvas_count = command['canvas_count'] 
                # print("adding B  with count %d" % canvas_count)
                boolean_command = self.boolean_to_execute[c_symbol]
                self.boolean_stack.append(boolean_command)
                self.boolean_start_index.append(self.canvas_pointer)
                # Also Clone the top of the transform codes:
                latest_coords_set = self.transformed_coords[-1]
                if add_splicing_info:
                    cur_expr_canvas = self.draw.base_coords.clone()
                    latest_coords_set.append(cur_expr_canvas)
                for j in range(canvas_count-1):
                    cloned_coords_set = [x.clone() for x in latest_coords_set]
                    self.transformed_coords.append(cloned_coords_set)
                self.bool_canvas_req.append(canvas_count)

                self.draw_node(graph, cur_parent_ind, counter, command)
                cur_parent_ind = counter
                self.non_terminal_expr_start_idx.append(ind)

                self.bool_command_stack.append(command)
                graph.nodes[counter]['bool_requirement'] = canvas_count

                
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
                # print("drawing %s with parent %d" %(command["name"], cur_parent_ind))
                # print("parent Children", len(graph.nodes[cur_parent_ind]['children']))
                cur_parent_ind = self.get_next_parent_after_draw(graph, cur_parent_ind)
                # Subexpr Stuff:
                if add_splicing_info:
                    cur_expr_canvas = self.draw.base_coords.clone()
                    new_canvas = self.draw_to_execute[c_symbol](coords=cur_expr_canvas)
                    deletable = command['deletable']
                    sa_command_ids = command['sa_command_ids']
                    # print(sa_command_ids)
                    self.update_subexpr_info(graph, new_canvas, ind, ind + 1, command_list, counter, deletable=deletable, 
                                        sa_command_ids=sa_command_ids)
            
            elif c_type == "DUMMY":
                assert add_splicing_info, "Only for Splicing"
                latest_coords_set = self.transformed_coords.pop()
                # Make a blank canvas
                new_canvas_set = [x.norm(dim=-1) + 1e-9 for x in latest_coords_set]
                self.canvas_stack.append(new_canvas_set)
                self.draw_node(graph, cur_parent_ind, counter, command)
                graph.nodes[counter]['master_y_lims'] = command["master_y_lims"]
                graph.nodes[counter]['cube_name'] = command["cube_name"]

                cur_parent_ind = self.get_next_parent_after_draw(graph, cur_parent_ind)

                cur_expr_canvas = self.draw.base_coords.clone()
                new_canvas = cur_expr_canvas.norm(dim=-1)+ 1e-9 
                self.update_subexpr_info(graph, new_canvas, ind, ind + 1, command_list, counter, dummy=True,)

                
            ## Housekeeping:
            if len(self.transformed_coords) == 0:
                self.transformed_coords.append([self.draw.base_coords.clone()])
            
            ## Apply Boolean if possible - multiple times: 
            self.tree_boolean_resolve(graph, target, command_list, ind, enable_subexpr_targets, add_splicing_info)
            
        self.check_errors(command_list)
        
        self._output = self.canvas_stack.pop()[0]
        if enable_subexpr_targets:
        # Now annotate the targets and target masks:
            self.create_subexpr_targets(graph, target)
            
        return graph
        
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


    def get_next_parent_after_draw(self, graph, cur_parent_ind):

        cur_parent = graph.nodes[cur_parent_ind]
        inverted_parent_pointer = -1
        # print("starting draw search with %d" % cur_parent_ind)
        while(True):
            cur_parent_ind = cur_parent['node_id']
            cur_parent_type = cur_parent['type']
            # print("cur parent Type %s" % cur_parent_type, cur_parent_ind)
            cur_siblings = cur_parent['children']
            if cur_parent_type == "B":
                # may do something:

                n_sibling = len(cur_siblings)
                if n_sibling < self.bool_canvas_req[inverted_parent_pointer]:
                    # print("Boolean has 1 child now. This works!")
                    break
                elif n_sibling == self.bool_canvas_req[inverted_parent_pointer]:
                    # print("Boolean has 2 children now; Going further up.")
                    inverted_parent_pointer -= 1
                    pass
                else:
                    print("WUT?", n_sibling, self.bool_canvas_req)
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

    def update_subexpr_info(self, graph, cur_canvas, expr_start, expr_end, command_list, graph_id, dummy=False, sa_command_ids=None, deletable=False):

        super(GraphicalSACompiler, self).update_subexpr_info(graph, cur_canvas, expr_start, expr_end, command_list, graph_id, dummy)
        graph.nodes[graph_id]['sa_info'] = {}
        graph.nodes[graph_id]['sa_info']['command_ids'] = sa_command_ids
        graph.nodes[graph_id]['sa_info']['deletable'] = deletable
        

    def tree_boolean_resolve(self, graph, target, command_list, ind, enable_subexpr_targets=False, add_splicing_info=False):
        # assert not (add_sweeping_info and add_splicing_info), "CANNOT MAKE GRAPH FOR SWEEPING AND SPLICING TOGETHER."
        ## Apply Boolean if possible - multiple times: 
        if self.boolean_stack:
            # print("checking Boolean stack")
            while(self.canvas_pointer >= self.boolean_start_index[-1] + self.bool_canvas_req[-1]):
                # We can apply the operation!
                cur_req = self.bool_canvas_req.pop()
                boolean_op = self.boolean_stack.pop()
                _ = self.boolean_start_index.pop()
                c_sets = []
                for j in range(cur_req):
                    c_sets.append(self.canvas_stack.pop())
                c_sets = c_sets[::-1]
                new_canvas_set = [boolean_op(*args) for args in zip(*c_sets)]
                self.canvas_stack.append(new_canvas_set)
                # For tree:
                # Update the node with information
                boolean_command_idx = self.non_terminal_expr_start_idx.pop()
                boolean_graph_idx = boolean_command_idx + 1
                if add_splicing_info:
                    cur_bool_canvas = new_canvas_set.pop() 
                    expr_start = boolean_command_idx
                    expr_end = ind +1
                    bool_command = self.bool_command_stack.pop()
                    deletable = bool_command['deletable']
                    if deletable:
                        sa_command_ids = bool_command["sa_command_ids"]
                    else:
                        sa_command_ids = None
                    self.update_subexpr_info(graph, cur_bool_canvas, expr_start, expr_end, command_list, boolean_graph_idx,
                                             deletable=deletable, sa_command_ids=sa_command_ids)
                    # cur_bool_canvas, expr_start, expr_end, command_list, graph_id
                    if enable_subexpr_targets:
                        # set targets:
                        other_childs = []
                        if cur_req == 1:
                            graph.nodes[boolean_graph_idx]['subexpr_info']["other_childs"] = [(c_sets[0][0] * 0).bool()]
                        else:
                            for j in range(cur_req):
                                cur_set = [x for ind, x in enumerate(c_sets) if ind != j]
                                new_children_canvas = [boolean_op(*args) for args in zip(*cur_set)]
                                new_children_canvas = (new_children_canvas[0] <= 0)
                                other_childs.append(new_children_canvas)
                            graph.nodes[boolean_graph_idx]['subexpr_info']["other_childs"] = other_childs

                if len(self.boolean_start_index) == 0:
                    # print("all bool reqs fulfilled")
                    break
            
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
                bool_req = cur_node['bool_requirement']
                # Save the node as the target:
                # perform inverse boolean on each and add to stack.
                child_canvases = cur_node['subexpr_info']["other_childs"]
                final_targets = []
                n_children = len(child_canvases)
                for ind in range(n_children):
                    child_target = self.inverse_boolean[c_symbol](cur_target, child_canvases[ind], 0)
                    final_targets.append(child_target)
                for target in final_targets[::-1]:
                    self.target_stack.append(target)
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

            elif node_type == "D":
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
                masked_target = th.logical_and(cur_target[:,:,:,0], cur_target[:,:,:,1]).float()
                floating_target_pseudo_sdf = -(masked_target-0.1)
                bbox, hcsg_command_list, sa_command_list = self.get_dummy_bbox_and_commands(cur_node, floating_target_pseudo_sdf)
                cur_node["dummy_sa_commands"] = sa_command_list
                cur_node["dummy_hcsg_commands"] = hcsg_command_list
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

    def get_dummy_bbox_and_commands(self, cur_node, floating_target_pseudo_sdf):
        bbox = self.draw.return_bounding_box(floating_target_pseudo_sdf, normalized=False)
                # Extend y for finding the spot to connect
        cur_bbox = bbox.copy()
        normalized_bbox = (-1 + (cur_bbox + 0.5)/self.draw.grid_divider)
        center = normalized_bbox.mean(0)
        cur_bbox[1] += 1
        size_normalized_bbox = (-1 + (cur_bbox + 0.5)/self.draw.grid_divider)
        size = size_normalized_bbox[1] - size_normalized_bbox[0]
        center_y = center[1]
        y_lims_hcsg = cur_node["master_y_lims"]
        if not abs(center_y) <= y_lims_hcsg:
                    # Problem! Have to adjust y
            if center_y - size[1] > cur_node["master_y_lims"]:
                        # needs adjustment
                cur_min_y = center_y - size[1]/2
                diff = cur_min_y - cur_node["master_y_lims"]
                center_y -= diff
                size[1] += 2 * diff
            elif center_y + size[1] < cur_node["master_y_lims"]:
                cur_max_y = center_y + size[1]
                diff = cur_node["master_y_lims"] - cur_max_y
                center_y += diff
                size[1] += 2 * diff
                # reform bbox:
        center[1] = center_y
        bbox = np.array([[center[0] - size[0]/2, center[1] - size[1]/2, center[2] - size[2]/2],
                         [center[0] + size[0]/2, center[1] + size[1]/2 , center[2] + size[2]/2]])
        bbox = (bbox + 1) * self.draw.grid_divider
        bbox[1] -= 1.0
        bbox = np.uint8(bbox)

        translate_dict = dict(type="T", symbol="translate", param=center)
        scale_dict = dict(type="T", symbol="scale", param=size)
        rotate_dict = dict(type="T", symbol="rotate_with_matrix", param=[1, 0, 0, 0, 1, 0, 0, 0, 1])
        draw_dict = dict(type="D", symbol="cuboid", cube_name=cur_node['cube_name'], 
                         has_subprogram=False, splicable=True, parent=None, siblings=[])
        
        hcsg_command_list = [translate_dict, rotate_dict, scale_dict, draw_dict]

        attach_center = center / 2 + 0.5
        sa_command_list = [
            "%s = cuboid(%f, %f, %f, 0)" % (cur_node['cube_name'], size[0]/2, size[1]/2, size[2]/2),
            "attach(%s, bbox, 0.5, 0.5, 0.5, %f, %f, %f)" % (cur_node['cube_name'], attach_center[0], attach_center[1], attach_center[2])
        ]

        return bbox, hcsg_command_list, sa_command_list

def sa_convert_hcsg_commands_to_cpu(command_list):
    hcsg_commands = []
    for cmd in command_list:
        new_cmd = dict()
        for key, value in cmd.items():
            if isinstance(value, th.Tensor):
                value =list(value.cpu().numpy())
            new_cmd[key] = value
        hcsg_commands.append(new_cmd)
    return hcsg_commands