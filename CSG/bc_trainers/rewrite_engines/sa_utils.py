import copy
import torch as th


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

def get_master_program_struct(temp_env, base_parser, compiler, graph_compiler, expression):
    command_list = base_parser.parse(expression)
    subprogram_dict = base_parser.get_all_subprograms(expression)
    subprog_info = []
    for subpr in subprogram_dict:
        sub_command_list = base_parser.parse(subpr)
        compiler._compile(sub_command_list)
        output_shape = (compiler._output <0)
        hcsg_commands = sa_convert_hcsg_commands_to_cpu(compiler.hcsg_program)
        action_len = temp_env.action_space.expression_to_action(subpr).shape[0]
        subprog_info.append({
                            "subpr_sa_commands": subpr,
                            "subpr_hcsg_commands": hcsg_commands,
                            "sa_action_length" : action_len,
                            "canonical_shape": output_shape
                        })
    
    # Correct the master program:
    action_len = temp_env.action_space.expression_to_action(expression).shape[0]
    subprog_info[0]['master_sa_action_length'] = action_len


    master_program_struct = graph_compiler.command_tree(command_list, subprog_info=subprog_info, target=None, 
                                                        create_sweeping_graph=False,
                                                        enable_subexpr_targets=False, add_splicing_info=True,
                                                        add_sa_subprogram=True)

    return master_program_struct


def sa_convert_hcsg_commands_to_cuda(command_list, device, dtype):
    hcsg_commands = []
    for cmd in command_list:
        new_cmd = dict()
        for key, value in cmd.items():
            if key == "param":
                if isinstance(value, list):
                    value =th.tensor(value, device=device, dtype=dtype)
            new_cmd[key] = value
        hcsg_commands.append(new_cmd)
        # Waste keys
        new_cmd['deletable'] = False
        new_cmd['sa_command_ids'] = (0, 1)
    return hcsg_commands


def get_cube_id_from_hcsg(command_list, cube_name):
    found = False
    for ind, command in enumerate(command_list):
        if command['type'] in ["D", "B", "DUMMY"]:
            if "cube_name" in command.keys():
                if command['cube_name'] == cube_name:
                    found = True
                    break
    if not found: 
        raise ValueError("Cound not find node in graph!")
    return ind


def min_copy_master_program_struct(master_struct):

    master_sa_commands = master_struct['master_sa_commands']
    master_hcsg_commands = copy.deepcopy(master_struct['master_hcsg_commands'])
    master_sa_action_length = master_struct["master_sa_action_length"]
    cube_dict = master_struct['cube_dict']

    new_cube_list = dict()
    for cube_name, cur_dict in cube_dict.items():
        new_dict = {
            "cube_name": cube_name,
            "has_subprogram": cur_dict['has_subprogram'],
            "subpr_sa_commands": cur_dict['subpr_sa_commands'].copy(),
            "subpr_hcsg_commands": cur_dict['subpr_hcsg_commands'].copy(),
            "sa_action_length": cur_dict['sa_action_length'],
            "splicable": cur_dict['splicable'],
            "parent": cur_dict['parent'],
            "siblings": cur_dict['siblings'].copy(),
        }
        new_cube_list[cube_name] = new_dict
    
    new_struct = dict(
        master_sa_commands=master_sa_commands,
        master_hcsg_commands=master_hcsg_commands,
        cube_dict=new_cube_list,
        master_sa_action_length=master_sa_action_length
    )
    return new_struct