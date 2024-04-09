import os


REFLECT_PARAMS = {
    "X": [1., 0., 0.],
    "Y": [0., 1., 0.],
    "Z": [0., 0., 1.],
}

def _create_command_list_from_cuboid(param, reflect, reflect_dir):
    t_p, s_p, r_p = get_params_from_sa(param)
    translate_dict = dict(type="T", symbol="translate", param=t_p)
    scale_dict = dict(type="T", symbol="scale", param=s_p)
    rotate_dict = dict(type="T", symbol="rotate_with_matrix", param=r_p)
    draw_dict = dict(type="D", symbol="cuboid")
    command_list = [translate_dict, rotate_dict, scale_dict, draw_dict]
    if reflect:
        mirror_param = REFLECT_PARAMS[reflect_dir]
        mirror_command = dict(type="T", symbol="reflect", param=mirror_param)
        command_list.insert(3, mirror_command)

    return command_list

def get_params_from_sa(param):
    t_p = 2 * param[3:6]
    s_p = (param[:3])
    r_p = param[6:15]
    return t_p, s_p, r_p