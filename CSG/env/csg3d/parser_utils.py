
import numpy as np
import torch as th
from .constants import TRANSLATE_MIN, TRANSLATE_MAX, ROTATE_MIN, ROTATE_MAX, SCALE_MIN, SCALE_MAX, ROTATE_MULTIPLIER, SCALE_ADDITION, CONVERSION_DELTA

transform_commands = ['translate', 'rotate', 'scale']
transform_commands_sans_scale = ['translate', 'rotate']
draw_commands = ['sphere', 'cylinder', 'cuboid']
# draw_commands = ['sphere', 'cylinder', 'cuboid', "infinite_cylinder", "infinite_cone"]
boolean_commands = ['union', 'intersection', 'difference']
macro_commands = ['mirror']
fixed_macro_commands = ['macro']

def ntcsg_to_pcsg(expression_list):
    expression_list = [x.replace("ellipsoid", "sphere") for x in expression_list]
    return expression_list

def pcsg_to_ntcsg(expression_list):
    expression_list = [x.replace("sphere", "ellipsoid") for x in expression_list]
    return expression_list

# Temporary HACK
def fcsg_to_mcsg(expression_list, transform_sequence=["translate", "scale",  "rotate"]):
    mcsg_expr = []
    for expr in expression_list:
        command_symbol = expr.split("(")[0]
        if command_symbol in boolean_commands: 
            mcsg_expr.append(expr)
        elif command_symbol in draw_commands:
            # This has to be split into three commands:
            param_str = expr.split("(")[1][:-1]
            param = [float(x.strip()) for x in param_str.split(",")]
            for ind, transform in enumerate(transform_sequence):
                sel_param = param[ind *3: (ind+1) * 3]
                command = "%s(%f, %f, %f)" % (transform, sel_param[0], sel_param[1], sel_param[2])
                mcsg_expr.append(command)
            mcsg_expr.append(command_symbol)
        elif command_symbol == "macro":
            pass
        elif "#" in command_symbol:
            # its a comment pass.
            pass
        elif command_symbol == "$":
            # its a comment pass.
            break
    return mcsg_expr

def mcsg_commands_to_lower_expr(command_list, fcsg_mode=True, clip=True, quantize=False, resolution=32):
    
    scale_stack = [np.array([1, 1, 1])]
    translate_stack = [np.array([0, 0, 0])]
    rotate_stack = [np.array([0, 0, 0])]
    lower_expr = []

    conversion_delta = CONVERSION_DELTA
    if quantize:
        two_scale_delta = (2 - 2 * conversion_delta)/(resolution - 1)
    for command in command_list:
        if 'macro_mode' in command.keys():
            expr = command['macro_mode']
            if not expr is None:
                lower_expr.append(expr)
        else:
            c_type = command['type']
            c_symbol = command['symbol']
            
            if c_type == "B":
                lower_expr.append(c_symbol)
                cloned_scale = np.copy(scale_stack[-1])
                scale_stack.append(cloned_scale)
                cloned_translate = np.copy(translate_stack[-1])
                translate_stack.append(cloned_translate)
                cloned_rotate = np.copy(rotate_stack[-1])
                rotate_stack.append(cloned_rotate)
                
            elif c_type == "T":
                param = command['param']
                if isinstance(param, th.Tensor):
                    param = param.detach().cpu().data.numpy()
                if c_symbol == "scale":
                    cur_scale = scale_stack.pop()
                    new_param = cur_scale * param
                    scale_stack.append(new_param)
                elif c_symbol == "translate":
                    cur_scale = scale_stack[-1]
                    cur_translate = translate_stack.pop()
                    new_translate = cur_scale * param + cur_translate
                    translate_stack.append(new_translate)
                elif c_symbol == "rotate":
                    cur_rotate = rotate_stack.pop()
                    new_rotate = cur_rotate + param
                    rotate_stack.append(new_rotate)
            elif c_type == "D":
                t_p = translate_stack.pop()
                s_p = scale_stack.pop()
                if clip:
                    t_p = np.clip(t_p, TRANSLATE_MIN + conversion_delta, TRANSLATE_MAX - conversion_delta)
                    s_p = np.clip(s_p, SCALE_MIN + conversion_delta, SCALE_MAX - conversion_delta)
                if fcsg_mode: 
                    r_p = rotate_stack.pop()
                    if clip: r_p = np.clip(r_p, ROTATE_MIN + conversion_delta, ROTATE_MAX - conversion_delta)
                    param = np.concatenate([t_p, s_p, r_p], 0)
                else:
                    param = np.concatenate([t_p, s_p], 0)
                
                if quantize:
                    param[3:6] -= SCALE_ADDITION
                    if fcsg_mode: param[6:9] /= ROTATE_MULTIPLIER
                    param = (param - (-1 + conversion_delta)) / two_scale_delta
                    param = np.round(param)
                    param = (param * two_scale_delta) + (-1 + conversion_delta)
                    param[3:6] += SCALE_ADDITION
                    if fcsg_mode: param[6:9] *=  ROTATE_MULTIPLIER

                param_str = ", ".join(["%f"%x for x in param])
                draw_expr = "%s(%s)" %(c_symbol, param_str)
                lower_expr.append(draw_expr)
    lower_expr.append("$") 
    return lower_expr
    


def mcsg_get_expression(command_list, clip=True, quantize=False, resolution=32):
    expression_list = []
    for command_dict in command_list:
        if 'macro_mode' in command_dict.keys():
            expr = command_dict['macro_mode']
        else:
            expr = command_dict['symbol']
            if 'param' in command_dict.keys():
                param = command_dict['param']
                if isinstance(param, th.Tensor):
                    param = param.detach().cpu().data.numpy()
                if expr == "translate":
                    min, max = (TRANSLATE_MIN, TRANSLATE_MAX)
                    mul, add = (1, 0)
                elif expr == "scale":
                    min, max = (SCALE_MIN, SCALE_MAX)
                    mul, add = (1, SCALE_ADDITION)
                elif expr == "rotate":
                    min, max = (ROTATE_MIN, ROTATE_MAX)
                    mul, add = (ROTATE_MULTIPLIER, 0)
                elif expr == "quat_rotate":
                    min, max = (-1, 1)
                    mul, add = (1, 0)
                elif expr == "mirror":
                    min, max = (TRANSLATE_MIN, TRANSLATE_MAX)
                    mul, add = (1, 0)
                    
                if clip: param = np.clip(param, min, max)
                if quantize: param = quantize_param(param, resolution, mul, add)
                param_str = ", ".join(["%f"%x for x in param])
                expr += "(" + param_str + ")"
        if not expr is None:
            expression_list.append(expr)
    expression_list.append("$")
    return expression_list

def quantize_param(param, resolution, mul, add):
    conversion_delta = CONVERSION_DELTA
    two_scale_delta = (2 - 2 * conversion_delta)/(resolution - 1)
    param -= add
    param /= mul
    param = (param - (-1 + conversion_delta)) / two_scale_delta
    param = np.round(param)
    param = (param * two_scale_delta) + (-1 + conversion_delta)
    param += add
    param *= mul
    return param
