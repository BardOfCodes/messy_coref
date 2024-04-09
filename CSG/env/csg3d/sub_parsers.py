from .parser import MCSG3DParser, draw_commands, boolean_commands, transform_commands
from .parser_utils import mcsg_commands_to_lower_expr, fcsg_to_mcsg
from .constants import TRANSLATE_RANGE_MULT, SCALE_RANGE_MULT, ROTATE_RANGE_MULT
from .constants import TRANSLATE_MIN, TRANSLATE_MAX, ROTATE_MIN, ROTATE_MAX, SCALE_MIN, SCALE_MAX, ROTATE_MULTIPLIER, SCALE_ADDITION, CONVERSION_DELTA
import os
import torch as th
import numpy as np


class HCSG3DParser(MCSG3DParser):
    """ Does not contain any macros
    """

    def load_language_specific_details(self):
        self.command_n_param = {
            "sphere": 0,
            "cylinder": 0,
            "cuboid": 0,
            "translate": 3,
            "rotate": 3,
            "scale": 3,
            "union": 0,
            "intersection": 0,
            "difference": 0,
        }
        self.command_symbol_to_type = {
            "sphere": "D",
            "cylinder": "D",
            "cuboid": "D",
            "translate": "T",
            "rotate": "T",
            "scale": "T",
            "union": "B",
            "intersection": "B",
            "difference": "B",
        }
        self.invalid_commands = ["mirror"]

        self.trivial_expression = ["sphere", "$"]
        self.load_fixed_macros("box_edges.csg")


class FCSG3DParser(MCSG3DParser):
    """ Does not contain any macros
    """

    def load_language_specific_details(self):
        self.command_n_param = {
            "sphere": 9,
            "cylinder": 9,
            "cuboid": 9,
            "union": 0,
            "intersection": 0,
            "difference": 0,
        }
        self.command_symbol_to_type = {
            "sphere": "D",
            "cylinder": "D",
            "cuboid": "D",
            "union": "B",
            "intersection": "B",
            "difference": "B",
        }
        self.invalid_commands = ["mirror", "translate", "scale", "rotate"]

        self.trivial_expression = [
            "sphere(0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0)", "$"]

        self.transform_sequence = ["translate", "scale", "rotate"]
        self.has_transform_commands = False
        self.fcsg_mode = True
        self.load_fixed_macros("box_edges.fcsg")

    def parse(self, expression_list):
        command_list = []
        for expr in expression_list:
            command_symbol = expr.split("(")[0]
            if command_symbol == "macro":
                macro_name = expr.split("(")[1][:-1]
                command_list.extend(self.fixed_macros[macro_name])
            elif "#" in command_symbol:
                continue
            elif command_symbol == "$":
                # END OF PROGRAM
                break
            else:
                command_type = self.command_symbol_to_type[command_symbol]
                command_dict = {'type': command_type, "symbol": command_symbol}
                n_param = self.command_n_param[command_symbol]
                if n_param > 0:
                    param_str = expr.split("(")[1][:-1]
                    param = np.array([float(x.strip())
                                     for x in param_str.split(",")])
                    # Now convert into MCSG3D
                    for ind, command_symbol in enumerate(self.transform_sequence):
                        transform_dict = {
                            'type': "T", "symbol": command_symbol, "param": param[ind*3: (ind+1)*3]}
                        command_list.append(transform_dict)
                command_list.append(command_dict)
        return command_list

    def noisy_parse(self, expression_list, noise_rate=0.1):
        command_list = []
        for expr in expression_list:
            command_symbol = expr.split("(")[0]
            if command_symbol == "macro":
                macro_name = expr.split("(")[1][:-1]
                command_list.extend(self.fixed_macros[macro_name])
            elif "#" in command_symbol:
                continue
            elif command_symbol == "$":
                # END OF PROGRAM
                break
            else:
                command_type = self.command_symbol_to_type[command_symbol]
                command_dict = {'type': command_type, "symbol": command_symbol}
                n_param = self.command_n_param[command_symbol]
                if n_param > 0:
                    param_str = expr.split("(")[1][:-1]
                    param = np.array([float(x.strip())
                                     for x in param_str.split(",")])
                    noise_vector = np.random.uniform(
                        1 - noise_rate, 1 + noise_rate, shape=param.shape)
                    clip_min, clip_max = self.clip_dict[command_symbol]
                    param = np.clip(param, clip_min, clip_max)
                    param = param * noise_vector
                    # Now convert into MCSG3D
                    for ind, command_symbol in enumerate(self.transform_sequence):
                        transform_dict = {
                            'type': "T", "symbol": command_symbol, "param": param[ind*3: (ind+1)*3]}
                        command_list.append(transform_dict)
                command_list.append(command_dict)
        return command_list

    def differentiable_parse(self, expression_list, add_noise=True, noise_rate=0.0):
        """ TODO: Add the tanh and the unit variable hack.
        """
        command_list = []
        variable_list = []
        for expr in expression_list:
            command_symbol = expr.split("(")[0]
            if command_symbol == "macro":
                macro_name = expr.split("(")[1][:-1]
                command_list.extend(self.fixed_macros[macro_name])
            elif "#" in command_symbol:
                continue
            elif command_symbol == "$":
                # END OF PROGRAM
                break
            else:
                command_type = self.command_symbol_to_type[command_symbol]
                n_param = self.command_n_param[command_symbol]
                command_dict = {'type': command_type, "symbol": command_symbol}
                if n_param > 0:
                    param_str = expr.split("(")[1][:-1]
                    param = [float(x.strip()) for x in param_str.split(",")]
                    # Here we need to noramalize accoding to the language
                    for ind, command_symbol in enumerate(self.transform_sequence):
                        variable, param_th = self.create_normalized_tensors(
                            param[ind*3: (ind+1)*3], command_symbol)
                        variable_list.append(variable)
                        transform_dict = {
                            'type': "T", "symbol": command_symbol, "param": param_th}
                        command_list.append(transform_dict)
                command_list.append(command_dict)
        return command_list, variable_list

    def rebuild_command_list(self, expression_list, variable_list, *args):

        command_list = []
        count = 0
        for expr in expression_list:
            command_symbol = expr.split("(")[0]
            if command_symbol == "macro":
                macro_name = expr.split("(")[1][:-1]
                command_list.extend(self.fixed_macros[macro_name])
            elif "#" in command_symbol:
                continue
            elif command_symbol == "$":
                # END OF PROGRAM
                break
            else:
                command_type = self.command_symbol_to_type[command_symbol]
                n_param = self.command_n_param[command_symbol]
                command_dict = {'type': command_type, "symbol": command_symbol}
                if n_param > 0:
                    # Here we need to noramalize accoding to the language
                    for ind, command_symbol in enumerate(self.transform_sequence):
                        variable = variable_list[count + ind]
                        mul, extra = self.get_variable_transform_param(
                            command_symbol)
                        param = th.tanh(variable) * mul + extra
                        transform_dict = {
                            'type': "T", "symbol": command_symbol, "param": param}
                        command_list.append(transform_dict)
                    if self.fcsg_mode:
                        count += 3
                    else:
                        count += 2
                command_list.append(command_dict)
        return command_list

    def get_expression(self, command_list, clip=True, quantize=False, resolution=32):
        fcsg_expression = mcsg_commands_to_lower_expr(command_list, fcsg_mode=self.fcsg_mode,
                                                    clip=clip, quantize=quantize, resolution=resolution)
        return fcsg_expression

    def convert_to_mcsg(self, expression):
        mcsg_expression = fcsg_to_mcsg(expression, self.transform_sequence)
        return mcsg_expression

    def get_random_transforms(self, *args, **kwargs):
        raise ValueError("Not valid for this language")

    def get_transform(self, *args, **kwargs):
        raise ValueError("Not valid for this language")

    def sample_random_primitive(self, valid_draws=draw_commands, *args, **kwargs):
        return self.sample_only_primitive(valid_draws)

    def sample_only_primitive(self, valid_draws=draw_commands):
        valid_draws = [
            x for x in valid_draws if not x in self.invalid_commands]
        primitive_symbol = np.random.choice(valid_draws)
        n_param = len(self.transform_sequence) * 3
        rand_var = np.random.beta(2, 3, size=3)
        sign = np.random.choice([-1, 1], size=3)
        rand_var = rand_var * sign
        translate_parameters = rand_var * TRANSLATE_RANGE_MULT
        rand_var = np.random.beta(2, 3, size=3)
        rand_var = rand_var * SCALE_RANGE_MULT * 2
        scale_parameters = rand_var + SCALE_ADDITION - 1

        if "rotate" in self.transform_sequence:
            rand_var = np.random.beta(2, 3, size=3)
            sign = np.random.choice([-1, 1], size=3)
            rand_var = rand_var * sign
            rotate_parameters = rand_var * ROTATE_RANGE_MULT
            parameters = np.concatenate(
                [translate_parameters, scale_parameters, rotate_parameters], 0)
        else:
            parameters = np.concatenate(
                [translate_parameters, scale_parameters], 0)

        param_str = ", ".join(["%f" % x for x in parameters])
        draw_expr = ["%s(%s)" % (primitive_symbol, param_str)]

        return draw_expr


class PCSG3DParser(FCSG3DParser):
    """ Does not contain any macros
    """

    def load_language_specific_details(self):
        self.command_n_param = {
            "sphere": 6,
            "cuboid": 6,
            "union": 0,
            "intersection": 0,
            "difference": 0,
        }
        self.command_symbol_to_type = {
            "sphere": "D",
            "cuboid": "D",
            "union": "B",
            "intersection": "B",
            "difference": "B",
        }
        self.invalid_commands = [
            "mirror", "translate", "scale", "rotate", "cylinder"]

        self.trivial_expression = ["sphere(0, 0, 0, 0.1, 0.1, 0.1)", "$"]

        self.transform_sequence = ["translate", "scale"]

        self.has_transform_commands = False
        self.fcsg_mode = False
        self.load_fixed_macros("box_edges.pcsg")



class RotationFCSG3DParser(FCSG3DParser):
    """ Does not contain any macros
    """


    def parse(self, expression_list):
        command_list = []
        for expr in expression_list:
            command_symbol = expr.split("(")[0]
            if command_symbol == "macro":
                macro_name = expr.split("(")[1][:-1]
                command_list.extend(self.fixed_macros[macro_name])
            elif "#" in command_symbol:
                continue
            elif command_symbol == "$":
                # END OF PROGRAM
                break
            else:
                command_type = self.command_symbol_to_type[command_symbol]
                command_dict = {'type': command_type, "symbol": command_symbol}
                n_param = self.command_n_param[command_symbol]
                if n_param > 0:
                    param_str = expr.split("(")[1][:-1]
                    param = np.array([float(x.strip())
                                     for x in param_str.split(",")])
                    # Now convert into MCSG3D
                    for ind, command_symbol in enumerate(self.transform_sequence[:2]):
                        transform_dict = {
                            'type': "T", "symbol": command_symbol, "param": param[ind*3: (ind+1)*3]}
                        command_list.append(transform_dict)
                    command_dict['type'] = "RD"
                    command_dict['param'] = param[2*3: (3)*3]
                    command_dict['symbol'] = "rotate_%s" % command_dict['symbol']
                command_list.append(command_dict)
        return command_list

    def convert_to_mcsg(self, expression):
        mcsg_expr = []
        for expr in expression:
            command_symbol = expr.split("(")[0]
            if command_symbol in boolean_commands: 
                mcsg_expr.append(expr)
            elif command_symbol in draw_commands:
                # This has to be split into three commands:
                param_str = expr.split("(")[1][:-1]
                param = [float(x.strip()) for x in param_str.split(",")]
                for ind, transform in enumerate(self.transform_sequence[:2]):
                    sel_param = param[ind *3: (ind+1) * 3]
                    command = "%s(%f, %f, %f)" % (transform, sel_param[0], sel_param[1], sel_param[2])
                    mcsg_expr.append(command)
                # Now add the rotation and change the draw
                transform = self.transform_sequence[2]
                sel_param = param[2 *3: (3) * 3]
                command = "%s_%s(%f, %f, %f)" % (transform, command_symbol, sel_param[0], sel_param[1], sel_param[2])
                mcsg_expr.append(command)
            elif command_symbol == "macro":
                pass
            elif "#" in command_symbol:
                # its a comment pass.
                pass
            elif command_symbol == "$":
                # its a comment pass.
                break
        return mcsg_expr
    
    def get_expression(self, command_list, clip=True, quantize=False, resolution=32):
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
                        raise ValueError("Rotates should be inside Rotate Draws")
                elif c_type == "D":
                    t_p = translate_stack.pop()
                    s_p = scale_stack.pop()
                    if clip:
                        t_p = np.clip(t_p, TRANSLATE_MIN + conversion_delta, TRANSLATE_MAX - conversion_delta)
                        s_p = np.clip(s_p, SCALE_MIN + conversion_delta, SCALE_MAX - conversion_delta)
                    r_p = rotate_stack.pop()
                    if clip: r_p = np.clip(r_p, ROTATE_MIN + conversion_delta, ROTATE_MAX - conversion_delta)
                    param = np.concatenate([t_p, s_p, r_p], 0)
                    
                    if quantize:
                        param[3:6] -= SCALE_ADDITION
                        param[6:9] /= ROTATE_MULTIPLIER
                        param = (param - (-1 + conversion_delta)) / two_scale_delta
                        param = np.round(param)
                        param = (param * two_scale_delta) + (-1 + conversion_delta)
                        param[3:6] += SCALE_ADDITION
                        param[6:9] *=  ROTATE_MULTIPLIER

                    param_str = ", ".join(["%f"%x for x in param])
                    draw_expr = "%s(%s)" %(c_symbol, param_str)
                    lower_expr.append(draw_expr)
                elif c_type == "RD":
                    param = command['param']
                    cur_rotate = rotate_stack.pop()
                    new_rotate = cur_rotate + param
                    rotate_stack.append(new_rotate)
                    t_p = translate_stack.pop()
                    s_p = scale_stack.pop()
                    if clip:
                        t_p = np.clip(t_p, TRANSLATE_MIN + conversion_delta, TRANSLATE_MAX - conversion_delta)
                        s_p = np.clip(s_p, SCALE_MIN + conversion_delta, SCALE_MAX - conversion_delta)
                    r_p = rotate_stack.pop()
                    if clip: r_p = np.clip(r_p, ROTATE_MIN + conversion_delta, ROTATE_MAX - conversion_delta)
                    param = np.concatenate([t_p, s_p, r_p], 0)
                    
                    if quantize:
                        param[3:6] -= SCALE_ADDITION
                        param[6:9] /= ROTATE_MULTIPLIER
                        param = (param - (-1 + conversion_delta)) / two_scale_delta
                        param = np.round(param)
                        param = (param * two_scale_delta) + (-1 + conversion_delta)
                        param[3:6] += SCALE_ADDITION
                        param[6:9] *=  ROTATE_MULTIPLIER
                    param_str = ", ".join(["%f"%x for x in param])
                    c_symbol = c_symbol.split("_")[1]
                    draw_expr = "%s(%s)" %(c_symbol, param_str)
                    lower_expr.append(draw_expr)
        lower_expr.append("$") 
        return lower_expr

