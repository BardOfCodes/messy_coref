import numpy as np
import os
import torch as th
from .parser_utils import boolean_commands, transform_commands, draw_commands, macro_commands, fixed_macro_commands, mcsg_get_expression
from .constants import (TRANSLATE_RANGE_MULT, SCALE_RANGE_MULT, ROTATE_RANGE_MULT, ROTATE_MULTIPLIER, SCALE_ADDITION, SCALE_MULTIPLIER,
                        TRANSLATE_MIN, TRANSLATE_MAX, ROTATE_MIN, ROTATE_MAX, SCALE_MIN, SCALE_MAX, CONVERSION_DELTA, DRAW_MIN, DRAW_MAX)

class MCSG3DParser():
    """ Expression to Command List.
    Eventually, a single macro might lead to multiple commands here. 
    """
    def __init__(self, module_path="", device="cuda"):
        
        self.module_path = module_path
        self.fixed_macros = None
        self.device = device
        self.tensor_type = th.float32

        self.load_language_specific_details()
        self.clip_dict = {
            "sphere": (DRAW_MIN, DRAW_MAX),
            "cylinder": (DRAW_MIN, DRAW_MAX),
            "cuboid": (DRAW_MIN, DRAW_MAX),
            # "infinite_cylinder": (DRAW_MIN, DRAW_MAX),
            # "infinite_cone": (DRAW_MIN, DRAW_MAX),
            "translate": (TRANSLATE_MIN, TRANSLATE_MAX),
            "rotate": (ROTATE_MIN, ROTATE_MAX),
            "quat_rotate": (-1, 1),
            "scale": (SCALE_MIN, SCALE_MAX),
            "union": 0,
            "intersection": 0,
            "difference": 0,
            "mirror": (TRANSLATE_MIN, TRANSLATE_MAX),
        }
        
    
    def load_language_specific_details(self):
        self.command_n_param = {
            "sphere": 0,
            "cylinder": 0,
            "cuboid": 0,
            # "infinite_cylinder": 0,
            # "infinite_cone": 0,
            "translate": 3,
            "rotate": 3,
            "quat_rotate": 4,
            "scale": 3,
            "union": 0,
            "intersection": 0,
            "difference": 0,
            "mirror": 3,
            # For FCSG
            "rotate_sphere": 3,
            "rotate_cylinder": 3,
            "rotate_cuboid": 3,
        }
        self.command_symbol_to_type = {
            "sphere": "D",
            "cylinder": "D",
            "cuboid": "D",
            # "infinite_cylinder": "D",
            # "infinite_cone": "D",
            "translate": "T",
            "quat_rotate": "T",
            "rotate": "T",
            "scale": "T",
            "union": "B",
            "intersection": "B",
            "difference": "B",
            "mirror": "M",
            "rotate_sphere": "RD",
            "rotate_cylinder": "RD",
            "rotate_cuboid": "RD",
        }
        self.mirror_params = {
            "MIRROR_X": [1., 0., 0.],
            "MIRROR_Y": [0., 1., 0.],
            "MIRROR_Z": [0., 0., 1.],
        }
        self.invalid_commands = []

        self.trivial_expression = ["sphere", "$"]
        self.has_transform_commands = True
        self.load_fixed_macros("box_edges.mcsg")

        self.tensor_type = th.float32
        # Can/should it do mcsg parsing -> NO

    def set_device(self, device):
        self.device = device

    def set_tensor_type(self, tensor_type):
        self.tensor_type = tensor_type
        
    def set_to_half(self):
        self.tensor_type = th.float16

    def set_to_full(self):
        self.tensor_type = th.float32

    def set_to_cuda(self):
        self.device = "cuda"
    
    def set_to_cpu(self):
        self.device = "cpu"
        
    def load_fixed_macros(self, file_name):
        
        macro_location = os.path.join(self.module_path, "rl_csg/CSG/env/csg3d/fixed_macros/")
        box_edges = open(os.path.join(macro_location, file_name), "r").readlines()
        box_edges = [x.strip() for x in box_edges]
        command_list = self.parse(box_edges)
        for ind, cmd in enumerate(command_list):
            if ind == 0:
                cmd["macro_mode"] = "macro(BOX_EDGES)"
            else:
                cmd["macro_mode"] = None
        self.fixed_macros = {
            "BOX_EDGES": command_list,
        }
        # Fixed mirror commands
        if not "mirror" in self.invalid_commands:
            command_type = "M"
            command_symbol = "mirror"
            for key, value in self.mirror_params.items():
                command = [{'type': command_type, "symbol": command_symbol, 'param': value, 'macro_mode': "macro(%s)" % key}]
                self.fixed_macros[key] = command
            
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
                    param = np.array([float(x.strip()) for x in param_str.split(",")])
                    command_dict['param'] = param
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
                    param = np.array([float(x.strip()) for x in param_str.split(",")])
                    noise_vector = np.random.uniform(1 - noise_rate, 1 + noise_rate, size = param.shape)
                    param = param * noise_vector
                    clip_min, clip_max = self.clip_dict[command_symbol]
                    param = np.clip(param, clip_min, clip_max)
                    command_dict['param'] = param
                command_list.append(command_dict)
        return command_list
        
    def check_parsability(self, expression_list):
        for invalid in self.invalid_commands:
            present = any(invalid in expr for expr in expression_list)
            if present:
                print("expression", expression_list)
                raise ValueError("Invalid command %s." % invalid)
        for expr in expression_list:
            if "macro" not in expr and "#" not in expr and "$" not in expr:
                c_sym = expr.split("(")[0]
                n_param = self.command_n_param[c_sym]
                if n_param > 0:
                    param_str = expr.split("(")[1][:-1]
                    param = [float(x.strip()) for x in param_str.split(",")]
                    real_n_param = len(param)
                    if n_param != real_n_param:
                        raise ValueError("%s requires %d parameters but has %d parameters." % (expr, n_param, real_n_param))

        return True
    
    def copy_command_list(self, command_list):
        new_command_list = []
        for command_dict in command_list:
            new_dict = {
                'type': command_dict['type'],
                'symbol': command_dict['symbol']
            }
            if 'macro_mode' in command_dict.keys():
                new_dict['macro_mode'] = command_dict['macro_mode']
            if 'param' in command_dict.keys():
                param = command_dict['param']
                if isinstance(param, th.Tensor):
                    new_param = param.clone()
                else:
                    new_param = param.copy()
                new_dict['param'] = new_param
            new_command_list.append(new_dict)
        return new_command_list


    def get_expression(self, command_list, clip=True, quantize=False, resolution=32):
        expr = mcsg_get_expression(command_list, clip, quantize, resolution)
        return expr


    def get_indented_expression(self, expression_list):

        expr_list = []
        stack_state = 0
        canvas_state = 0
        boolean_stack = []
        boolean_start_index = []
        for expr in expression_list:
            command_symbol = expr.split("(")[0]
            if command_symbol == "macro":
                expr_str = "\t" * stack_state + expr + "\n"
                stack_state += 1
                expr_list.append(expr_str)
            elif command_symbol == "$":
                # END OF PROGRAM
                break
            else:
                command_type = self.command_symbol_to_type[command_symbol]
                if "(" in expr:
                    parsed_expr = expr.split("(")
                    floats = parsed_expr[1][:-1].split(",")
                    floats = [f"{float(x):.3f}" for x in floats]
                    expr = parsed_expr[0] + "(" + ",".join(floats) + ")"
                expr_str = "\t" * stack_state + expr + "\n"
                expr_list.append(expr_str)
                if command_type == "B":
                    stack_state += 1
                    boolean_stack.append(stack_state)
                    boolean_start_index.append(canvas_state)
                elif command_type == "M":
                    stack_state += 1
                elif command_type == "D":
                    canvas_state += 1
                    state_update = True
                    boolean_update = True
                    while(state_update or boolean_update):
                        state_update = False
                        boolean_update = False
                        if boolean_stack:
                            while(stack_state > boolean_stack[-1]):
                                stack_state -= 1
                                state_update = True
                            if (canvas_state >= boolean_start_index[-1] + 2):
                                canvas_state -=1
                                stack_state -= 1
                                boolean_start_index.pop()
                                boolean_stack.pop()
                                boolean_update = True
                        else:
                            while(stack_state > 0):
                                stack_state -= 1
                                state_update = True
        return expr_list
    
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
                    variable, param_th = self.create_normalized_tensors(param, command_symbol)
                    variable_list.append(variable)
                    command_dict['param'] = param_th
                command_list.append(command_dict)
        return command_list, variable_list

    def create_normalized_tensors(self, param, command_symbol):
        
        clip_min, clip_max = self.clip_dict[command_symbol]
        param = np.clip(param, clip_min, clip_max)
        mul, extra = self.get_variable_transform_param(command_symbol)
        variable = th.atanh((th.tensor(param, device=self.device, dtype=self.tensor_type) - extra)/ mul)
        # if add_noise:
        #     variable = variable + th.randn(variable.shape) * noise_rate
        variable = th.autograd.Variable(variable, requires_grad=True)
        param = th.tanh(variable) * mul + extra
        return variable, param
    
    
    def csgstump_create_normalized_tensors(self, param, command_symbol):
        
        variable = th.tensor(param, device=self.device, dtype=self.tensor_type)
        variable = th.autograd.Variable(variable, requires_grad=True)
        param = th.tanh(variable) # * mul + extra
        return variable, param
    
    def get_variable_transform_param(self, command_symbol):

        mul = 1
        extra = 1e-9
        if command_symbol == 'rotate':
            mul = ROTATE_MULTIPLIER
        elif command_symbol == 'scale':
            extra = SCALE_ADDITION
            mul = SCALE_MULTIPLIER
        return mul, extra
    
    def csgstump_rebuild_command_list(self, expression_list, variable_list, *args):
        
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
                    variable = variable_list[count]
                    mul, extra = self.get_variable_transform_param(command_symbol)
                    param = variable # th.tanh(variable) * mul + extra
                    command_dict['param'] = param
                    count += 1
                command_list.append(command_dict)

        return command_list
    
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
                    variable = variable_list[count]
                    mul, extra = self.get_variable_transform_param(command_symbol)
                    param = th.tanh(variable) * mul + extra
                    command_dict['param'] = param
                    count += 1
                command_list.append(command_dict)

        return command_list
    

    def get_random_transforms(self, valid_transforms=transform_commands, max_count=-1, min_count=0, bbox=None):

        if max_count < 0:
            max_count = len(valid_transforms)
        transform_count = np.round(np.random.sample() *  max_count).astype(np.int32)
        transform_count = max(min_count, transform_count)
        random_transforms = np.random.choice(valid_transforms, replace=False, size=transform_count)
        
        transform_expr = []
        for transform in random_transforms:
            expr, bbox = self.get_transform(transform, bbox)
            transform_expr.extend(expr)
            
        return transform_expr
    
    def get_transform(self, transform, bbox):
        rand_var = np.random.beta(2, 3, size=3)
        if transform == "translate":
            sign = np.random.choice([-1, 1], size=3)
            rand_var = rand_var * sign
            parameters =  rand_var * TRANSLATE_RANGE_MULT
            if not bbox is None:
                max_translate = 1 - bbox[1]
                min_translate = -1 - bbox[0]
                parameters = min_translate + parameters * (max_translate - min_translate)
                bbox += parameters
        elif transform == "scale":
            rand_var = rand_var * SCALE_RANGE_MULT * 2
            parameters = rand_var + SCALE_ADDITION - 1

            if not bbox is None:
                abs_max = np.max(np.abs(bbox), 0)
                size = np.abs(bbox[0] - bbox[1]) 
                min_scale = np.maximum(0.1, 0.025/(size + 1e-9))
                max_scale = np.minimum(1.0, 1 / (abs_max + 1e-9))
                parameters = min_scale + parameters * max_scale
                bbox *= parameters
        elif transform == "rotate":
            # SKIP BBOX adjustment
            sign = np.random.choice([-1, 1], size=3)
            rand_var = rand_var * sign
            parameters = rand_var * ROTATE_RANGE_MULT
        expr = transform + "(%f, %f, %f)" % tuple(parameters)
        expr = [expr]
        
        return expr, bbox

        
    def sample_random_primitive(self, valid_draws=draw_commands, valid_transforms=transform_commands):
        max_transform_count = len(valid_transforms)
        primitive_expr = self.sample_only_primitive(valid_draws)
        transform_expr = self.get_random_transforms(valid_transforms, max_transform_count, min_count=1)
        final_expr = transform_expr + primitive_expr
        return final_expr
    
    
    def get_mirror_transform(self, bbox):
        if not bbox is None:
            bbox_center = (bbox[1] - bbox[0]) + 1e-9
            parameters = list(bbox_center)
        else:
            rand_var = np.random.beta(2, 3, size=3)
            sign = np.random.choice([-1, 1], size=3)
            parameters = rand_var * sign

        expr = "mirror" + "(%f, %f, %f)" % tuple(parameters)
        expr = [expr]
        
        return expr, None
    

    def get_macro_mirror(self, bbox=None):
        mirrors = ["MIRROR_X", "MIRROR_Y", "MIRROR_Z"]
        if not bbox is None:
            valid = [(bbox[0, i] > -0.1) or (bbox[1, i] < 0.1) for i in range(3)]
            valid_mirrors = [x for ind, x in enumerate(mirrors) if valid[ind]]
            if len(valid_mirrors) > 0:
                primitive_symbol = np.random.choice(valid_mirrors)
                expr = "macro(%s)" % primitive_symbol
                expr = [expr]
            else:
                expr = []
        else:
            primitive_symbol = np.random.choice(mirrors)
            expr = "macro(%s)" % primitive_symbol
            expr = [expr]
        return expr, None
    
    # This will be there only in ? None.
    def sample_only_primitive(self, valid_draws):
        primitive_symbol = np.random.choice(valid_draws)
        primitive_expr = [primitive_symbol]
        return primitive_expr
    
    def convert_to_mcsg(self, expr):
        return expr