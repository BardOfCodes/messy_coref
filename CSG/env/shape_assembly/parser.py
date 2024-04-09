"""
Simply convert SA program to a compiler command list
"""
from collections import OrderedDict
from dataclasses import replace
from CSG.env.csg3d.parser import MCSG3DParser
from CSG.env.csg3d.constants import CONVERSION_DELTA
import re
import torch as th
import numpy as np

CONVERSION_DELTA = 1e-3
class SAParser(MCSG3DParser):
    
    def __init__(self, module_path="", device="cuda"):

        self.module_path = module_path
        self.fixed_macros = None
        self.device = device
        self.load_language_specific_details()
        self.trivial_expression = ["bbox = cuboid(1, 0.9, 1, 0)", "cube0 = cuboid(0.5, 0.5, 0.5, 0)", "squeeze(cube0, bbox, bbox, top, 0.5, 0.5)", "$$"]

    def load_fixed_macros(self, file_name):
        pass
    def parse(self, expression_list):
        sa_programs = []
        command_list = []
        for expression in expression_list:
            if "cuboid(" in expression:
                param = self.parse_cuboid_command(expression)
                command = dict(type="sa_cuboid", param=param)
                command_list.append(command)
            elif "attach(" in expression:
                param = self.parse_attach_command(expression)
                command = dict(type="sa_attach", param=param)
                command_list.append(command)
            elif "reflect(" in expression:
                param = self.parse_reflect_command(expression)
                command = dict(type="sa_reflect", param=param)
                command_list.append(command)
            elif "translate(" in expression:
                param = self.parse_translate_command(expression)
                command = dict(type="sa_translate", param=param)
                command_list.append(command)
            elif "squeeze(" in expression:
                param = self.parse_squeeze_command(expression)
                command = dict(type="sa_squeeze", param=param)
                command_list.append(command)
            elif expression == "$$":
                sa_programs.append(command_list)
                # the program has ended. 
                command_list = []
            elif expression == "$":
                sa_programs.append(command_list)
                # the program has ended. 
                command_list = []
        return sa_programs
    
    def get_all_subprograms(self, expression_list):
        sa_programs = []
        command_list = []
        for expression in expression_list:
            if "cuboid(" in expression:
                # remove the 1)?
                expression = expression[:-2] + "0)"
                command_list.append(expression)
            elif "attach(" in expression:
                command_list.append(expression)
            elif "reflect(" in expression:
                command_list.append(expression)
            elif "translate(" in expression:
                command_list.append(expression)
            elif "squeeze(" in expression:
                command_list.append(expression)
            elif expression == "$$":
                command_list.append(expression)
                sa_programs.append(command_list)
                # the program has ended. 
                command_list = []
            elif expression == "$":
                command_list.append("$$")
                sa_programs.append(command_list)
                # the program has ended. 
                command_list = []
        return sa_programs

    def parse_cuboid_command(self, expression):
        s = re.split(r'[()]', expression)
        name = s[0].split("=")[0].strip()
        params = s[1].split(',')
        dim0 = th.tensor(float(params[0]), device=self.device, dtype=self.tensor_type)
        dim1 = th.tensor(float(params[1]), device=self.device, dtype=self.tensor_type)
        dim2 = th.tensor(float(params[2]), device=self.device, dtype=self.tensor_type)
        cuboid_type = int(params[3])
        aligned = False
        param = (name, dim0, dim1, dim2, cuboid_type)
        return param
    
    def parse_attach_command(self, expression):
        s = re.split(r'[()]', expression)
        args = [a.strip() for a in s[1].split(',')]
        return (
            args[0],
            args[1],
            th.tensor(float(args[2]), device=self.device, dtype=self.tensor_type),
            th.tensor(float(args[3]), device=self.device, dtype=self.tensor_type),
            th.tensor(float(args[4]), device=self.device, dtype=self.tensor_type),
            th.tensor(float(args[5]), device=self.device, dtype=self.tensor_type),
            th.tensor(float(args[6]), device=self.device, dtype=self.tensor_type),
            th.tensor(float(args[7]), device=self.device, dtype=self.tensor_type)
        )

    # Parses a reflect command
    def parse_reflect_command(self, expression):
        s = re.split(r'[()]', expression)
        args = [a.strip() for a in s[1].split(',')]
        return (
            args[0],
            args[1],
        )

    # Parses a translate command
    def parse_translate_command(self, expression):
        s = re.split(r'[()]', expression)
        args = [a.strip() for a in s[1].split(',')]
        return (
            args[0],
            args[1],
            int(args[2]),
            th.tensor(float(args[3]), device=self.device, dtype=self.tensor_type),
        )

    # Parses a queeze command
    def parse_squeeze_command(self, expression):
        s = re.split(r'[()]', expression)
        args = [a.strip() for a in s[1].split(',')]
        return (
            args[0],
            args[1],
            args[2],
            args[3],
            th.tensor(float(args[4]), device=self.device, dtype=self.tensor_type),
            th.tensor(float(args[5]), device=self.device, dtype=self.tensor_type),
        )

    # Generally required
    def get_expression(self, command_bundle, clip=True, quantize=False, resolution=32):
        expression_list = []
        sa_programs = command_bundle
        min, max = (CONVERSION_DELTA, 1 - CONVERSION_DELTA)
        one_scale_delta = (1 - 2 * CONVERSION_DELTA)/(resolution - 1)
        for cur_program in sa_programs:
            for command in cur_program:
                command_type = command['type']
                param = command['param']
                if command_type == "sa_cuboid":
                    # Do something to get valid numbers Quantization + resolution etc.
                    n_param = param[1:4] 
                    if isinstance(n_param[0], th.Tensor):
                        n_param = th.stack(n_param).detach().cpu().data.numpy()
                    if clip: n_param = np.clip(n_param, min, max)
                    if quantize: n_param = self.quantize_param(n_param, one_scale_delta)
                    expr = "%s = cuboid(%f, %f, %f, %d)" % (param[0], n_param[0], n_param[1], n_param[2], param[4])
                elif command_type == "sa_attach":
                    # Similar quantization etc.
                    n_param = param[2:8] 
                    if isinstance(n_param[0], th.Tensor):
                        n_param = th.stack(n_param).detach().cpu().data.numpy()
                    if clip: n_param = np.clip(n_param, min, max)
                    if quantize: n_param = self.quantize_param(n_param, one_scale_delta)
                    param_str = ", ".join([str(x) for x in n_param])
                    expr = "attach(%s, %s, %s)" % (param[0], param[1], param_str)
                elif command_type == "sa_reflect":
                    expr = "reflect(%s, %s)" % (param[0], param[1])
                elif command_type == "sa_translate":
                    # Again for the last value:
                    n_param = param[3]
                    if isinstance(n_param, th.Tensor):
                        n_param = np.array([n_param.item()])
                    if clip: n_param = np.clip(n_param, min, max)
                    if quantize: n_param = self.quantize_param(n_param, one_scale_delta)
                    expr = "translate(%s, %s, %d, %f)" % (param[0], param[1], param[2], n_param[0])
                elif command_type == "sa_squeeze":
                    # Again for the last value:
                    n_param = param[4:6] 
                    if isinstance(n_param[0], th.Tensor):
                        n_param = th.stack(n_param).detach().cpu().data.numpy()
                    if clip: n_param = np.clip(n_param, min, max)
                    if quantize: n_param = self.quantize_param(n_param, one_scale_delta)
                    expr = "squeeze(%s, %s, %s, %s, %f, %f)" % (param[0], param[1], param[2], 
                                                                param[3], n_param[0], n_param[1])
                expression_list.append(expr)
            expression_list.append("$")
        expression_list[-1] = "$$"
        return expression_list

    def quantize_param(self, param, one_scale_delta):
        
        param = (param - CONVERSION_DELTA) / one_scale_delta
        param = np.round(param).astype(np.uint32)
        param = CONVERSION_DELTA + (param * one_scale_delta)
        return param

    def check_parsability(self, expression_list):
        # Low priority
        # Program has to have valid structure
        # each program has to be closed. 
        # Lower programs cannot call other programs.
        # Syntax for each command is coherent.

        return True
        

    def differentiable_parse(self, expression_list, add_noise=True, noise_rate=0.0):
        sa_programs = []
        command_list = []
        variable_list = []
        for expression in expression_list:
            if "cuboid(" in expression:
                if "bbox" in expression:
                    if len(sa_programs) == 0:
                        param, variable = self.dif_parse_bbox_command(expression)
                        variable_list.append(variable)
                        
                    else:
                        param = self.parse_cuboid_command(expression)
                else:
                    # Note: The first bbox has to be changed
                    param, variable = self.dif_parse_cuboid_command(expression)
                    variable_list.append(variable)
                command = dict(type="sa_cuboid", param=param)
                command_list.append(command)
            elif "attach(" in expression:
                param, variable = self.dif_parse_attach_command(expression)
                command = dict(type="sa_attach", param=param)
                command_list.append(command)
                variable_list.append(variable)
            elif "reflect(" in expression:
                param = self.parse_reflect_command(expression)
                command = dict(type="sa_reflect", param=param)
                command_list.append(command)
            elif "translate(" in expression:
                param, variable = self.dif_parse_translate_command(expression)
                command = dict(type="sa_translate", param=param)
                command_list.append(command)
                variable_list.append(variable)
            elif "squeeze(" in expression:
                param, variable = self.dif_parse_squeeze_command(expression)
                command = dict(type="sa_squeeze", param=param)
                command_list.append(command)
                variable_list.append(variable)
            elif expression == "$$":
                sa_programs.append(command_list)
                command_list = []
            elif expression == "$":
                sa_programs.append(command_list)
                command_list = []
        return sa_programs, variable_list
    
    def dif_parse_bbox_command(self, expression):
        s = re.split(r'[()]', expression)
        name = s[0].split("=")[0].strip()
        params = s[1].split(',')
        dim0 = th.tensor(float(params[0]), device=self.device)
        dim2 = th.tensor(float(params[2]), device=self.device)
        difparam = np.clip(float(params[1]), CONVERSION_DELTA, 1-CONVERSION_DELTA)
        dim1var = th.atanh(th.tensor((difparam- 0.5) * 2, device=self.device)).float()
        variable = th.autograd.Variable(dim1var, requires_grad=True)
        dim1 = th.tanh(variable) / 2.0 + 0.5
        cuboid_type = int(params[3])
        param = (name, dim0, dim1, dim2, cuboid_type)
        return param, variable

    def dif_parse_cuboid_command(self, expression):
        s = re.split(r'[()]', expression)
        name = s[0].split("=")[0].strip()
        params = s[1].split(',')
        float_param = np.array([float(x) for x in params[:3]])
        difparam = np.clip(float_param, CONVERSION_DELTA, 1-CONVERSION_DELTA)
        dim0var = th.atanh(th.tensor((difparam[0]- 0.5) * 2, device=self.device)).float()
        dim1var = th.atanh(th.tensor((difparam[1] - 0.5) * 2, device=self.device)).float()
        dim2var = th.atanh(th.tensor((difparam[2] - 0.5) * 2, device=self.device)).float()
        variable = th.stack([dim0var, dim1var, dim2var], 0)
        variable = th.autograd.Variable(variable, requires_grad=True)

        dim0 = th.tanh(variable[0]) / 2.0 + 0.5
        dim1 = th.tanh(variable[1]) / 2.0 + 0.5
        dim2 = th.tanh(variable[2]) / 2.0 + 0.5
        cuboid_type = int(params[3])
        param = (name, dim0, dim1, dim2, cuboid_type)
        return param, variable
    
    def dif_parse_attach_command(self, expression):
        s = re.split(r'[()]', expression)
        args = [a.strip() for a in s[1].split(',')]
        float_param = np.array([float(x) for x in args[2:8]])
        difparam = np.clip(float_param, CONVERSION_DELTA, 1-CONVERSION_DELTA)
        pt0var = th.atanh(th.tensor((difparam[0] - 0.5) * 2, device=self.device)).float()
        pt1var = th.atanh(th.tensor((difparam[1] - 0.5) * 2, device=self.device)).float()
        pt2var = th.atanh(th.tensor((difparam[2] - 0.5) * 2, device=self.device)).float()
        pt3var = th.atanh(th.tensor((difparam[3] - 0.5) * 2, device=self.device)).float()
        pt4var = th.atanh(th.tensor((difparam[4] - 0.5) * 2, device=self.device)).float()
        pt5var = th.atanh(th.tensor((difparam[5] - 0.5) * 2, device=self.device)).float()
        variable = th.stack([pt0var, pt1var, pt2var, pt3var, pt4var, pt5var], 0)
        variable = th.autograd.Variable(variable, requires_grad=True)
        param = (
            args[0],
            args[1],
            th.tanh(variable[0]) / 2.0 + 0.5,
            th.tanh(variable[1]) / 2.0 + 0.5,
            th.tanh(variable[2]) / 2.0 + 0.5,
            th.tanh(variable[3]) / 2.0 + 0.5,
            th.tanh(variable[4]) / 2.0 + 0.5,
            th.tanh(variable[5]) / 2.0 + 0.5
        )
        return param, variable

    # Parses a translate command
    def dif_parse_translate_command(self, expression):
        s = re.split(r'[()]', expression)
        args = [a.strip() for a in s[1].split(',')]
        difparam = np.clip(float(args[3]), CONVERSION_DELTA, 1-CONVERSION_DELTA)
        var = th.atanh(th.tensor((difparam - 0.5) * 2, device=self.device)).float()
        variable = th.autograd.Variable(var, requires_grad=True)

        param = (
            args[0],
            args[1],
            int(args[2]),
            th.tanh(variable) / 2.0 + 0.5,
        )
        return param, variable 

    # Parses a queeze command
    def dif_parse_squeeze_command(self, expression):
        s = re.split(r'[()]', expression)
        args = [a.strip() for a in s[1].split(',')]
        float_param = np.array([float(x) for x in args[4:6]])
        difparam = np.clip(float_param, CONVERSION_DELTA, 1-CONVERSION_DELTA)
        pt0var = th.atanh(th.tensor((difparam[0] - 0.5) * 2, device=self.device)).float()
        pt1var = th.atanh(th.tensor((difparam[1] - 0.5) * 2, device=self.device)).float()
        variable = th.stack([pt0var, pt1var], 0)
        variable = th.autograd.Variable(variable, requires_grad=True)

        param = (
            args[0],
            args[1],
            args[2],
            args[3],
            th.tanh(variable[0]) / 2.0 + 0.5,
            th.tanh(variable[1]) / 2.0 + 0.5,
        )
        return param, variable

    # for diff opr
    def copy_command_list(self, sa_programs):
        new_comman_list = []
        new_sa_programs = []
        for command_list in sa_programs:
            for command in command_list:
                c_symbol = command['type']
                if c_symbol == "sa_cuboid":
                    prev_param = command['param']
                    new_param = (prev_param[0], prev_param[1].detach().clone(), 
                                 prev_param[2].detach().clone(), prev_param[3].detach().clone(), prev_param[-1])
                    new_command = {"type": c_symbol, "param": new_param}
                    new_comman_list.append(new_command)
                elif c_symbol == "sa_attach":
                    prev_param = command['param']
                    new_param = (prev_param[0], prev_param[1], prev_param[2].detach().clone(), 
                                 prev_param[3].detach().clone(), prev_param[4].detach().clone(),
                                 prev_param[5].detach().clone(), prev_param[6].detach().clone(),
                                 prev_param[7].detach().clone())
                    new_command = {"type": c_symbol, "param": new_param}
                    new_comman_list.append(new_command)
                elif c_symbol == "sa_reflect":
                    prev_param = command['param']
                    new_command = {"type": c_symbol, "param": prev_param}
                    new_comman_list.append(new_command)
                elif c_symbol == "sa_translate":
                    prev_param = command['param']
                    new_param = (prev_param[0], prev_param[1], prev_param[2], prev_param[3].detach().clone())
                    new_command = {"type": c_symbol, "param": new_param}
                    new_comman_list.append(new_command)
                elif c_symbol == "sa_squeeze":
                    prev_param = command['param']
                    new_param = (prev_param[0], prev_param[1], prev_param[2], prev_param[3], 
                                 prev_param[4].detach().clone(), prev_param[5].detach().clone())
                    new_command = {"type": c_symbol, "param": new_param}
                    new_comman_list.append(new_command)
            new_sa_programs.append(new_comman_list)
            # the program has ended. 
            new_comman_list = []
        return new_sa_programs
    
    def rebuild_command_list(self, expression_list, variable_list, sa_programs):
        counter = 0
        new_comman_list = []
        new_sa_programs = []
        for command_list in sa_programs:
            for command in command_list:
                c_symbol = command['type']
                if c_symbol == "sa_cuboid":
                    prev_param = command['param']
                    if prev_param[0] == "bbox":
                        if len(new_sa_programs) == 0:
                            cur_variable = variable_list[counter]
                            counter += 1
                            dim1 = th.tanh(cur_variable) / 2.0 + 0.5
                            new_param = (prev_param[0], prev_param[1].detach().clone(), dim1, 
                                         prev_param[3].detach().clone(), prev_param[-1])
                            command['param'] = new_param
                    else:
                        cur_variable = variable_list[counter]
                        counter += 1
                        dim0 = th.tanh(cur_variable[0]) / 2.0 + 0.5
                        dim1 = th.tanh(cur_variable[1]) / 2.0 + 0.5
                        dim2 = th.tanh(cur_variable[2]) / 2.0 + 0.5
                        new_param = (prev_param[0], dim0, dim1, dim2, prev_param[-1])
                        command['param'] = new_param
                    new_comman_list.append(command)
                elif c_symbol == "sa_attach":
                    cur_variable = variable_list[counter]
                    counter += 1
                    pt0 = th.tanh(cur_variable[0]) / 2.0 + 0.5
                    pt1 = th.tanh(cur_variable[1]) / 2.0 + 0.5
                    pt2 = th.tanh(cur_variable[2]) / 2.0 + 0.5
                    pt3 = th.tanh(cur_variable[3]) / 2.0 + 0.5
                    pt4 = th.tanh(cur_variable[4]) / 2.0 + 0.5
                    pt5 = th.tanh(cur_variable[5]) / 2.0 + 0.5
                    prev_param = command['param']
                    new_param = (prev_param[0], prev_param[1], pt0, pt1, pt2, pt3, pt4, pt5)
                    command['param'] = new_param
                    new_comman_list.append(command)
                elif c_symbol == "sa_reflect":
                    new_comman_list.append(command)
                elif c_symbol == "sa_translate":
                    cur_variable = variable_list[counter]
                    counter += 1
                    value = th.tanh(cur_variable) / 2.0 + 0.5
                    prev_param = command['param']
                    new_param = (prev_param[0], prev_param[1], prev_param[2], value)
                    command['param'] = new_param
                    new_comman_list.append(command)
                elif c_symbol == "sa_squeeze":
                    cur_variable = variable_list[counter]
                    counter += 1
                    pt0 = th.tanh(cur_variable[0]) / 2.0 + 0.5
                    pt1 = th.tanh(cur_variable[1]) / 2.0 + 0.5
                    prev_param = command['param']
                    new_param = (prev_param[0], prev_param[1], prev_param[2], prev_param[3], 
                                 pt0, pt1)
                    command['param'] = new_param
                    new_comman_list.append(command)
            new_sa_programs.append(new_comman_list)
            # the program has ended. 
            new_comman_list = []
        return new_sa_programs
                
        

    def get_variable_transform_param(self, command_symbol):
        # Use to bound variables in diff Parse -> Used only for the parameters - bounded (0, 1) in SA.
        mul = 1
        extra = 1e-9
        return mul, extra

    def noisy_parse(self, expression_list, noise_rate=0.1):
        # Low priority
        raise ValueError("Not allowed in Shape Assembly")

    def get_indented_expression(self, expression_list):
        # Low priority
        raise ValueError("Not allowed in Shape Assembly")

    def convert_to_mcsg(self, expr):
        raise ValueError("Need compiler to convert - Parser cannot in SA")

    def get_random_transforms(self, *args, **kwargs):
        # We will generator with PLAD code.
        raise ValueError("Not allowed in Shape Assembly")
    
    def get_transform(self, transform, bbox):
        raise ValueError("Not allowed in Shape Assembly")

    def sample_random_primitive(self,  *args, **kwargs):
        raise ValueError("Not allowed in Shape Assembly")
    
    def get_mirror_transform(self, bbox):
        raise ValueError("Not allowed in Shape Assembly")
    
    # This will be there only in ? None.
    def sample_only_primitive(self, valid_draws):
        raise ValueError("Not allowed in Shape Assembly")


# Will be used in tandem with the existing random program generator.
def convert_sa_to_valid_hsa(expressions):
    for ind, expression in enumerate(expressions):
        if "Cuboid(" in expression:
            expression = expression.replace("Cuboid", "cuboid")
            # split_expr = expression.split(")")
            expressions[ind] = expression[:-6] +  "0)"
    if expressions[-1] != "$$":
        expressions.append("$$")
    return expressions

def convert_hsa_to_valid_hsa(expressions):
    expressions = [x.strip() for x in expressions]
    # extract all the different programs:
    all_programs = OrderedDict()
    ind = 0
    master_program = True
    while(ind < len(expressions)):
        expr = expressions[ind]
        if "Assembly" in expr:
            # start of program:
            program_name = expr.split(" ")[1]
            jind = ind + 1
            while(jind < len(expressions)):
                temp_expr = expressions[jind]
                if "}" in temp_expr:
                    break
                else:
                    jind += 1
            all_programs[program_name] = expressions[ind +1:jind]
        ind = jind +1
    # now replace the indices in all programs:
    # only for master program:
    all_exprs = []
    for program_key, expressions in all_programs.items():
        for ind, expression in enumerate(expressions):
            if "Cuboid(" in expression:
                expression = expression.replace("Cuboid", "cuboid")
                # check if name is in ind, if it is, then apply it
                cube_name = expression.split(" ")[0]
                if cube_name in all_programs.keys():
                    level = int(cube_name.split("_")[1])
                else:
                    level = 0
                # split_expr = expression.split(")")
                expressions[ind] = expression[:-6] +"%d) " % level 
        # all_programs[program_key] = expressions
        all_exprs.extend(expressions)
        all_exprs.append("$$")
    return all_exprs
    