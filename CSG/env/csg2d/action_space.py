from os import stat
import torch as th
import torch
import numpy as np

from gym.spaces.discrete import Discrete

from .parser import draw_commands, boolean_commands, transform_commands, macro_commands, fixed_macro_commands
from CSG.env.csg3d.constants import (ROTATE_MULTIPLIER, SCALE_ADDITION, TRANSLATE_MIN, TRANSLATE_MAX, 
                        SCALE_MIN, SCALE_MAX, ROTATE_MIN, ROTATE_MAX, DRAW_MIN, DRAW_MAX, CONVERSION_DELTA)
from CSG.env.csg3d.action_space import MCSGAction3D

class MCSGAction2D(MCSGAction3D):

    def __init__(self, resolution=64):

        ## N = res + here stands for
        # 1) Numbers = Resolution = RES
        # 4) Primitive Types = 3
        # 2) Transforms = 3
        # 3) Booleans = 3
        # 4) Macro(MIRROR) = 1
        # 5) Fixed Macro(macro) = 3
        # 5) Stop Symbol
        self.resolution = resolution

        self.action_types = {
            "NUMERIC": resolution,
            "DRAW": 2,
            "TRANSFORM": 3,
            "BOOL": 3,
            "MACRO": 1,
            "FIXED_MACRO": 2,
            "STOP": 1
        }
        self.command_index = {
            'sphere': self.resolution, 
            'cuboid': self.resolution + 1, 
            'translate': self.resolution + 2,
            'rotate': self.resolution + 3, 
            'scale': self.resolution + 4,
            'union': self.resolution + 5, 
            'intersection': self.resolution + 6, 
            'difference': self.resolution + 7,
            'mirror': self.resolution + 8,
            'macro(MIRROR_X)': self.resolution + 9,
            'macro(MIRROR_Y)': self.resolution + 10,
            '$': self.resolution + 11,
            ## ADD MACROS HERE
        } 
        #Tuple for (Type of command, number of continuous parameters)
        self.index_to_command = {x:(0, 0) for x in range(self.resolution)} 
        self.index_to_command.update(
            {
                self.resolution: (1, 0),
                self.resolution + 1: (1, 0),
                # self.resolution  3: (1, 3),
                self.resolution + 2: (2, 2),
                self.resolution + 3: (2, 1),
                self.resolution + 4: (2, 2),
                self.resolution + 5: (3, 0),
                self.resolution + 6: (3, 0),
                self.resolution + 7: (3, 0),
                ## FUNCTIONALLY EQUIVALENT TO "TRANFORMS"
                self.resolution + 8: (2, 2),
                self.resolution + 9: (2, 0),
                self.resolution + 10: (2, 0),
                self.resolution + 11: (4, 0)
            }
        )
        self.stop_action = self.resolution + 11

        self.init_state()
        self.init_action_limits()
        
class HCSGAction2D(MCSGAction2D):

    def __init__(self, resolution=64):

        self.resolution = resolution

        self.action_types = {
            "NUMERIC": resolution,
            "DRAW": 2,
            "TRANSFORM": 3,
            "BOOL": 3,
            "MACRO": 0,
            "FIXED_MACRO": 0,
            "STOP": 1 
        }
        self.command_index = {
            'sphere': self.resolution, 
            'cuboid': self.resolution + 1, 
            'translate': self.resolution + 2,
            'rotate': self.resolution + 3, 
            'scale': self.resolution + 4,
            'union': self.resolution + 5, 
            'intersection': self.resolution + 6, 
            'difference': self.resolution + 7,
            '$': self.resolution + 8,
            ## ADD MACROS HERE
        } 
        #Tuple for (Type of command, number of continuous parameters)
        self.index_to_command = {x:(0, 0) for x in range(self.resolution)} 
        self.index_to_command.update(
            {
                self.resolution: (1, 0),
                self.resolution + 1: (1, 0),
                # self.resolution + 3: (1, 3),
                self.resolution + 2: (2, 2),
                self.resolution + 3: (2, 1),
                self.resolution + 4: (2, 2),
                self.resolution + 5: (3, 0),
                self.resolution + 6: (3, 0),
                self.resolution + 7: (3, 0),
                self.resolution + 8: (4, 0)
            }
        )
        self.stop_action = self.resolution + 8

        self.init_state()
        self.init_action_limits()


class FCSGAction2D(MCSGAction3D):

    def __init__(self, resolution=64):

        self.resolution = resolution

        self.action_types = {
            "NUMERIC": resolution,
            "DRAW": 2,
            "TRANSFORM": 0,
            "BOOL": 3,
            "MACRO": 0,
            "FIXED_MACRO": 0,
            "STOP": 1 
        }
        self.command_index = {
            'sphere': self.resolution, 
            'cuboid': self.resolution + 1, 
            'union': self.resolution + 2, 
            'intersection': self.resolution + 3, 
            'difference': self.resolution + 4,
            '$': self.resolution + 5,
            ## ADD MACROS HERE
        } 
        #Tuple for (Type of command, number of continuous parameters)
        self.index_to_command = {x:(0, 0) for x in range(self.resolution)} 
        self.index_to_command.update(
            {
                self.resolution: (1, 5),
                self.resolution + 1: (1, 5),
                self.resolution + 2: (3, 0),
                self.resolution + 3: (3, 0),
                self.resolution + 4: (3, 0),
                self.resolution + 5: (4, 0)
            }
        )
        self.stop_action = self.resolution + 5

        self.init_state()
        self.init_action_limits()

    def single_expression_to_action(self, expr):
        command_symbol = expr.split("(")[0]
        if command_symbol in boolean_commands:
            action_list = [self.command_index[command_symbol]]
        elif command_symbol in draw_commands:
            param_str = expr.split("(")[1][:-1]
            param = np.array([float(x.strip()) for x in param_str.split(",")])
            translate_param = np.clip(param[:2], TRANSLATE_MIN + self.conversion_delta, TRANSLATE_MAX - self.conversion_delta)
            scale_param = np.clip(param[2:4], SCALE_MIN + self.conversion_delta, SCALE_MAX - self.conversion_delta)
            scale_param -= SCALE_ADDITION
            rotate_param = np.clip(param[4:5], ROTATE_MIN + self.conversion_delta, ROTATE_MAX - self.conversion_delta)
            rotate_param = rotate_param / ROTATE_MULTIPLIER
                # param = [self.mid_point + np.round( (x-1.25)/self.two_scale_delta) for x in param]
            param = np.concatenate([translate_param, scale_param, rotate_param], 0)
            param = (param - (-1 + self.conversion_delta)) / self.two_scale_delta
            param = np.round(param).astype(np.uint32)
            action_list = [self.command_index[command_symbol], ] + list(param)
        elif command_symbol == "$":
            action_list = [self.command_index[command_symbol]]
        return action_list

    def action_to_expression(self, actions):
        size_ = actions.shape[0]
        pointer = 0
        expression_list = []
        while(pointer < size_):
            cur_command = actions[pointer]
            cur_expr = self.index_to_expr[cur_command]
            n_param = self.index_to_command[cur_command][1]
            if n_param > 0:
                # Has to be for transform or mirror
                param = np.array(actions[pointer+1: pointer + 1 + n_param])
                translate_param = -1 + self.conversion_delta + param[:2] * self.two_scale_delta
                scale_param = -1 + self.conversion_delta + param[2:4] * self.two_scale_delta + SCALE_ADDITION
                rotate_param = (-1 + self.conversion_delta + param[4:5] * self.two_scale_delta) * ROTATE_MULTIPLIER
                
                param = np.concatenate([translate_param, scale_param, rotate_param], 0)
                param_str = ", ".join(["%f" % x for x in param])
                cur_expr = "%s(%s)" %(cur_expr, param_str)
                pointer += n_param
            expression_list.append(cur_expr)
            pointer += 1
                
        return expression_list


class PCSGAction2D(FCSGAction2D):

    def __init__(self, resolution=64):

        self.resolution = resolution

        self.action_types = {
            "NUMERIC": resolution,
            "DRAW": 2,
            "TRANSFORM": 0,
            "BOOL": 3,
            "MACRO": 0,
            "FIXED_MACRO": 0,
            "STOP": 1 
        }
        self.command_index = {
            'sphere': self.resolution, 
            'cuboid': self.resolution + 1, 
            'union': self.resolution + 2, 
            'intersection': self.resolution + 3, 
            'difference': self.resolution + 4,
            '$': self.resolution + 5,
            ## ADD MACROS HERE
        } 
        #Tuple for (Type of command, number of continuous parameters)
        self.index_to_command = {x:(0, 0) for x in range(self.resolution)} 
        self.index_to_command.update(
            {
                self.resolution: (1, 4),
                self.resolution + 1: (1, 4),
                self.resolution + 2: (3, 0),
                self.resolution + 3: (3, 0),
                self.resolution + 4: (3, 0),
                self.resolution + 5: (4, 0)
            }
        )
        self.stop_action = self.resolution + 5

        self.init_state()
        self.init_action_limits()


    def single_expression_to_action(self, expr):
        command_symbol = expr.split("(")[0]
        if command_symbol in boolean_commands:
            action_list = [self.command_index[command_symbol]]
        elif command_symbol in draw_commands:
            param_str = expr.split("(")[1][:-1]
            param = np.array([float(x.strip()) for x in param_str.split(",")])
            translate_param = np.clip(param[:2], TRANSLATE_MIN + self.conversion_delta, TRANSLATE_MAX - self.conversion_delta)
            scale_param = np.clip(param[2:4], SCALE_MIN + self.conversion_delta, SCALE_MAX - self.conversion_delta)
            scale_param -= SCALE_ADDITION
                # param = [self.mid_point + np.round( (x-1.25)/self.two_scale_delta) for x in param]
            param = np.concatenate([translate_param, scale_param], 0)
            param = (param - (-1 + self.conversion_delta)) / self.two_scale_delta
            param = np.round(param).astype(np.uint32)
            action_list = [self.command_index[command_symbol], ] + list(param)
        elif command_symbol == "$":
            action_list = [self.command_index[command_symbol]]
        return action_list

    def action_to_expression(self, actions):
        size_ = actions.shape[0]
        pointer = 0
        expression_list = []
        while(pointer < size_):
            cur_command = actions[pointer]
            cur_expr = self.index_to_expr[cur_command]
            n_param = self.index_to_command[cur_command][1]
            if n_param > 0:
                # Has to be for transform or mirror
                param = np.array(actions[pointer+1: pointer + 1 + n_param])
                translate_param = -1 + self.conversion_delta + param[:2] * self.two_scale_delta
                scale_param = -1 + self.conversion_delta + param[2:4] * self.two_scale_delta + SCALE_ADDITION
                param = np.concatenate([translate_param, scale_param], 0)
                param_str = ", ".join(["%f" % x for x in param])
                cur_expr = "%s(%s)" %(cur_expr, param_str)
                pointer += n_param
            expression_list.append(cur_expr)
            pointer += 1
                
        return expression_list