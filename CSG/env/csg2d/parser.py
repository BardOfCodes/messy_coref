import numpy as np
import os
import torch as th
from CSG.env.csg3d.parser_utils import boolean_commands, transform_commands, draw_commands, macro_commands, fixed_macro_commands, mcsg_get_expression
from CSG.env.csg3d.constants import (TRANSLATE_RANGE_MULT, SCALE_RANGE_MULT, ROTATE_RANGE_MULT, ROTATE_MULTIPLIER, SCALE_ADDITION, SCALE_MULTIPLIER,
                        TRANSLATE_MIN, TRANSLATE_MAX, ROTATE_MIN, ROTATE_MAX, SCALE_MIN, SCALE_MAX, CONVERSION_DELTA, DRAW_MIN, DRAW_MAX)
from CSG.env.csg3d.parser import MCSG3DParser

class MCSG2DParser(MCSG3DParser):
    """ Expression to Command List.
    Eventually, a single macro might lead to multiple commands here. 
    """
    def load_language_specific_details(self):
        self.command_n_param = {
            "sphere": 0,
            "cuboid": 0,
            "translate": 2,
            "rotate": 1,
            "scale": 2,
            "union": 0,
            "intersection": 0,
            "difference": 0,
            "mirror": 2,
            # For FCSG
            "rotate_sphere": 2,
            "rotate_cylinder": 2,
            "rotate_cuboid": 2,
        }
        self.command_symbol_to_type = {
            "sphere": "D",
            "cuboid": "D",
            "translate": "T",
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
            "MIRROR_X": [1., 0.,],
            "MIRROR_Y": [0., 1.,],
        }
        self.invalid_commands = []

        self.trivial_expression = ["sphere", "$"]
        self.has_transform_commands = True
        self.load_fixed_macros("box_edges.mcsg2d")

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
        
    
    def get_transform(self, transform, bbox):
        rand_var = np.random.beta(2, 3, size=2)
        if transform == "translate":
            sign = np.random.choice([-1, 1], size=2)
            rand_var = rand_var * sign
            parameters =  rand_var * TRANSLATE_RANGE_MULT
            if not bbox is None:
                max_translate = 1 - bbox[1]
                min_translate = -1 - bbox[0]
                parameters = min_translate + parameters * (max_translate - min_translate)
                bbox += parameters
            expr = transform + "(%f, %f)" % tuple(parameters)
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
            expr = transform + "(%f, %f)" % tuple(parameters)
        elif transform == "rotate":
            # SKIP BBOX adjustment
            sign = np.random.choice([-1, 1], size=1)
            rand_var = rand_var * sign
            rand_var = rand_var[:1]
            parameters = rand_var * ROTATE_RANGE_MULT
            expr = transform + "(%f)" % tuple(parameters)
        
        expr = [expr]
        
        return expr, bbox

    
    def get_mirror_transform(self, bbox):
        if not bbox is None:
            bbox_center = (bbox[1] - bbox[0]) + 1e-9
            parameters = list(bbox_center)
        else:
            rand_var = np.random.beta(2, 3, size=2)
            sign = np.random.choice([-1, 1], size=2)
            parameters = rand_var * sign

        expr = "mirror" + "(%f, %f)" % tuple(parameters)
        expr = [expr]
        
        return expr, None
    

    def get_macro_mirror(self, bbox=None):
        mirrors = ["MIRROR_X", "MIRROR_Y"]
        if not bbox is None:
            valid = [(bbox[0, i] > -0.1) or (bbox[1, i] < 0.1) for i in range(2)]
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
    