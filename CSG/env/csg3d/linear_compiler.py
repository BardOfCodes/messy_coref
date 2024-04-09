import networkx as nx
import torch as th
import numpy as np
from .compiler import MCSG3DCompiler
from .compiler_utils import get_reward
from .linear_draw import LinearDraw3D

class LinearMCSG3DCompiler(MCSG3DCompiler):

    def __init__(self, resolution=64, scale=64, scad_resolution=30, device="cpu", draw_mode="inverted", *args, **kwargs):

        self.draw = LinearDraw3D(resolution=resolution, device=device, mode=draw_mode)

        self.scad_resolution = scad_resolution
        self.space_scale = scale
        self.resolution = resolution
        self.scale = scale
        self.draw_mode = draw_mode

        self.mirror_start_boolean_stack = []
        self.mirror_start_canvas_stack = []
        self.mirror_init_size = []

        self.device = device
        self.set_to_full()

        self.reset()
        self.set_init_mapping()
    
    def _compile(self, command_list, coord_array=None, reset=True):
        
        if reset:
            self.reset()
        if coord_array is not None:
            self.draw.base_coords = coord_array
        self.transformed_coords = [[self.draw.base_coords.clone()]]
        return super()._compile(command_list, reset = False)
    
# Avoid the graph part
