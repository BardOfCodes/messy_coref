from multiprocessing.sharedctypes import Value
import time
from .draw import DiffDraw2D
# from .compiler_utils import SACuboid, SAAttach, BooleanHolder
from CSG.env.csg3d.compiler import MCSG3DCompiler
import torch as th
import numpy as np
from collections import defaultdict


class MCSG2DCompiler(MCSG3DCompiler):

    def __init__(self, resolution=64, scale=64, scad_resolution=30, device="cuda", draw_mode="direct", *args, **kwargs):

        self.draw = DiffDraw2D(resolution=resolution, device=device, mode=draw_mode)

        self.scad_resolution = scad_resolution
        self.space_scale = scale
        self.resolution = resolution
        self.scale = scale

        self.mirror_start_boolean_stack = []
        self.mirror_start_canvas_stack = []
        self.mirror_init_size = []

        self.device = device
        self.set_to_full()

        self.reset()
        self.set_init_mapping()

    ### Conversion Functions
    def _compile_to_scad(self, command_list):
        raise ValueError("2D Open SCAD?")

    def draw_command_to_scad(self, command, stack_state):
        raise ValueError("2D Open SCAD?")

    def write_to_scad(self, command_list, file_name):
        raise ValueError("2D Open SCAD?")
    
    def write_to_stl(self, command_list, file_name):
        raise ValueError("2D STL?")
    
    def march_to_ply(self, command_list, file_name):
        raise ValueError("2D STL?")
    
    def write_to_gltf(self, command_list, file_name):
        raise ValueError("2D STL?")
        
    
