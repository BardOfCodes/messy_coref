from CSG.env.csg3d.graph_compiler import GraphicalMCSG3DCompiler
from CSG.env.reward_function import chamfer
from .draw import DiffDraw2D
import torch as th
import numpy as np
class GraphicalMCSG2DCompiler(GraphicalMCSG3DCompiler):
    
    def __init__(self, resolution=64, scale=64, scad_resolution=30, device="cuda", draw_mode="direct", *args, **kwargs):

        self.draw = DiffDraw2D(resolution=resolution, device=device, mode=draw_mode)

        self.scad_resolution = scad_resolution
        self.space_scale = scale
        self.resolution = resolution
        self.scale = scale

        self.mirror_start_boolean_stack = []
        self.mirror_start_canvas_stack = []
        self.mirror_init_size = []

        self.denoise = True
        filter_size = 5
        self.denoisizing_threshold = 8
        self.denoising_filter = th.ones([1, 1, filter_size, filter_size],
                                        device=self.draw.device, dtype=th.float32)
        self.device = device
        self.set_to_full()

        self.reset()
        self.set_init_mapping()

        self.threshold = 0.025
        self.threshold_diff = 0.05
        self.mode = "2D"

    def get_reward(self, new_canvas_set, target):
        # Cheat and add it to only one:
        selected_shape = new_canvas_set[0]
        selected_shape = selected_shape.cpu().numpy()
        target_np = target.cpu().numpy()

        R = 100 - chamfer(target_np[None, :, :], selected_shape[None, :, :])[0]
        return R

    def calculate_subexpr_stats(self, cur_node):
        
        pred_shape = cur_node['subexpr_info']['expr_shape']
        target = cur_node['subexpr_info']['expr_target']
        target_mask = target[:,:,1]
        target_shape = target[:,:,0]

        # use Bounding box
        bbox = cur_node['subexpr_info']['bbox']
        min_x, min_y = bbox[0]
        max_x, max_y = bbox[1] + 1

        pred_shape = pred_shape[min_x:max_x, min_y:max_y]
        target_shape = target_shape[min_x:max_x, min_y:max_y]
        target_mask = target_mask[min_x:max_x, min_y:max_y]
        
        ## Increase the mask, and also calculate the score in canonical form.

        R = th.sum(th.logical_and(th.logical_and(pred_shape, target_shape), target_mask)) / \
                (th.sum(th.logical_and(th.logical_or(pred_shape, target_shape), target_mask)) + 1e-6)
        cur_node['subexpr_info']['masked_iou'] = R
        cur_node['subexpr_info']['masking_rate'] = 1 - target_mask.float().mean()
        cur_node['subexpr_info']['masked_matching'] = th.logical_and((pred_shape == target_shape), target_mask).sum() / target_mask.sum()
        ### TODO: Add Hierarchy Volume and Expression length
        cur_node['subexpr_info']['sa_sweep_metric'] = th.logical_and(th.logical_and(pred_shape, target_shape), target_mask).sum()
