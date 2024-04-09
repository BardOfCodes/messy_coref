""" 
Draw cuboids and multiple unions.
"""

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from CSG.env.csg3d.draw_utils import (direct_sub_rbb, direct_shape_transform, direct_sub_gbc,
                                      direct_grid_operation, direct_rot_operation, get_rotation_matrix)
from CSG.env.csg3d.constants import PRIMITIVE_OCCUPANCY_MAX_THRES, PRIMITIVE_OCCUPANCY_MIN_THRES, BBOX_LIMIT, CONVERSION_DELTA
from CSG.env.csg3d.draw import DiffDraw3D

class SADiffDraw3D(DiffDraw3D):
    
    def __init__(self, resolution=64, device="cpu"):
        """
        Helper function for drawing the canvases.
        DRAW is always in a Unit Canvas (BBox -1 to 1).
        :param canvas_shape: shape of the canvas on which to draw objects
        """
        super(SADiffDraw3D, self).__init__(resolution=resolution, device=device, mode="direct")
        
    def draw_cuboid(self, param=[1.0, 1.0, 1.0], coords=None):
        # Just to change the default parameters.
        base_sdf = super(SADiffDraw3D, self).draw_cuboid(param, coords)
        return base_sdf
    
    def draw_sphere(self, param=[0.5], coords=None):
        ''' Draw a cricle centered at 0
        '''
        raise ValueError("Not Allowed in ShapeAssembly")
    
    def draw_cylinder(self, param=[0.5, 0.5], coords=None):
        """ Default cylinder is z-axis aligned.
        """
        raise ValueError("Not Allowed in ShapeAssembly")
    
    def draw_ellipsoid(self, param=[0.5, 0.5, 0.5], coords=None):
        """ Since it involves scaling its not a proper sdf!
        """
        raise ValueError("Not Allowed in ShapeAssembly")

    def rotate_with_matrix(self, param=[1, 0, 0, 0, 1, 0, 0, 0, 1], coords=None):

        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, device=self.device, dtype=self.tensor_type)
        # param = -param


        row_1 = th.stack([param[0], param[1], param[2]])
        row_2 = th.stack([param[3], param[4], param[5]])
        row_3 = th.stack([param[6], param[7], param[8]])
        # row_1 = th.stack([param[0], param[3], param[6]])
        # row_2 = th.stack([param[1], param[4], param[7]])
        # row_3 = th.stack([param[2], param[5], param[8]])
        rotation_matrix = th.stack([row_1, row_2, row_3], 0)
        # rotation_matrix = th.stack([row_3, row_2, row_1], 0)
        if coords is None:
            coords = self.base_coords.clone()
        
        coords = th.reshape(coords, (-1, 3, 1))
        
        rotation_matrix = th.unsqueeze(rotation_matrix, 0)
        rotation_matrix = rotation_matrix.expand(coords.shape[0], -1, -1)
        coords = th.bmm(rotation_matrix, coords).squeeze()
        res = self.resolution
        coords = th.reshape(coords, (res, res, res, 3))
        
        return coords

    def shape_rotate_with_matrix(self, param, input_shape, units='degree', inverted=True):
        # Assume Shape to be a RES, RES, RES, 2 object


        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, device=self.device, dtype=self.tensor_type)
        # self.selected_shape_transform(param)
        # Performing post multiplication here.
        param = param
        # R = R^T?
        row_1 = th.stack([param[0], param[1], param[2]])
        row_2 = th.stack([param[3], param[4], param[5]])
        row_3 = th.stack([param[6], param[7], param[8]])
        # row_1 = th.stack([param[0], param[3], param[6]])
        # row_2 = th.stack([param[1], param[4], param[7]])
        # row_3 = th.stack([param[2], param[5], param[8]])
        
        rotation_matrix = th.stack([row_1, row_2, row_3], 0)
        coords = self.base_coords.clone()
        coords = th.reshape(coords, (-1, 1, 3))
        rotation_matrix = th.unsqueeze(rotation_matrix, 0)
        rotation_matrix = rotation_matrix.expand(coords.shape[0], -1, -1)
        rotated_3d_positions = th.bmm(coords, rotation_matrix)# .squeeze()
        # print(rotated_3d_positions.shape)
        rot_locs = th.split(rotated_3d_positions, split_size_or_sections=1, dim=2)
        
        res = self.resolution
        grid = self.selected_rot_operation(rot_locs, res)
        
        pred_shape = self._shape_sample(grid, input_shape)

        return pred_shape

    def union(self, *args):
        output = th.min(th.stack(args, 0), 0)[0]
        return output
    
    def intersection(self, *args):
        raise ValueError("Not Allowed in ShapeAssembly")
    
    def difference(self, sdf_1, sdf_2):
        raise ValueError("Not Allowed in ShapeAssembly")

    def shape_mirror(self, param, input_shape):

        coords = self.base_coords.clone()
        mirrored_coords = coords.clone()

        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, device=self.device, dtype=self.tensor_type)
        
        param = th.reshape(param, (1, 1, 1, 3))
        param = param/(th.norm(param) + CONVERSION_DELTA)
        mirrored_coords = mirrored_coords - 2 * th.sum(mirrored_coords * param, -1, keepdim=True) * param

        coords = th.reshape(coords, (-1, 1, 3))
        mirrored_coords = th.reshape(mirrored_coords, (-1, 1, 3))

        res = self.resolution
        mirrored_grid = self.selected_grid_operation(mirrored_coords, res)
        # print("reverse reflect")
        # print(input_shape.shape)
        mirrored_shape = self._shape_sample(mirrored_grid, input_shape)
        # print(mirrored_shape.shape)

        # now do a union of the shapes, and a intersection of the masks.
        
        # also mask out the other part. 
        # mirrored_coords = coords.clone()
        # mirror_mask = th.sum(mirrored_coords * param, -1, keepdim=True)>0
        # mask = th.logical_or(mask, mirror_mask)
        # pred_shape = th.stack([pred_shape, mask], 0)

        return mirrored_shape
    
        
    def intersection_invert(self, target, other_child, index=0):
        raise ValueError("Not allowed with ShapeAssembly")
    
    def difference_invert(self, target, other_child, index=0):
        raise ValueError("Not allowed with ShapeAssembly")

    def shape_difference(self, obj1, obj2):
        raise ValueError("Not allowed with ShapeAssembly")