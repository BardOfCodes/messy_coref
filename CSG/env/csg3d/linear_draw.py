import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .draw_utils import (inverted_sub_rbb, direct_sub_rbb, inverted_shape_transform, direct_shape_transform,
                         direct_sub_gbc, inverted_sub_gbc, inverted_grid_operation, direct_grid_operation,
                         inverted_rot_operation, direct_rot_operation, get_rotation_matrix)
from .constants import PRIMITIVE_OCCUPANCY_MAX_THRES, PRIMITIVE_OCCUPANCY_MIN_THRES, BBOX_LIMIT, CONVERSION_DELTA
from .draw import DiffDraw3D
class LinearDraw3D(DiffDraw3D):
    
    def get_base_coords(self):
        res = self.resolution
        coords = self.selected_sub_gbc(res)
        coords = coords.astype(np.float32)
        coords = ((coords + 0.5) / res - 0.5) * 2
        coords = th.from_numpy(coords).float().to(self.device)
        coords = th.reshape(coords, (-1, 3))
        return coords
    def reset(self):
        self.base_coords = self.get_base_coords()
        
    def translate(self, param=[0,0,0], coords=None):
        # param = self.adjust_scale(param)
        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)
        param = th.reshape(param, (1, 3))
        
        if coords is None:
            coords = self.base_coords.clone()
        coords -= param
        return coords

    def scale(self, param=[1, 1, 1], coords=None):
        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)
        param = th.reshape(param, (1, 3))
        
        if coords is None:
            coords = self.base_coords.clone()
        coords = coords / param
        return coords
    
    def rotate(self, param=[0,0,0], coords=None, units="degree", inverted=False):
        
        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)
        if units == "degree":
            param = (param * np.pi) / 180.0

        # since we want inverted: 
        param = - param
        rotation_matrix = get_rotation_matrix(param, inverted)
        if coords is None:
            coords = self.base_coords.clone()
        
        # coords = th.reshape(coords, (-1, 1, 3))
        coords = coords.unsqueeze(1)
        
        rotation_matrix = th.unsqueeze(rotation_matrix, 0)
        rotation_matrix = rotation_matrix.expand(coords.shape[0], -1, -1)
        coords = th.bmm(coords, rotation_matrix)
        res = self.resolution
        # coords = th.reshape(coords, (res, res, res, 3))
        coords = coords.squeeze(1)
        return coords
    
        
    def draw_sphere(self, param=[0.5], coords=None):
        ''' Draw a cricle centered at 0
        '''
        # param = self.adjust_scale(param)
        if coords is None:
            coords = self.base_coords.clone()
            
        r = param[0]
        base_sdf = coords.norm(dim=-1)
        base_sdf = base_sdf - r
        ## to zero or not: 
        return base_sdf
    
    def draw_cuboid(self, param=[0.5, 0.5, 0.5], coords=None):
        
        # param = self.adjust_scale(param)
        if coords is None:
            coords = self.base_coords.clone()
            
        base_sdf = th.abs(coords)
        base_sdf[:, 0] -= param[0]
        base_sdf[:, 1] -= param[1]
        base_sdf[:, 2] -= param[2]
        base_sdf = th.norm(th.clip(base_sdf, min=0), dim=-1)  + th.clip(th.amax(base_sdf, -1), max=0)
        return base_sdf
    
    def draw_cylinder(self, param=[0.5, 0.5], coords=None):
        """ Default cylinder is z-axis aligned.
        """
        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)
        
        # param = self.adjust_scale(param)
        if coords is None:
            coords = self.base_coords.clone()
        r = param[0]
        h = param[1] ## Treat everything as a radial measure
        xy_vec = th.norm(coords[:, :2], dim=-1) - r
        height = th.abs(coords[:, 2]) - h
        vec2 = th.stack([xy_vec, height], -1)
#         vec2[:,:,:,0] -= r
#         vec2[:,:,:,1] -= h
        sdf = th.amax(vec2, -1) + th.norm(th.clip(vec2, min=0.0) + CONVERSION_DELTA, -1)
        return sdf
    
            
    
    def draw_ellipsoid(self, param=[0.5, 0.5, 0.5], coords=None):
        """ Since it involves scaling its not a proper sdf!
        """
        # param = self.adjust_scale(param)
        if coords is None:
            coords = self.base_coords.clone()
        # param = [np.sqrt(x) for x in param]
        coords = self.scale(param=param, coords=coords)
        base_sdf = coords.norm(dim=-1)
        base_sdf = base_sdf - 1
        ## to zero or not: 
        return base_sdf
        
    def mirror_coords(self, param=[1, 1, 1], coords=None, mode="euclidian"):
        # Invert the coords along the axis?
        if coords is None:
            coords = self.base_coords.clone()
        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)
        if mode == "polar":
            phi = param[0]
            theta = param[1]
            param = th.stack([th.cos(theta) * th.cos(phi), th.cos(theta) * th.sin(phi), th.sin(theta)], 0)
        mirrored_coords = coords.clone()
        # refect about the origin
        # r = d - 2 (d.n)n
        param = th.reshape(param, (1, 3))
        param = param/(th.norm(param) + CONVERSION_DELTA)
        mirrored_coords = mirrored_coords - 2 * th.sum(mirrored_coords * param, -1, keepdim=True) * param
        return mirrored_coords

    
    # def return_inside_coords(self, sdf, normalized=True):
    #     sdf_np = sdf.cpu().numpy()
    #     sdf_coords = np.stack(np.where(sdf_np <=0), -1)
    #     if normalized:
    #         sdf_coords = -1 + (sdf_coords + 0.5)/self.grid_divider
    #     return sdf_coords
    
    # def return_bounding_box(self, sdf, normalized=True):
        
    #     sdf_coords = self.return_inside_coords(sdf, normalized=False)
    #     # print("in bbox, shape of sdf_coords", sdf_coords.shape)
    #     if sdf_coords.shape[0] == 0:
    #         if normalized:
    #             bbox = np.zeros((2, 3), dtype=np.float32)
    #         else:
    #             bbox = np.zeros((2, 3), dtype=np.int32)
    #     else:
    #         bbox = self.selected_sub_rbb(sdf_coords)
    #         if normalized:
    #             bbox = -1 + (bbox + 0.5)/self.grid_divider
    #     return bbox
    
    # def union(self, *args):
    #     output = th.minimum(*args)
    #     return output
    
    # def intersection(self, *args):
    #     output = th.maximum(*args)
    #     return output
    
    # def difference(self, sdf_1, sdf_2):
        
    #     output = th.maximum(sdf_1, -sdf_2)
    #     return output
    
    # def is_valid_primitive(self, sdf):
    #     sdf_shape = (sdf<=0)
    #     occupancy_rate = th.sum(sdf_shape)/float(sdf.nelement())
    #     occupancy_check = False
    #     bbox_check = False
    #     if self.primitive_occupancy_min_thres < occupancy_rate < self.primitive_occupancy_max_thres:
    #         occupancy_check = True
    #         bbox = self.return_bounding_box(sdf)
    #         if self.valid_bbox(bbox):
    #             bbox_check = True
    #     return bbox_check and occupancy_check

    # def is_valid_sdf(self, sdf):
    #     sdf_shape = (sdf<=0)
    #     occupancy_rate = th.sum(sdf_shape)/float(sdf.nelement())
    #     occupancy_check = False
    #     bbox_check = False
    #     if self.primitive_occupancy_min_thres < occupancy_rate:
    #         occupancy_check = True
    #         bbox = self.return_bounding_box(sdf)
    #         if self.valid_bbox(bbox):
    #             bbox_check = True
    #     return bbox_check and occupancy_check
    
    # def valid_bbox(self, bbox):
    #     bbox_check = np.abs(bbox).max() < self.bbox_limit
    #     return bbox_check
        
    def shape_translate(self, param, input_shape):
        # Assume Shape to be a RES, RES, RES, 2 object
        # The second should be updated with 0 wherever required. 
        # use input shape as boolean.
        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)
        self.selected_shape_transform(param)    

        coords = self.base_coords.clone()
        # coords = th.reshape(coords, (-1, 1, 3))
        scaled_coords = coords - param
        
        res = self.resolution
        grid = self.selected_grid_operation(scaled_coords, res)
        
        pred_shape = self._shape_sample(grid, input_shape)
        return pred_shape
        

    def shape_rotate(self, param, input_shape, units='degree', inverted=True):
        # Assume Shape to be a RES, RES, RES, 2 object


        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)
        # self.selected_shape_transform(param)    
        if units == "degree":
            param = (param * np.pi) / 180.0
        # This is also done in normal rotate.
        param = - param

        rotation_matrix = get_rotation_matrix(param, inverted=inverted)

        coords = self.base_coords.clone()
        coords = th.reshape(coords, (-1, 1, 3))
        rotation_matrix = th.unsqueeze(rotation_matrix, 0)
        rotation_matrix = rotation_matrix.expand(coords.shape[0], -1, -1)
        rotated_3d_positions = th.bmm(coords, rotation_matrix)
        # print(rotated_3d_positions.shape)
        rot_locs = th.split(rotated_3d_positions, split_size_or_sections=1, dim=2)
        
        res = self.resolution
        grid = self.selected_rot_operation(rot_locs, res)
        
        pred_shape = self._shape_sample(grid, input_shape)

        return pred_shape
        
    def shape_scale(self, param, input_shape):

        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)
        # self.selected_shape_transform(param) 


        coords = self.base_coords.clone()
        coords = th.reshape(coords, (-1, 1, 3))
        scaled_coords = coords / param
        
        res = self.resolution
        grid = self.selected_grid_operation(scaled_coords, res)
        
        pred_shape = self._shape_sample(grid, input_shape)
        return pred_shape

    def _shape_sample(self, grid, input_shape):

        input_shape_sampling = input_shape.permute(3, 0, 1, 2)
        input_shape_sampling = input_shape_sampling.to(self.tensor_type)
        input_shape_sampling = input_shape_sampling.unsqueeze(0)
        pred_shape = F.grid_sample(input=input_shape_sampling, grid=grid, mode='nearest',  align_corners=False)
        # pred_shape = pred_shape > 0.0
        pred_shape = pred_shape.bool()
        pred_shape = pred_shape.squeeze(0)
        grid_mask = (th.max(th.abs(grid[0]), 3)[0] <= 1)
        pred_shape[1] = th.logical_and(pred_shape[1], grid_mask)

        pred_shape = pred_shape.permute(1, 2, 3, 0)
        return pred_shape

    def shape_mirror(self, param, input_shape):

        coords = self.base_coords.clone()
        mirrored_coords = coords.clone()
        
        param = th.reshape(param, (1, 1, 1, 3))
        param = param/(th.norm(param) + CONVERSION_DELTA)
        mirrored_coords = mirrored_coords - 2 * th.sum(mirrored_coords * param, -1, keepdim=True) * param

        coords = th.reshape(coords, (-1, 1, 3))
        mirrored_coords = th.reshape(mirrored_coords, (-1, 1, 3))

        res = self.resolution
        grid = self.selected_grid_operation(coords, res)
        normal_shape = self._shape_sample(grid, input_shape)
        mirrored_grid = self.selected_grid_operation(mirrored_coords, res)
        mirrored_shape = self._shape_sample(mirrored_grid, input_shape)

        # now do a union of the shapes, and a intersection of the masks.
        
        # also mask out the other part. 
        mirrored_coords = coords.clone()
        mirror_mask = th.sum(mirrored_coords * param, -1, keepdim=True)>0
        mask = th.logical_or(mask, mirror_mask)
        pred_shape = th.stack([pred_shape, mask], 0)

        return pred_shape
    
        
    def union_invert(self, target, other_child, index=0):
        # Doesnt matter which index is used:
        target_shape = target[:,:,:,0]
        prior_mask = target[:,:,:,1]
        masked_region = self.shape_intersection(target_shape, other_child)
        output_mask = th.logical_and(prior_mask,  ~masked_region)
        output_shape = th.stack([target_shape, output_mask], 3)
        return output_shape

        
    def difference_invert(self, target, other_child, index=0):
        target_shape = target[:,:,:,0]
        prior_mask = target[:,:,:,1]

        if index == 0:
            output_target = target_shape
            output_mask = th.logical_and(prior_mask, ~other_child)
        else:
            output_target = ~target_shape
            masked_region = self.shape_union(target_shape, other_child)
            output_mask = th.logical_and(prior_mask, masked_region)
        output_shape = th.stack([output_target, output_mask], 3)
        return output_shape

    def intersection_invert(self, target, other_child, index=0):
        target_shape = target[:,:,:,0]
        prior_mask = target[:,:,:,1]
        masked_region = self.shape_union(target_shape, other_child)
        output_mask = th.logical_and(prior_mask, masked_region)
         # (1 - other_child)
        # output_target = self.shape_intersection(target_shape, other_child)
        output_shape = th.stack([target_shape, output_mask], 3)
        return output_shape

    def shape_union(self, obj1, obj2, *args):
        output = th.logical_or(obj1, obj2)
        return output
    
    def shape_intersection(self, obj1, obj2, *args):
        output = th.logical_and(obj1, obj2)
        return output
    
    def shape_difference(self, obj1, obj2):
        output = th.logical_and(obj1, ~obj2)
        return output
