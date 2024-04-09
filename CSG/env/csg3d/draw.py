import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .draw_utils import (inverted_sub_rbb, direct_sub_rbb, inverted_shape_transform, direct_shape_transform,
                         direct_sub_gbc, inverted_sub_gbc, inverted_grid_operation, direct_grid_operation,
                         inverted_rot_operation, direct_rot_operation, get_rotation_matrix)
from .constants import PRIMITIVE_OCCUPANCY_MAX_THRES, PRIMITIVE_OCCUPANCY_MIN_THRES, BBOX_LIMIT, CONVERSION_DELTA
class DiffDraw3D:
    
    def __init__(self, resolution=64, device="cpu", mode="inverted"):
        """
        Helper function for drawing the canvases.
        DRAW is always in a Unit Canvas (BBox -1 to 1).
        :param canvas_shape: shape of the canvas on which to draw objects
        """
        self.resolution = resolution
        self.grid_shape = [resolution, resolution, resolution]
        self.grid_divider = [(resolution/2.), (resolution/2.), (resolution/2.)]
        self.primitive_occupancy_min_thres = PRIMITIVE_OCCUPANCY_MIN_THRES
        self.primitive_occupancy_max_thres= PRIMITIVE_OCCUPANCY_MAX_THRES
        self.bbox_limit = BBOX_LIMIT
        self.device = device
        self.resolution_tensor = th.tensor([self.resolution], dtype=th.float32)
        self.zero_tensor = th.tensor([0], dtype=th.float32)
        
        self.mode = mode
        if self.mode == "inverted":
            self.selected_sub_gbc = inverted_sub_gbc
            self.selected_sub_rbb = inverted_sub_rbb
            self.selected_shape_transform = inverted_shape_transform
            self.selected_grid_operation = inverted_grid_operation
            self.selected_rot_operation = inverted_rot_operation
        elif self.mode == "direct":
            self.selected_sub_gbc = direct_sub_gbc
            self.selected_sub_rbb = direct_sub_rbb
            self.selected_shape_transform = direct_shape_transform
            self.selected_grid_operation = direct_grid_operation
            self.selected_rot_operation = direct_rot_operation
            
        self.base_coords = self.get_base_coords()
        if device == "cuda":
            self.set_to_cuda()
        else:
            self.set_to_cpu()

        self.set_to_half()

        
        
        
    def set_to_cuda(self):
        self.device = "cuda"
        self.base_coords = self.base_coords.to(self.device)
        self.zero_tensor = self.zero_tensor.to(self.device)
        self.resolution_tensor = self.resolution_tensor.to(self.device)
        
    def set_to_cpu(self):
        self.device = "cpu"
        self.base_coords = self.base_coords.to(self.device)
        self.zero_tensor = self.zero_tensor.to(self.device)
        self.resolution_tensor = self.resolution_tensor.to(self.device)
        self.tensor_type = th.float32
        
    def set_to_half(self):
        self.base_coords = self.base_coords.half()
        self.zero_tensor = self.zero_tensor.half()
        self.resolution_tensor = self.resolution_tensor.half()
        self.tensor_type = th.float16
        # self.tensor_type = th.FloatTensor

    def set_to_full(self):
        self.base_coords = self.base_coords.float()
        self.zero_tensor = self.zero_tensor.float()
        self.resolution_tensor = self.resolution_tensor.float()
        self.tensor_type = th.float32

        
    def get_base_coords(self):
        res = self.resolution
        coords = self.selected_sub_gbc(res)
        coords = coords.astype(np.float32)
        coords = ((coords + 0.5) / res - 0.5) * 2
        
        coords = th.from_numpy(coords).float().to(self.device)
        return coords
    
    def translate(self, param=[0,0,0], coords=None):
        # param = self.adjust_scale(param)
        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)
        param = th.reshape(param, (1, 1, 1, 3))
        
        if coords is None:
            coords = self.base_coords.clone()
        coords -= param
        return coords

    def scale(self, param=[1, 1, 1], coords=None):
        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)
        param = th.reshape(param, (1, 1, 1, 3))
        
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
        
        coords = th.reshape(coords, (-1, 1, 3))
        
        rotation_matrix = th.unsqueeze(rotation_matrix, 0)
        rotation_matrix = rotation_matrix.expand(coords.shape[0], -1, -1)
        coords = th.bmm(coords, rotation_matrix)
        res = self.resolution
        coords = th.reshape(coords, (res, res, res, 3))
        
        return coords
    
    def quat_rotate(self, param, coords):
        
        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)

        # since we want inverted: 
        q0, q1, q2, q3 = param / th.norm(param)  # normalize quaternion
        q0_sq = q0**2
        q1_sq = q1**2
        q2_sq = q2**2
        q3_sq = q3**2
        q0q1 = q0*q1
        q0q2 = q0*q2
        q0q3 = q0*q3
        q1q2 = q1*q2
        q1q3 = q1*q3
        q2q3 = q2*q3

        r1 = th.stack([2 * (q0_sq + q1_sq) - 1,
                       2 * (q1q2 - q0q3),
                       2 * (q1q3 + q0q2)])

        r2 = th.stack([2 * (q1q2 + q0q3),
                       2 * (q0_sq - q2_sq) - 1,
                       2 * (q2q3 - q0q1)])

        r3 = th.stack([2 * (q1q3 - q0q2),
                       2 * (q2q3 + q0q1),
                       2 * (q0_sq + q3_sq) - 1])

        rotation_matrix = th.stack([r1, r2, r3], axis=0)# .to(param.device)
        
        affine_matrix = th.eye(4, device=self.device)
        affine_matrix[:3, :3] = rotation_matrix
        
        if coords is None:
            coords = self.base_coords.clone()
        points = coords.reshape(-1, 3)
        points = th.cat(
        [points, th.ones(points.shape[0], 1).to(points.device)], dim=1)

        # Apply the affine transformation matrix to the points
        transformed_points = th.matmul(
            points, affine_matrix)[:, :3]

        # Remove the homogeneous coordinate and return the transformed points
        transformed_points = transformed_points.reshape(coords.shape)

        
        return transformed_points
    
        
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
        base_sdf[:, :, :, 0] -= param[0]
        base_sdf[:, :, :, 1] -= param[1]
        base_sdf[:, :, :, 2] -= param[2]
        base_sdf = th.norm(th.clip(base_sdf, min=0), dim=-1)  + th.clip(th.amax(base_sdf, 3), max=0)
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
        xy_vec = th.norm(coords[:, :, :, :2], dim=-1) - r
        height = th.abs(coords[:, :, :, 2]) - h
        vec2 = th.stack([xy_vec, height], -1)
#         vec2[:,:,:,0] -= r
#         vec2[:,:,:,1] -= h
        sdf = th.amax(vec2, 3) + th.norm(th.clip(vec2, min=0.0) + CONVERSION_DELTA, -1)
        return sdf
    
    def draw_infinite_cylinder(self, param=[0.5], coords=None):
        """ Default an infinite cylinder is z-axis aligned.
        """
        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)
        
        # param = self.adjust_scale(param)
        if coords is None:
            coords = self.base_coords.clone()
        r = param[0]
        
        yz_vec = th.norm(coords[:, :, :, 1:], dim=-1) - r
        return yz_vec

    def draw_infinite_cone(self, param=[0.5], coords=None):
        """ About X
        """
        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)
        
        if coords is None:
            coords = self.base_coords.clone()
            
        tan_alpha = param[0]
        tan_alpha = th.abs(tan_alpha)
        distance_to_apex = th.norm(coords, dim=-1).unsqueeze(-1)
        px = coords[:,:,:,0]
        py = coords[:,:,:,1]
        pz = coords[:,:,:,2]
        distance_1 = th.norm(th.stack((pz, py), dim=-1),dim=-1).unsqueeze(-1) - px.unsqueeze(-1) * tan_alpha
        cos_alpha = th.div(1,th.sqrt(1+ tan_alpha**2))
        distance_to_surface = distance_1 * cos_alpha# .unsqueeze(1).repeat(1,points.shape[1],1,1)
        signed_distance = th.where(px.unsqueeze(-1) < 0, distance_to_apex, distance_to_surface).squeeze(-1)
        return signed_distance
        
        
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
        param = th.reshape(param, (1, 1, 1, 3))
        param = param/(th.norm(param) + CONVERSION_DELTA)
        mirrored_coords = mirrored_coords - 2 * th.sum(mirrored_coords * param, -1, keepdim=True) * param
        return mirrored_coords

    def return_inside_coords(self, sdf, normalized=True):
        sdf_np = sdf.cpu().numpy()
        sdf_coords = np.stack(np.where(sdf_np <=0), -1)
        if normalized:
            sdf_coords = -1 + (sdf_coords + 0.5)/self.grid_divider
        return sdf_coords
    
    def return_bounding_box(self, sdf, normalized=True):
        
        sdf_coords = self.return_inside_coords(sdf, normalized=False)
        # print("in bbox, shape of sdf_coords", sdf_coords.shape)
        if sdf_coords.shape[0] == 0:
            if normalized:
                bbox = np.zeros((2, 3), dtype=np.float32)
            else:
                bbox = np.zeros((2, 3), dtype=np.int32)
        else:
            bbox = self.selected_sub_rbb(sdf_coords)
            if normalized:
                bbox = -1 + (bbox + 0.5)/self.grid_divider
        return bbox
    
    def union(self, *args):
        output = th.minimum(*args)
        return output
    
    def intersection(self, *args):
        output = th.maximum(*args)
        return output
    
    def difference(self, sdf_1, sdf_2):
        
        output = th.maximum(sdf_1, -sdf_2)
        return output
    
    def is_valid_primitive(self, sdf):
        sdf_shape = (sdf<=0)
        occupancy_rate = th.sum(sdf_shape)/float(sdf.nelement())
        occupancy_check = False
        bbox_check = False
        if self.primitive_occupancy_min_thres < occupancy_rate < self.primitive_occupancy_max_thres:
            occupancy_check = True
            bbox = self.return_bounding_box(sdf)
            if self.valid_bbox(bbox):
                bbox_check = True
        return bbox_check and occupancy_check

    def is_valid_sdf(self, sdf):
        sdf_shape = (sdf<=0)
        occupancy_rate = th.sum(sdf_shape)/float(sdf.nelement())
        occupancy_check = False
        bbox_check = False
        if self.primitive_occupancy_min_thres < occupancy_rate:
            occupancy_check = True
            bbox = self.return_bounding_box(sdf)
            if self.valid_bbox(bbox):
                bbox_check = True
        return bbox_check and occupancy_check
    
    def valid_bbox(self, bbox):
        bbox_check = np.abs(bbox).max() < self.bbox_limit
        return bbox_check
        
    def shape_translate(self, param, input_shape):
        # Assume Shape to be a RES, RES, RES, 2 object
        # The second should be updated with 0 wherever required. 
        # use input shape as boolean.
        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)
        self.selected_shape_transform(param)    

        coords = self.base_coords.clone()
        coords = th.reshape(coords, (-1, 1, 3))
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
