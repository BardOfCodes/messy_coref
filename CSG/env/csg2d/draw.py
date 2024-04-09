import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from CSG.env.csg3d.constants import PRIMITIVE_OCCUPANCY_MAX_THRES, PRIMITIVE_OCCUPANCY_MIN_THRES, BBOX_LIMIT, CONVERSION_DELTA
from CSG.env.csg3d.draw import DiffDraw3D
class DiffDraw2D(DiffDraw3D):
    
    def __init__(self, resolution=64, device="cpu", mode="direct"):
        """
        Helper function for drawing the canvases.
        DRAW is always in a Unit Canvas (BBox -1 to 1).
        :param canvas_shape: shape of the canvas on which to draw objects
        """
        self.resolution = resolution
        self.grid_shape = [resolution, resolution]
        self.grid_divider = [(resolution/2.), (resolution/2.)]
        self.primitive_occupancy_min_thres = PRIMITIVE_OCCUPANCY_MIN_THRES
        self.primitive_occupancy_max_thres= PRIMITIVE_OCCUPANCY_MAX_THRES
        self.bbox_limit = BBOX_LIMIT
        self.device = device
        self.resolution_tensor = th.tensor([self.resolution], dtype=th.float32)
        self.zero_tensor = th.tensor([0], dtype=th.float32)
        
        self.mode = mode
            
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
        
        coords = np.stack(np.meshgrid(range(res), range(res), indexing="ij"), axis=-1)
        coords = coords.astype(np.float32)
        coords = ((coords + 0.5) / res - 0.5) * 2
        
        coords = th.from_numpy(coords).float().to(self.device)
        return coords
    
    def translate(self, param=[0, 0], coords=None):
        # param = self.adjust_scale(param)
        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)
        param = th.reshape(param, (1, 1, 2))
        
        if coords is None:
            coords = self.base_coords.clone()
        coords -= param
        return coords

    def scale(self, param=[1, 1], coords=None):
        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)
        param = th.reshape(param, (1, 1, 2))
        
        if coords is None:
            coords = self.base_coords.clone()
        coords = coords / param
        return coords
    
    def rotate(self, param=[0], coords=None, units="degree", inverted=False):
        
        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)
        if units == "degree":
            param = (param * np.pi) / 180.0

        # since we want inverted: 
        param = - param
        if coords is None:
            coords = self.base_coords.clone()

        sin = th.sin(param)
        cos = th.cos(param)
        rot_coords = coords.clone()
        try:
            rot_coords[:,:, 0] = coords[:,:, 0] * cos - coords[:,:, 1] * sin
            rot_coords[:,:, 1] = coords[:,:, 0] * sin + coords[:,:, 1] * cos
        except:
            print("WUT")
        return rot_coords
    
        
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
    
    def draw_cuboid(self, param=[0.5, 0.5], coords=None):
        
        # param = self.adjust_scale(param)
        if coords is None:
            coords = self.base_coords.clone()
            
        base_sdf = th.abs(coords)
        base_sdf[:, :, 0] -= param[0]
        base_sdf[:, :, 1] -= param[1]
        base_sdf = th.norm(th.clip(base_sdf, min=0), dim=-1)  + th.clip(th.amax(base_sdf, 2), max=0)
        return base_sdf
    
    def draw_cylinder(self, param=[], coords=None):
        """ Default cylinder is z-axis aligned.
        """
        raise ValueError("Cylinder is invalid command in 2D CSG")

    
    def draw_ellipsoid(self, param=[], coords=None):
        """ Since it involves scaling its not a proper sdf!
        """
        raise ValueError("Cylinder is invalid command in 2D CSG (simply use scaled sphere)")
        
    def mirror_coords(self, param=[1, 1], coords=None, mode="euclidian"):
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
        param = th.reshape(param, (1, 1, 2))
        param = param/(th.norm(param) + CONVERSION_DELTA)
        mirrored_coords = mirrored_coords - 2 * th.sum(mirrored_coords * param, -1, keepdim=True) * param
        return mirrored_coords
    
    def return_bounding_box(self, sdf, normalized=True):
        
        sdf_coords = self.return_inside_coords(sdf, normalized=False)
        # print("in bbox, shape of sdf_coords", sdf_coords.shape)
        if sdf_coords.shape[0] == 0:
            if normalized:
                bbox = np.zeros((2, 2), dtype=np.float32)
            else:
                bbox = np.zeros((2, 2), dtype=np.int32)
        else:

            min_x, max_x = np.min(sdf_coords[:,0]), np.max(sdf_coords[:,0])
            min_y, max_y = np.min(sdf_coords[:,1]), np.max(sdf_coords[:,1])
            bbox = np.array([[min_x, min_y],
                                [max_x, max_y]])
            if normalized:
                bbox = -1 + (bbox + 0.5)/self.grid_divider
        return bbox
        
    def shape_translate(self, param, input_shape):
        # Assume Shape to be a RES, RES, RES, 2 object
        # The second should be updated with 0 wherever required. 
        # use input shape as boolean.
        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)

        coords = self.base_coords.clone()
        # coords = th.reshape(coords, (-1, 1, 2))
        param = th.reshape(param, (1, 1, 2))
        scaled_coords = coords - param
        
        res = self.resolution
        grid = th.stack([scaled_coords[:,:,0], scaled_coords[:,:,1]], dim=2).view(1, res, res, 2)
        
        pred_shape = self._shape_sample(grid, input_shape)
        return pred_shape
        

    def shape_rotate(self, param, input_shape, units='degree', inverted=True):
        # Assume Shape to be a RES, RES, RES, 2 object


        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device) 
        if units == "degree":
            param = (param * np.pi) / 180.0
        # This is also done in normal rotate.
        param = - param

        coords = self.base_coords.clone()
        sin = th.sin(param)
        cos = th.cos(param)
        rot_coords = coords.clone()
        rot_coords[:,:, 0] = coords[:,:, 0] * cos - coords[:,:, 1] * sin
        rot_coords[:,:, 1] = coords[:,:, 0] * sin + coords[:,:, 1] * cos

        res = self.resolution
        grid = th.stack([rot_coords[:,:,0], rot_coords[:,:,1]], dim=2).view(1, res, res, 2)
        
        
        pred_shape = self._shape_sample(grid, input_shape)

        return pred_shape
        
    def shape_scale(self, param, input_shape):

        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)

        coords = self.base_coords.clone()
        # coords = th.reshape(coords, (-1, 1, 2))
        scaled_coords = coords / param
        res = self.resolution
        grid = th.stack([scaled_coords[:,:,0], scaled_coords[:,:,1]], dim=2).view(1, res, res, 2)
        
        pred_shape = self._shape_sample(grid, input_shape)

        return pred_shape

    def _shape_sample(self, grid, input_shape):

        input_shape_sampling = input_shape.permute(2, 0, 1)
        input_shape_sampling = input_shape_sampling.to(self.tensor_type)
        input_shape_sampling = input_shape_sampling.unsqueeze(0)
        pred_shape = F.grid_sample(input=input_shape_sampling, grid=grid, mode='nearest',  align_corners=False)
        # pred_shape = pred_shape > 0.0
        pred_shape = pred_shape.bool()
        pred_shape = pred_shape.squeeze(0)
        grid_mask = (th.max(th.abs(grid[0]), -1)[0] <= 1)
        pred_shape[1] = th.logical_and(pred_shape[1], grid_mask)

        pred_shape = pred_shape.permute(1, 2, 0)
        return pred_shape

    def shape_mirror(self, param, input_shape):

        if not isinstance(param, th.autograd.Variable):
            param = th.tensor(param, dtype=self.tensor_type, device=self.device)
        coords = self.base_coords.clone()
        mirrored_coords = coords.clone()
        
        param = th.reshape(param, (1, 1, 2))
        param = param/(th.norm(param) + CONVERSION_DELTA)
        mirrored_coords = mirrored_coords - 2 * th.sum(mirrored_coords * param, -1, keepdim=True) * param

        coords = th.reshape(coords, (-1, 1, 2))
        mirrored_coords = th.reshape(mirrored_coords, (-1, 1, 2))

        res = self.resolution
        grid = th.stack([mirrored_coords[:,:,0], mirrored_coords[:,:,1]], dim=2).view(1, res, res, 2)
        pred_shape = self._shape_sample(grid, input_shape)

        return pred_shape
    
        
    def union_invert(self, target, other_child, index=0):
        # Doesnt matter which index is used:
        target_shape = target[:,:,0]
        prior_mask = target[:,:,1]
        masked_region = self.shape_intersection(target_shape, other_child)
        output_mask = th.logical_and(prior_mask,  ~masked_region)
        output_shape = th.stack([target_shape, output_mask], 2)
        return output_shape

        
    def difference_invert(self, target, other_child, index=0):
        target_shape = target[:,:,0]
        prior_mask = target[:,:,1]

        if index == 0:
            output_target = target_shape
            output_mask = th.logical_and(prior_mask, ~other_child)
        else:
            output_target = ~target_shape
            masked_region = self.shape_union(target_shape, other_child)
            output_mask = th.logical_and(prior_mask, masked_region)
        output_shape = th.stack([output_target, output_mask], 2)
        return output_shape

    def intersection_invert(self, target, other_child, index=0):
        target_shape = target[:,:,0]
        prior_mask = target[:,:,1]
        masked_region = self.shape_union(target_shape, other_child)
        output_mask = th.logical_and(prior_mask, masked_region)
         # (1 - other_child)
        # output_target = self.shape_intersection(target_shape, other_child)
        output_shape = th.stack([target_shape, output_mask], 2)
        return output_shape
