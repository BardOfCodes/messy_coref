import torch as th
import numpy as np

def get_rotation_matrix(param, inverted=False):
    # Sequence = Z Y X
    # Inverted sequence - X, Y Z
    sins = th.sin(param)
    coss = th.cos(param)
    sin_a, sin_b, sin_g = sins # np.sin(param)
    cos_a, cos_b, cos_g = coss # np.cos(param)
    
    if inverted:
        row_1 = th.stack([cos_b * cos_g, 
                -cos_b * sin_g,
                sin_b])
        row_2 = th.stack([sin_a * sin_b * cos_g + cos_a * sin_g,
                -sin_a * sin_b * sin_g + cos_a * cos_g,
                -sin_a * cos_b])
        row_3 = th.stack([- cos_a * sin_b * cos_g + sin_a * sin_g, 
                cos_a * sin_b * sin_g + sin_a * cos_g,
                cos_a * cos_b])
    else:
        row_1 = th.stack([cos_b * cos_g, 
                sin_a * sin_b * cos_g - cos_a * sin_g,
                cos_a * sin_b * cos_g + sin_a * sin_g])
        row_2 = th.stack([cos_b * sin_g,
                sin_a * sin_b * sin_g + cos_a * cos_g,
                cos_a * sin_b * sin_g - sin_a * cos_g])
        row_3 = th.stack([- sin_b, 
                sin_a * cos_b,
                cos_a * cos_b])
    rot_matrix = th.stack([row_1, row_2, row_3], 0)
    return rot_matrix

def inverted_sub_rbb(sdf_coords):

    min_x, max_x = np.min(sdf_coords[:,0]), np.max(sdf_coords[:,0])
    min_y, max_y = np.min(sdf_coords[:,1]), np.max(sdf_coords[:,1])
    min_z, max_z = np.min(sdf_coords[:,2]), np.max(sdf_coords[:,2])
    # bbox = np.array([[min_y, min_x, min_z],
    #                 [max_y, max_x, max_z]])
    bbox = np.array([[min_x, min_y, min_z],
                [max_x, max_y, max_z]])
    return bbox

def direct_sub_rbb(sdf_coords):

    min_x, max_x = np.min(sdf_coords[:,0]), np.max(sdf_coords[:,0])
    min_y, max_y = np.min(sdf_coords[:,1]), np.max(sdf_coords[:,1])
    min_z, max_z = np.min(sdf_coords[:,2]), np.max(sdf_coords[:,2])
    bbox = np.array([[min_x, min_y, min_z],
                        [max_x, max_y, max_z]])
    return bbox
  
def inverted_shape_transform(param):
    param[[0, 1, 2]] = param[[1, 0, 2]]
    return param

def direct_shape_transform(param):
    return param

def direct_sub_gbc(res):
    coords = np.stack(np.meshgrid(range(res), range(res), range(res), indexing="ij"), axis=-1)
    return coords
    
def inverted_sub_gbc(res):
    coords = np.stack(np.meshgrid(range(res), range(res), range(res)), axis=-1)
    return coords

def inverted_rot_operation(rot_locs, res):
    normalised_locs_x = rot_locs[0]
    normalised_locs_y = rot_locs[1]
    normalised_locs_z = rot_locs[2]
    grid = th.stack([normalised_locs_z, normalised_locs_x, normalised_locs_y], dim=3).view(1, res, res, res, 3)
    return grid

def direct_rot_operation(rot_locs, res):
    normalised_locs_x = rot_locs[0]
    normalised_locs_y = rot_locs[1]
    normalised_locs_z = rot_locs[2]
    grid = th.stack([normalised_locs_z, normalised_locs_y, normalised_locs_x], dim=3).view(1, res, res, res, 3)
    return grid

def inverted_grid_operation(scaled_coords, res):
    grid = th.stack([scaled_coords[:,:,2], scaled_coords[:,:,0], scaled_coords[:,:,1]], dim=2).view(1, res, res, res, 3)
    return grid

def direct_grid_operation(scaled_coords, res):
    grid = th.stack([scaled_coords[:,:,2], scaled_coords[:,:,1], scaled_coords[:,:,0]], dim=2).view(1, res, res, res, 3)
    return grid
