import _pickle as cPickle
import os
from scipy.spatial.transform import Rotation as R
import numpy as np

class CSGStumpConverter():
    
    def __init__(self, dim=32, n_primitive=8):
        self.dim = dim
        self.n_primitives = n_primitive
        
    def get_transform_params(self, quaternion, translate):
        translate = translate * 2
        r = R.from_quat(quaternion)
        euler = r.as_euler('xyz', degrees=True)
        euler[-1]  = - euler[-1]
        
        transforms = [
        "translate(%f, %f, %f)" % (translate[-1], translate[-2], translate[-3]),
        "rotate(%f, %f, %f)" % (euler[0], euler[1], euler[2]),
        ]
        return transforms
    
    def convert_to_mcsg(self, primitive_parameters, intersection_layer_connections, union_layer_connections):
        # TODO: Do you need the bbox for intersection?
        
        
        boxes = primitive_parameters[:10, :]
        cylinder = primitive_parameters[10:18, :] 
        sphere = primitive_parameters[18:26, :]
        cone = primitive_parameters[26:, :]

        primitive_list = []
            
        for i in range(self.n_primitives):
            quaternion = cylinder[:4, i]
            translate = cylinder[4:7, i]
            c_rad = np.abs(cylinder[7:, i])
            transforms = self.get_transform_params(quaternion, translate)
            # Adjust the scales
            c_rad = c_rad * 4
            
            cylinder_expr = [
            "scale(%f, %f, %f)" % (c_rad[0], c_rad, c_rad),
            "infinite_cylinder",
            ]
            expr = transforms + cylinder_expr
            primitive_list.append(expr)


        for i in range(self.n_primitives):
            quaternion = boxes[:4, i]
            translate = boxes[4:7, i]
            box_scale = boxes[7:, i]
            transforms = self.get_transform_params(quaternion, translate)
            # Adjust the scales
            box_scale = np.abs(box_scale * 4)
            box_expr = [
            "scale(%f, %f, %f)" % (box_scale[-1], box_scale[-2], box_scale[-3]),
            "cuboid",
            ]
            expr = transforms + box_expr
            primitive_list.append(expr)


        for i in range(self.n_primitives):
            quaternion = cone[:4, i]
            translate = cone[4:7, i]
            cone_rad = np.abs(cone[7:, i])
            transforms = self.get_transform_params(quaternion, translate)
            # Adjust the scales
            cone_rad = cone_rad * 2
            cone_expr = [
            "scale(1, %f, %f)" % (cone_rad, cone_rad),
            "infinite_cone",
            ]
            expr = transforms + cone_expr
            primitive_list.append(expr)



        for i in range(self.n_primitives):
            quaternion = sphere[:4, i]
            translate = sphere[4:7, i]
            s_scale = sphere[7:, i]
            transforms = self.get_transform_params(quaternion, translate)
            # Adjust the scales
            s_scale = np.abs(s_scale * 4)
            sphere_expr = [
            "scale(%f, %f, %f)" % (s_scale[-1], s_scale[-1], s_scale[-1]),
            "sphere",
            ]
            expr = transforms + sphere_expr
            primitive_list.append(expr)
        # Now create an expression from the intersection list, and union_list:
        overall_expr = []

        # add n unions
        # add the n intersections
        # each intersection = n intersections + draw lists
        tot_unions = np.sum(union_layer_connections).astype(int)  - 1
        unions = ["union",] * tot_unions
        overall_expr += unions

        for i in range(self.dim):
            if union_layer_connections[i] == 1:
                # Add this intersection to the list:
                cur_intersection_row = intersection_layer_connections[:, i]
                cur_intersection_list = ["intersection",] * (np.sum(cur_intersection_row).astype(int) - 1)
                for j in range(self.dim):
                    if cur_intersection_row[j] == 1:
                        cur_intersection_list.extend(primitive_list[j])
                overall_expr.extend(cur_intersection_list)
        return overall_expr