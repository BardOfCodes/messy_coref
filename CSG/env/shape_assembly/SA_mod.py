import torch
import re
import numpy as np
import math
import ast
import sys
import faiss
from copy import deepcopy
import torch as th

"""
  This file contains all of the logic in the ShapeAssembly DSL.

  You can execute a ShapeAssembly program as follows:

  > from ShapeAssembly import ShapeAssembly
  > sa = ShapeAssembly()
  > lines = sa.load_lines({path_to_program})
  > sa.run(lines, {out_file_name})

  The classes in this file are:

  Cuboid -> Part Proxies represented as Cuboids in space
  AttPoint -> Points that live within the local coordinate frames of cuboids -> specify where cuboids should attach
  Program -> Parses lines, locally executes lines by creating Cuboids, AttPoints, and changing their attributes. 
  ShapeAssembly -> Entrypoint to language logic

"""

# Params controlling execution behavior
EPS = .01
SMALL_EPS = 1e-4# th.finfo(th.float16).resolution
COS_DIST_THRESH = 0.9
DIM_MIN_LIMIT = 2 / 32.
 
# Helper function: given angle + normal compute a rotation matrix that will accomplish the operation
def getRotMatrix(angle, normal, device="cuda", dtype=th.float16):
    s = torch.sin(angle)# .to(device)
    c = torch.cos(angle)# .to(device)

    if dtype == th.float16:
        s = s.half()
        c = c.half()
        
    nx = normal[0]
    ny = normal[1]
    nz = normal[2]
        
    rotmat = torch.stack((
        torch.stack((c + (1 - c) * nx * nx, (1 - c) * nx * ny - s * nz, (1 - c) * nx * nz + s * ny)),
        torch.stack(((1 - c) * nx * ny + s * nz, c + (1 - c) * ny * ny, (1 - c) * ny * nz - s * nx)),
        torch.stack(((1 - c) * nx * nz - s * ny, (1 - c) * ny * nz + s * nx, c + (1 - c) * nz * nz))
    ))
    return rotmat


# Helper function: Find a minimum rotation from the current direction to the target direction
def findMinRotation(cur, target, device="cuda",dtype=th.float32):
        
    # assert(cur.norm() != 0)
    # assert(target.norm() != 0)
        
    ncur = cur / cur.norm() 
    ntarget = target / target.norm()
        
    normal = torch.cross(ncur, ntarget)

    # co-linear
    if normal.norm() == 0:
        r_x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0, 1.0, 0.0]], device=device, dtype=dtype)
        r_y = torch.tensor([[0.0, 0, 1.0], [0.0, 1.0, 0.0], [ -1.0, 0.0, 0.0]], device=device, dtype=dtype)
        r_z = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
            
        if torch.dot(r_x @ ncur, ncur) != 0:
            cnormal = r_x @ ncur
        elif torch.dot(r_y @ ncur, ncur) != 0:
            cnormal = r_y @ cur
        elif torch.dot(r_z @ ncur, ncur) != 0:
            cnormal = r_z @ ncur

        assert(cnormal.norm() != 0)
        nnormal = cnormal / cnormal.norm()
        angle = torch.tensor(math.pi, device=device, dtype=dtype)

    else:
        
        nnormal = normal / normal.norm()
        angle = torch.acos(torch.dot(ncur, ntarget))
        if angle == 0 or torch.isnan(angle).any():
            return torch.eye(3, device=device, dtype=dtype)

    return getRotMatrix(angle, nnormal, device, dtype)

class Cuboid():
    """
    Cuboids are the base (and only) objects of a ShapeAssembly program. Dims are their dimensions, pos is the center of the cuboid, rfnorm (right face), tfnorm (top face) and ffnorm (front face) specify the orientation of the cuboid. The bounding volume is just a non-visible cuboid. Cuboids marked with the aligned flag behavior differently under attachment operations. 
    """
    def __init__(self, name, aligned = False, vis = True, device="cuda", dtype=th.float32, 
                 dims=None, pos=None, r_mat=None, noisy_eye=None, unit_zero=None, dim_to_scale=None):
        # The default cube is unit, axis-aligned, centered at the origin
        self.device = device
        self.dtype = dtype
        self.dims = dims
        self.pos = pos
        self.r_mat = r_mat
        self.noisy_eye = noisy_eye
        self.unit_zero = unit_zero
        self.dim_to_scale = dim_to_scale
        # Keep track of all attachment obligations this cube has
        self.attachments = []
        self.move_atts = []
        # The bbox is not visible, but is still a cuboid, otherwise this should be True
        self.is_visible = vis
        self.name = name
        self.parent = None
        self.parent_axis = None
        self.aligned = aligned

    # Rotate the cuboid by the rotation matrix
    def rotateCuboid(self, rotation):
        self.r_mat = (rotation @ self.r_mat).T 

    # Get the relative position of global poiunt gpt
    def getRelPos(self, gpt, normalize = False):
        O = self.getPos(self.unit_zero,self.unit_zero, self.unit_zero)
        m = self.dims.unsqueeze(0).expand(3, 3)
        A = (m * self.r_mat.T).clone()# .T# .T
        B = gpt - O
        # This fails at this part. Why??
        A = A + self.noisy_eye # To ensure inverse exists
        if self.dtype == th.float16:
            A = A.float()
            A = A.inverse()
            A = A.half()
            p = A @ B
        else:
            p = A.inverse() @ B
            
        if normalize:
            return torch.clamp(p, 0.0, 1.0)
        
        return p                
    
    # Get the global point specified by relative coordinates x,y,z 
    def getPos(self, x, y, z):
        
        pt = torch.stack((x, y, z))
    
        r = self.r_mat.T

        # t_dims = self.dims
        return (r @ ((pt - .5) * self.dims)) + self.pos

    # Make the cuboid bigger by a multiplied factor of scale (either dim 3 or dim 1)
    def scaleCuboid(self, scale):
        self.dims = self.dims * scale

    # Make the cuboid bigger by an added factor of scale to a specific dimension
    def increaseDim(self, dim, inc):
        s = self.dim_to_scale[dim].clone() * inc
        self.dims = self.dims + s
        # what if Dim becomes 0? 
        # self.dims[self.dims < DIM_MIN_LIMIT] = DIM_MIN_LIMIT
        
    # Move the center of the cuboid by the translation vector
    def translateCuboid(self, translation):
        self.pos = self.pos + translation

    # Return any attachments that are on this cuboid
    def getAttachments(self):
        return self.attachments
    
    # Return the cuboid's parameterization
    def getParams(self):
        return torch.cat((
            self.dims, self.pos, self.r_mat.reshape(-1)
        )) 

class AttPoint():
    """ 
    Attachment Points live with the local coordinate frame [0, 1]^3 of a cuboid. They are used to connect cuboids together.
    """
    def __init__(self, cuboid, x, y, z):
        self.cuboid = cuboid

        self.x = x
        self.y = y
        self.z = z
        self.dim_to_sf = {
            'height': self.getChangeVectorHeight,
            'length': self.getChangeVectorLength,
            'width': self.getChangeVectorWidth,
        }
        self.dim_to_dir = {
            'height': self.getChangeDirHeight,
            'length': self.getChangeDirLength,
            'width': self.getChangeDirWidth,
        }

    # To get the global position, all we need is the cuboid+face info, and the relative uv pos
    def getPos(self):
        return self.cuboid.getPos(self.x, self.y, self.z)
    
    # If we scale the height of the cuboid, what is the rate of change of this AP
    def getChangeVectorHeight(self):
        norm = self.cuboid.r_mat[1]
        return (self.y - .5) * norm

    # If we scale the length of the cuboid, what is the rate of change of this AP
    def getChangeVectorLength(self):
        norm = self.cuboid.r_mat[0]
        return (self.x - .5) * norm
        
    # If we scale the width of the cuboid, what is the rate of change of this AP
    def getChangeVectorWidth(self):
        norm = self.cuboid.r_mat[2]
        return (self.z - .5) * norm
        
    # get rate of change of this AP when we change the specified dimension
    def getChangeVector(self, dim):
        return self.dim_to_sf[dim]()                

    # If we scale the height of the cuboid, what direction does the AP move with
    def getChangeDirHeight(self):
        if self.y > .5:
            return 'top'
        elif self.y < .5:
            return 'bot'
        else:
            return 'none'

    # If we scale the length of the cuboid, what direction does the AP move with
    def getChangeDirLength(self):
        if self.x > .5:
            return 'right'
        elif self.x < .5:
            return 'left'
        else:
            return 'none'    

    # If we scale the width of the cuboid, what direction does the AP move with
    def getChangeDirWidth(self):        
        if self.z > .5:
            return 'front'
        elif self.z < .5:
            return 'back'
        else:
            return 'none'
    
    def getChangeDir(self, dim):
        return self.dim_to_dir[dim]()
    


class Program():
    """
    A program maintains a representation of entire shape, including all of the member cuboids
    and all of the attachment points. The execute function is the entrypoint of text programs.
    """
    of = {
        'right': 'left',
        'left': 'right',
        'top': 'bot',
        'bot': 'top',
        'front': 'back',
        'back': 'front',
    }
    pad = torch.nn.ConstantPad1d((0, 1), 1.0)
    
    def __init__(self, cuboids = {}, ft={}, device="cuda", dtype=th.float32):
        self.device= device
        self.dtype = dtype
        self.cuboids = self.getBoundBox()
        self.cuboids.update(cuboids)
        self.commands = []
        self.parameters = []
        self.att_points = {}
        self.ft=ft

        

            
    # Each program starts off with an invisible bounding box
    def getBoundBox(self):
        bbox = Cuboid("bbox", aligned = True, vis=False, device=self.device, dtype=self.dtype)
                
        return {
            "bbox": bbox
        }
    # Logic for cuboids with no previous attachment. Finds a translation to satisfy the attachment
    def first_attach(self, ap, gpos):
        cur_pos = ap.getPos()
        diff = gpos - cur_pos
        ap.cuboid.translateCuboid(diff)
        return True
        
    # Logic for unaligned cuboids with one previous attachment. Find a scale and rotation to satisfy the attachment
    def second_attach(self, ap, gpos, prev_att):
        p_ap = prev_att[0]
        p_gpos = prev_att[1]
        
        a = p_gpos
        b = ap.getPos()
        c = gpos

        if (b-c).norm() < SMALL_EPS:
            return True
        
        # Increase dimension to fix distance
        dist = (c-a).norm()
        min_dim = 'height'
        min_sf = th.finfo(self.dtype).max
        th_eps = th.finfo(self.dtype).eps

        for dim in ('height', 'width', 'length'):
            
            nsf = ap.getChangeVector(dim)
            psf = p_ap.getChangeVector(dim)

            if nsf.abs().sum() + psf.abs().sum() < SMALL_EPS:
                continue
                        
            cn = b - a
            dn = nsf - psf
            
            at = (dn**2).sum()
            bt = 2 * (cn*dn).sum()
            ct = (cn**2).sum() - (dist**2)

            # Take the positive solution of the quadratic equation
            sf = ((-1 * bt) + (bt**2 - (4*at*ct) ).sqrt()) / ((2 * at) + th_eps)            
            if abs(sf) < abs(min_sf) and (bt**2 - (4*at*ct)) > 0:
                min_sf = sf
                min_dim = dim
                
        if min_sf ==  th.finfo(self.dtype).max:                        
        
            nsf = ap.getChangeVector('height') + \
                  ap.getChangeVector('length') + \
                  ap.getChangeVector('width')
            
            psf = p_ap.getChangeVector('height') + \
                  p_ap.getChangeVector('length') + \
                  p_ap.getChangeVector('width')
            
            cn = b - a
            dn = nsf - psf
            
            at = (dn**2).sum()
            bt = 2 * (cn*dn).sum()
            ct = (cn**2).sum() - (dist**2)

            # Take the positive solution of the quadratic equation
            sf = ((-1 * bt) + (bt**2 - (4*at*ct) ).sqrt()) / ((2 * at) + th_eps)

            if not torch.isnan(sf) and (bt**2 - (4*at*ct)) > 0:            
                ap.cuboid.increaseDim('height', sf)
                ap.cuboid.increaseDim('length', sf)
                ap.cuboid.increaseDim('width', sf)                        

        else:
            ap.cuboid.increaseDim(min_dim, min_sf)

        # Reset the position of the cuboid such that the previous attachment is satisfied
        diff = p_gpos - p_ap.getPos()
        ap.cuboid.translateCuboid(diff)
        
        # find rotation to match points

        nb = ap.getPos() - p_gpos
        nc = c - p_gpos
        
        # If we are already in the correct position, don't rotate
        if nb.norm() == 0 or nc.norm() == 0 or (nb-nc).norm() < SMALL_EPS:
            return True

        rot_mat = findMinRotation(nb, nc, device=self.device, dtype=self.dtype)
        ap.cuboid.rotateCuboid(rot_mat)
        
        # Reset the position of the cuboid such that the attachments are satisfied
        sdiff = p_gpos - p_ap.getPos()
        
        ap.cuboid.translateCuboid(sdiff)

        return True

    # Moves the attach point to the global position
    def attach(self, ap, gpos, oci, oap=None):
        self.free_cube_attach(ap, gpos, oci)

    # Non-aligned attachment
    def free_cube_attach(self, ap, gpos, oci):
        prev_atts = ap.cuboid.getAttachments()
                
        if len(prev_atts) == 0:
            self.first_attach(ap, gpos)
        elif len(prev_atts) == 1:
            self.second_attach(ap, gpos, prev_atts[0])
        else:
            raise ValueError("three attaches not allowed!")
            # self.gen_attach(ap, gpos, prev_atts)

        prev_atts.append((ap, gpos, oci))
        ap.cuboid.move_atts.append((ap, gpos, oci))
        
    # Help function for getting direction of reflect commands
    def getRefDir(self, d):
        bbox = self.cuboids['bbox']
        if d == 'X':
            return bbox.r_mat[0].clone()
        elif d == 'Y':
            return bbox.r_mat[1].clone()
        elif d == 'Z':
            return bbox.r_mat[2].clone()
        else:
            assert False, 'bad reflect argument'

    # Help function for getting direction + scale of translate commands
    def getTransDir(self, d):
        bbox = self.cuboids['bbox']
        if d == 'X':
            return bbox.r_mat[0].clone(), bbox.dims[0].clone()
        elif d == 'Y':
            return bbox.r_mat[1].clone(), bbox.dims[1].clone()
        elif d == 'Z':
            return bbox.r_mat[2 ].clone(), bbox.dims[2].clone()
        else:
            assert False, 'bad reflect argument'
            
    # Given an axis + a center, consructs a tranformation matrix to satisfy reflection
    def getRefMatrixHomo(self, axis, center):

        m = center
        d = axis / axis.norm()

        refmat = torch.stack((
            torch.stack((1 - 2 * d[0] * d[0], -2 * d[0] * d[1], -2 * d[0] * d[2], 2 * d[0] * d[0] * m[0] + 2 * d[0] * d[1] * m[1] + 2 * d[0] * d[2] * m[2])),
            torch.stack((-2 * d[1] * d[0], 1 - 2 * d[1] * d[1], -2 * d[1] * d[2], 2 * d[1] * d[0] * m[0] + 2 * d[1] * d[1] * m[1] + 2 * d[1] * d[2] * m[2])),
            torch.stack((-2 * d[2] * d[0], -2 * d[2] * d[1], 1 - 2 * d[2] * d[2], 2 * d[2] * d[0] * m[0] + 2 * d[2] * d[1] * m[1] + 2 * d[2] * d[2] * m[2]))
        ))

        return refmat

    # Reflect a point p, about center and a direction ndir
    def reflect_point(self, p, center, ndir):
        reflection = self.getRefMatrixHomo(ndir, center)
        posHomo = self.pad(p)
        return reflection @ posHomo
    
    # Execute an attach line, creates two attachment points, then figures out how to best satisfy new constraint
    def executeAttach(self, parse):
        ap1 = AttPoint(
            self.cuboids[parse[0]],
            parse[2],
            parse[3],
            parse[4],
        )

        ap2 = AttPoint(
            self.cuboids[parse[1]],
            parse[5],
            parse[6],
            parse[7],
        )

        ap_pt_name = f'{parse[0]}_to_{parse[1]}'
        # Attach points should have unique names
        while ap_pt_name in self.att_points:
            ap_pt_name += '_n'
        self.att_points[ap_pt_name] = ap2
        
        ap2.cuboid.getAttachments().append((ap2, ap2.getPos(), ap1.cuboid.name))
        self.attach(ap1, ap2.getPos(), ap2.cuboid.name, ap2)

    # Executes a squeeze line by making + executing new Cuboid and attach lines
    def executeSqueeze(self, parse):
        face = parse[3]
        oface = self.getOppFace(face)

        atc1, ato1 = self.getSqueezeAtt(
            face, parse[4], parse[5], parse[1] == 'bbox'
        )

        atc2, ato2 = self.getSqueezeAtt(
            oface, parse[4], parse[5], parse[2] == 'bbox'
        )        
            
        self.executeAttach([parse[0], parse[1], atc1[0], atc1[1], atc1[2], ato1[0], ato1[1], ato1[2]])
        self.executeAttach([parse[0], parse[2], atc2[0], atc2[1], atc2[2], ato2[0], ato2[1], ato2[2]])

    # Helper function for finding opposite face
    def getOppFace(self, face):
        return self.of[face]
    
    # Converts squeeze parameters into parameters needed for the two attachment operators.
    def getSqueezeAtt(self, face, u, v, is_bbox):
        at1, ind, val = self.getFacePos(face)
        # bbox is "flipped"
        if is_bbox:
            rval = 1-val
        else:
            rval = val
        at2 = self.zeros_3.clone()
        q = [u, v] 
        for i in range(3):
            if i == ind:
                at2[i] = rval
            else:
                at2[i] = q.pop(0)

        return at1, at2
    
    
    # Local coordinate frame to center of face conversion
    def getFacePos(self, face):
        return self.ft[face]