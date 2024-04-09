import sys
sys.path.insert(0, "../ShapeAssembly/code/")
sys.path.insert(0, "../cvpr22_plad/PLAD")

import executors.ShapeAssembly as pladSA
import ShapeAssembly as mainSA
from executors.ex_sa import prog_random_sample, T2I, execute, tokens_to_lines, make_voxels
# from code.utils import sample_surface

import sys
from CSG.env.csg3d.draw import DiffDraw3D
from CSG.env.shape_assembly.draw import SADiffDraw3D
from CSG.env.shape_assembly.compiler import SACompiler
from CSG.env.shape_assembly.parser import SAParser, convert_sa_to_valid_hsa, convert_hsa_to_valid_hsa
from CSG.utils.visualization import viz_points
import torch as th
import os
import numpy as np
import torch
def sample_surface(faces, vs, count, return_normals=True):
    """
    sample mesh surface
    sample method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Args
    ---------
    vs: vertices (batch x nvs x 3d coordinate)
    faces: triangle faces (torch.long) (num_faces x 3)
    count: number of samples
    Return
    ---------
    samples: (count, 3) points in space on the surface of mesh
    face_index: (count,) indices of faces for each sampled point
    """
    if torch.isnan(faces).any() or torch.isnan(vs).any():
        assert False, 'saw nan in sample_surface'

    device = vs.device
    bsize, nvs, _ = vs.shape
    area, normal = face_areas_normals(faces, vs)
    area_sum = torch.sum(area, dim=1)

    assert not (area <= 0.0).any().item(), "Saw negative probability while sampling"
    assert not (area_sum <= 0.0).any().item(), "Saw negative probability while sampling"
    assert not (area > 1000000.0).any().item(), "Saw inf"
    assert not (area_sum > 1000000.0).any().item(), "Saw inf"

    dist = torch.distributions.categorical.Categorical(probs=area / (area_sum[:, None]))
    face_index = dist.sample((count,))

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vs[:, faces[:, 0], :]
    tri_vectors = vs[:, faces[:, 1:], :].clone()
    tri_vectors -= tri_origins.repeat(1, 1, 2).reshape((bsize, len(faces), 2, 3))

    # pull the vectors for the faces we are going to sample from
    face_index = face_index.transpose(0, 1)
    face_index = face_index[:, :, None].expand((bsize, count, 3))
    tri_origins = torch.gather(tri_origins, dim=1, index=face_index)
    face_index2 = face_index[:, :, None, :].expand((bsize, count, 2, 3))
    tri_vectors = torch.gather(tri_vectors, dim=1, index=face_index2)

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = torch.rand(count, 2, 1, device=vs.device, dtype=tri_vectors.dtype)

    # points will be distributed on a quadrilateral if we use 2x [0-1] samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(dim=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = torch.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths[None, :]).sum(dim=2)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    if return_normals:
        samples = torch.cat((samples, torch.gather(normal, dim=1, index=face_index)), dim=2)
        return samples
    else:
        return samples

def face_areas_normals(faces, vs):
    face_normals = torch.cross(
        vs[:, faces[:, 1], :] - vs[:, faces[:, 0], :],
        vs[:, faces[:, 2], :] - vs[:, faces[:, 1], :],
        dim=2,
    )
    face_areas = torch.norm(face_normals, dim=2) + 1e-8
    face_normals = face_normals / face_areas[:, :, None]
    face_areas = 0.5 * face_areas
    return face_areas, face_normals

def main():
    device = "cuda"
    draw = SADiffDraw3D(resolution=32, device=device)
    prog = prog_random_sample(1, 100, 8, 2, ret_voxels=False)[0]

    # for p in prog:
    #     print(p)
    prog = " ".join(prog)
    print(prog)
    print("===========")
    lines = tokens_to_lines(prog.split())
    lines[0] = "bbox = Cuboid(1., 0.2, 1., False)"
    # lines = ['bbox = Cuboid(1, 0.75, 1.0, False)',
    #          'cube0 = Cuboid(0.1, 0.1, 0.1, False)',
    #          'squeeze(cube0, bbox, bbox, bot, 0.5, 0.5)',
    #          'cube1 = Cuboid(1.0, 0.1, 0.1, False)',
    #          "attach(cube1, bbox, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)",
    #          "attach(cube1, cube0, 0.1, 0.1, 1.0, 0.5, 0.5, 0.5)"
    #         ]
    for line in lines:
        print(line)
    print("===========")

    sa_csg_expr = convert_sa_to_valid_hsa(lines.copy())

    for line in sa_csg_expr:
        print(line)
    print("===========")

    device = "cuda"
    draw = SADiffDraw3D(resolution=32, device=device)
    compiler = SACompiler(resolution=32, device=device)
    parser = SAParser(module_path="../../", device=device)

    compiler.set_to_full()
    compiler.reset()

    command_list = parser.parse(sa_csg_expr)
    compiler._compile(command_list)

    # load random chairs:
    chair_loc = "../ShapeAssembly/code/data/chair/"
    chairs = os.listdir(chair_loc)
    random_index = int(np.random.uniform() * len(chairs))
    random_chair_file = os.path.join(chair_loc, chairs[random_index])

    sa = mainSA.ShapeAssembly()
    lines = sa.load_lines(random_chair_file)
    # for ind, line in enumerate(lines):
    #     lines[ind] = line.replace("True", "False")
    # hier, param_dict, param_list = sa.make_hier_param_dict(lines)
    # verts, faces = sa.diff_run(hier, param_dict)
    # samps = sample_surface(faces, verts.unsqueeze(0), 10000)
    # samps = samps.detach().cpu().numpy()[0, :, :3]
    # viz_points(samps)
    norm_lines = [x.strip() for x in lines]
    sa_expression = norm_lines# [1:8]
    diver = 2
    flip = False
    for ind, expr in enumerate(sa_expression):
        if not flip:
            if "Cuboid(" in expr:
                expr_split = expr.split("(")
                param = expr_split[1].split(", ")[:3]
                param = [str(float(x)/diver) for x in param]
                expr = "".join([expr_split[0], "(", ", ".join(param), ", False)"])
            sa_expression[ind] = expr.replace("True", "False")
        else:
            # second leveL
    #         if "Cuboid(" in expr and not 'bbox' in expr:
    #             expr_split = expr.split("(")
    #             param = expr_split[1].split(", ")[:3]
    #             param = [str(float(x)/diver) for x in param]
    #             expr = "".join([expr_split[0], "(", ", ".join(param), ", False)"])
            sa_expression[ind] = expr.replace("True", "False")
        if "}" in expr:
            flip = True
            diver = 0.5
    # sa_expression[11] = 'bbox = Cuboid(1, 1, 1, False)'
    # sa_expression[14] = 'cube2 = Cuboid(0.643, 0.16, 0.589, False)'
    # sa_expression[0] = "bbox = Cuboid( 0.4, 0.8, 0.4, False)"
    for line in sa_expression:
        print(line)
    print("===========")

    sa_csg_expr = convert_hsa_to_valid_hsa(sa_expression.copy())
    command_list = parser.parse(sa_csg_expr)
    # compiler = SACompiler(resolution=32, device="cuda")
    compiler._compile(command_list)
    cuboid = compiler._output
    cuboid_points = draw.return_inside_coords(cuboid)
    # viz_points(cuboid_points)


if __name__ == "__main__":
    main()