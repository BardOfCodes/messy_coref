import pygltflib
import numpy as np
from stl import mesh
from math import sqrt
import numpy as np
import plyfile
import skimage.measure
import time
import logging
import torch as th


def write_ply_triangle(name: str, vertices: np.ndarray, triangles: np.ndarray):
    fout = open(name, "w")
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face " + str(len(triangles)) + "\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(
            str(vertices[ii, 0])
            + " "
            + str(vertices[ii, 1])
            + " "
            + str(vertices[ii, 2])
            + "\n"
        )
    for ii in range(len(triangles)):
        fout.write(
            "3 "
            + str(triangles[ii, 0])
            + " "
            + str(triangles[ii, 1])
            + " "
            + str(triangles[ii, 2])
            + "\n"
        )
    fout.close()

def get_center_offset(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    offset=None,
    scale=None,
):
    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    # verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
    #     numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    # )
    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3, method = "lewiner"
    )
    # verts = verts * (32 /0.6125755)
    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # center it
    min_x, max_x = np.min(mesh_points[:,0]), np.max(mesh_points[:,0])
    min_y, max_y = np.min(mesh_points[:,1]), np.max(mesh_points[:,1])
    min_z, max_z = np.min(mesh_points[:,2]), np.max(mesh_points[:,2])
    bbox = np.array([[min_x, min_y, min_z],
                        [max_x, max_y, max_z]])
    bbox_scale = np.linalg.norm(bbox[0] - bbox[1])
    
    bbox_center = bbox.mean(0)
    return bbox_center, bbox_scale
    

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_size,
    ply_filename_out,
    voxel_grid_origin=None,
    offset=None,
    scale=None,
    centered=False,
    bbox_center=None,
    bbox_scale=None
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    # verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
    #     numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    # )
    # Add padding, and adjust it to the original size
    padded_array = np.pad(numpy_3d_sdf_tensor, 1, mode='constant', constant_values=0.1)
    
    verts, faces, normals, values = skimage.measure.marching_cubes(
        padded_array, level=0.0, spacing=[voxel_size,] * 3, method = "lorensen"
    )
    verts -= voxel_size
    
    
    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    if voxel_grid_origin is None:
        voxel_grid_origin = np.zeros(3)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # center it
    if centered:
        min_x, max_x = np.min(mesh_points[:,0]), np.max(mesh_points[:,0])
        min_y, max_y = np.min(mesh_points[:,1]), np.max(mesh_points[:,1])
        min_z, max_z = np.min(mesh_points[:,2]), np.max(mesh_points[:,2])
        bbox = np.array([[min_x, min_y, min_z],
                            [max_x, max_y, max_z]])
        bbox_scale = np.linalg.norm(bbox[0] - bbox[1])
        
        bbox_center = bbox.mean(0)
        mesh_points -= bbox_center
        mesh_points /= bbox_scale
    else:
        mesh_points -= bbox_center
        mesh_points /= bbox_scale
        
    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )

        

def normalize(vector):
    norm = 0
    for i in range(0, len(vector)):
        norm += vector[i] * vector [i]
    norm = sqrt(norm)
    for i in range(0, len(vector)):
        vector[i] = vector[i] / norm

    return vector


def convert_stl_to_gltf(input_file, output_file):
    ''' Reference: https://stackoverflow.com/questions/66341118/how-do-i-import-an-stl-into-pygltflib
    '''
    stl_mesh = mesh.Mesh.from_file(input_file)

    stl_points = []
    for i in range(0, len(stl_mesh.points)): # Convert points into correct numpy array
        stl_points.append([stl_mesh.points[i][0],stl_mesh.points[i][1],stl_mesh.points[i][2]])
        stl_points.append([stl_mesh.points[i][3],stl_mesh.points[i][4],stl_mesh.points[i][5]])
        stl_points.append([stl_mesh.points[i][6],stl_mesh.points[i][7],stl_mesh.points[i][8]])

    points = np.array(
        stl_points,
        dtype="float32",
    )

    stl_normals = []
    for i in range(0, len(stl_mesh.normals)): # Convert points into correct numpy array
        normal_vector = [stl_mesh.normals[i][0],stl_mesh.normals[i][1],stl_mesh.normals[i][2]]
        normal_vector = normalize(normal_vector)
        stl_normals.append(normal_vector)
        stl_normals.append(normal_vector)
        stl_normals.append(normal_vector)

    normals = np.array(
        stl_normals,
        dtype="float32"
    )

    points_binary_blob = points.tobytes()
    normals_binary_blob = normals.tobytes()

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[
            pygltflib.Mesh(
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(POSITION=0, NORMAL=1), indices=None
                    )
                ]
            )
        ],
        accessors=[
            pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.FLOAT,
                count=len(points),
                type=pygltflib.VEC3,
                max=points.max(axis=0).tolist(),
                min=points.min(axis=0).tolist(),
            ),
            pygltflib.Accessor(
                bufferView=1,
                componentType=pygltflib.FLOAT,
                count=len(normals),
                type=pygltflib.VEC3,
                max=None,
                min=None,
            ),
        ],
        bufferViews=[
            pygltflib.BufferView(
                buffer=0,
                byteOffset=0,
                byteLength=len(points_binary_blob),
                target=pygltflib.ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(points_binary_blob),
                byteLength=len(normals_binary_blob),
                target=pygltflib.ARRAY_BUFFER,
            ),
        ],
        buffers=[
            pygltflib.Buffer(
                byteLength=len(points_binary_blob) + len(normals_binary_blob)
            )
        ],
    )
    gltf.set_binary_blob(points_binary_blob + normals_binary_blob)
    print("Saving gltf to ", output_file)
    gltf.save(output_file)


def get_reward(new_canvas, target):
    # conver to shape:
    new_shape = (new_canvas <0)
    R = th.sum(th.logical_and(new_shape, target)) / \
        (th.sum(th.logical_or(new_shape, target)) + 1e-6)
    return R.item()