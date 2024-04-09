
import open3d as o3d
import sys
import os
import numpy as np
import CSG.env as csg_env
import torch as th
from CSG.utils.train_utils import arg_parser, load_config, prepare_model_config_and_env

from CSG.env.csg3d.languages import language_map as language_map_csg3d
from CSG.env.csg3d.shapenet_generator import ShapeNetGenerateData as ShapeNetGenerateData3D
from CSG.env.csg2d.languages import language_map as language_map_csg2d
from CSG.env.csg2d.data_generators import ShapeNetGenerateData as ShapeNetGenerateData2D
from CSG.env.shape_assembly.data_generators import SAShapeNetGenerateData
from CSG.env.csg3d.graph_compiler import GraphicalMCSG3DCompiler
from CSG.env.csg3d.parser import MCSG3DParser
from CSG.env.csg2d.parser import MCSG3DParser
from CSG.env.csg2d.graph_compiler import GraphicalMCSG2DCompiler
from CSG.env.shape_assembly.graph_compiler import GraphicalSACompiler
from CSG.env.csg3d.action_space import MCSGAction3D
from CSG.env.csg2d.action_space import MCSGAction2D

from CSG.env.shape_assembly.parser import SAParser
from CSG.env.shape_assembly.compiler import SACompiler
from CSG.env.shape_assembly.action_space import HSA3DAction
from CSG.env.reward_function import chamfer, iou_func
from CSG.env.csg3d.compiler_utils import convert_sdf_samples_to_ply, get_center_offset
import _pickle as cPickle

def load_pointcloud_names(data_type="test"):
    shapenet_classes = ['03001627_chair', '04379243_table', '02828884_bench', '04256520_couch']
    shapenet_location = "/home/aditya/data/3d_csg/data/"
    names = {}
    for cur_class in shapenet_classes:
        save_file = os.path.join(shapenet_location,"%s/%s_%s_names.txt" % (cur_class, cur_class.split('_')[0], data_type) )
        # FOR CSGSTUMP
        # save_file = os.path.join(shapenet_location,"%s/%s_csgstump_%s_names.txt" % (cur_class, cur_class.split('_')[0], data_type) )
        # FOR NEW DATA
        # save_file = os.path.join(shapenet_location,"%s/%s_new_%s_vox.txt" % (cur_class, cur_class.split('_')[0], data_type) )
        cur_names = open(save_file, "r").readlines()
        cur_names = [x.strip() for x in cur_names if x.strip() != ""]
        names[cur_class] = cur_names
    return names


    
def get_language_specific_generators(config, language_name, compiler, program_generator, skip_64=False):
    # Higher res compiler:
    if "CSG3D" in language_name:
        res = 256
        res_256_compiler = language_map_csg3d[language_name]['compiler'](resolution=res, scale=res, device=compiler.device, 
                                            draw_mode=compiler.draw_mode)
        res = 64
        res_64_compiler = language_map_csg3d[language_name]['compiler'](resolution=res, scale=res, device=compiler.device, 
                                            draw_mode=compiler.draw_mode)
        
        # Higher res generator:
        if not skip_64:
            phase_config = config.EVAL
            new_CSG_CONF = phase_config.ENV.CSG_CONF.clone()
            new_CSG_CONF.RESOLUTION = 64
            res_64_program_gen = ShapeNetGenerateData3D(data_dir=config.MACHINE_SPEC.DATA_PATH, mode=phase_config.ENV.MODE, n_proc=1, proc_id=0,
                                                        proportion=config.TRAIN_PROPORTION, program_lengths=program_generator.program_lengths, 
                                                        csg_config=new_CSG_CONF, proportions=program_generator.proportions, 
                                                        sampling=phase_config.ENV.SAMPLING, project_root=config.MACHINE_SPEC.PROJECT_ROOT)
        else:
            res_64_program_gen = None
        graph_compiler = GraphicalMCSG3DCompiler(resolution=program_generator.compiler.resolution,
                                scale=program_generator.compiler.scale,
                                draw_mode=program_generator.compiler.draw.mode)
        mcsg_parser = MCSG3DParser(program_generator.parser.module_path)
        
        action_class = MCSGAction3D
        
    elif "CSG2D" in language_name:
        res = 256
        res_256_compiler = language_map_csg3d[language_name]['compiler'](resolution=res, scale=res, device=compiler.device, 
                                            draw_mode=compiler.draw_mode)
        res_64_compiler = compiler
        res_64_program_gen = program_generator
        graph_compiler = GraphicalMCSG2DCompiler(resolution=program_generator.compiler.resolution,
                                scale=program_generator.compiler.scale,
                                draw_mode=program_generator.compiler.draw.mode)
        mcsg_parser = MCSG3DParser(program_generator.parser.module_path)
        
        action_class = MCSGAction2D
        # res = 64
        # res_64_compiler = language_map_csg3d[language_name]['compiler'](resolution=res, scale=res, device=compiler.device, 
        #                                     draw_mode=compiler.draw_mode)
        # phase_config = config.EVAL
        # new_CSG_CONF = phase_config.ENV.CSG_CONF.clone()
        # new_CSG_CONF.RESOLUTION = 64
        # res_64_program_gen = ShapeNetGenerateData2D(data_dir=config.MACHINE_SPEC.DATA_PATH, mode=phase_config.ENV.MODE, n_proc=1, proc_id=0,
        #                                             proportion=config.TRAIN_PROPORTION, program_lengths=program_generator.program_lengths, 
        #                                             csg_config=new_CSG_CONF, proportions=program_generator.proportions, 
        #                                             sampling=phase_config.ENV.SAMPLING, project_root=config.MACHINE_SPEC.PROJECT_ROOT)
        
    elif "SA3D" in language_name:
        res = 256
        res_256_compiler = SACompiler(resolution=res, scale=res, device=compiler.device)
        res = 64
        res_64_compiler = SACompiler(resolution=res, scale=res, device=compiler.device)
        
        phase_config = config.EVAL
        new_SA_CONF = phase_config.ENV.SA_CONF.clone()
        res_64_program_gen = SAShapeNetGenerateData(data_dir=config.MACHINE_SPEC.DATA_PATH, mode=phase_config.ENV.MODE, n_proc=1, proc_id=0,
                                                    proportion=config.TRAIN_PROPORTION, program_lengths=program_generator.program_lengths, 
                                                    sa_config=new_SA_CONF, proportions=program_generator.proportions, 
                                                    sampling=phase_config.ENV.SAMPLING, project_root=config.MACHINE_SPEC.PROJECT_ROOT)
        
        graph_compiler = GraphicalSACompiler(resolution=program_generator.compiler.resolution,
                                scale=program_generator.compiler.scale,
                                draw_mode=program_generator.compiler.draw.mode)
        mcsg_parser = SAParser(program_generator.parser.module_path)
        
        action_class = HSA3DAction
        
    graph_compiler.set_to_cuda()
    graph_compiler.set_to_full()
    graph_compiler.reset()
    
    res_256_compiler.set_to_cuda()
    res_256_compiler.set_to_full()
    res_256_compiler.reset()
    
    res_64_compiler.set_to_cuda()
    res_64_compiler.set_to_full()
    res_64_compiler.reset()
    
    if res_64_program_gen:
        res_64_program_gen.compiler.set_to_cuda()
        res_64_program_gen.compiler.set_to_full()
        res_64_program_gen.compiler.reset()

    return res_64_compiler, res_256_compiler, res_64_program_gen, graph_compiler, mcsg_parser, action_class
    
def get_point_clouds(pred_voxels, voxels, res, higher_res, gt_pc_name, num_surface_points=2048):
    save_name = 'tmp.ply'
    # flip the voxel: for PLAD
    # Don't flip for CSGStump
    pred_voxels = th.flip(pred_voxels, [0])
    # pred_voxels = th.flip(pred_voxels, [2])
    sdf_values = -(pred_voxels.float() - 1e-4)
    
    voxels = th.flip(voxels, [0])
    # voxels = th.flip(voxels, [2])
    sdf_values_orig = -(voxels.float() - 1e-4)
    
    
    final_scale = res
    voxel_grid_origin =  [0.5, 0.5, 0.5]
    bbox_center, bbox_scale = get_center_offset(pytorch_3d_sdf_tensor=sdf_values_orig.cpu(), 
                            voxel_grid_origin=voxel_grid_origin,
                            voxel_size=1,
                            scale=final_scale
                            )
    
    convert_sdf_samples_to_ply(pytorch_3d_sdf_tensor=sdf_values.cpu(), 
                            voxel_grid_origin=voxel_grid_origin,
                            voxel_size=1,
                            scale=higher_res,
                            centered=False,
                            bbox_center=bbox_center,
                            bbox_scale=bbox_scale,
                            ply_filename_out=save_name)
    # convert_sdf_samples_to_ply(pytorch_3d_sdf_tensor=sdf_values_orig.cpu(), 
    #                         voxel_grid_origin=voxel_grid_origin,
    #                         voxel_size=1,
    #                         scale=res,
    #                         centered=False,
    #                         bbox_center=bbox_center,
    #                         bbox_scale=bbox_scale,
    #                         ply_filename_out="tmp_gt.ply")

    mesh = o3d.io.read_triangle_mesh(save_name)
    pcd = mesh.sample_points_poisson_disk(num_surface_points)
    pred_points = np.asarray(pcd.points, dtype=np.float32)
    
    points = o3d.io.read_point_cloud(gt_pc_name)
    points = np.asarray(points.points)# .pointsE
    surface_selection_index = np.random.randint(0, points.shape[0], num_surface_points)
    # points[:, 0] = -points[:, 0]
    gt_points = points[surface_selection_index].astype(np.float32)
    
    return pred_points, gt_points

def show_logs(per_program_metrics):
    
    ious = [x['iou'] for x in per_program_metrics]
    ious_64 = [x['iou_64'] for x in per_program_metrics]
    chamfer_dist = [x['chamfer_dist'] for x in per_program_metrics]
    program_lengths = [x['program_length'] for x in per_program_metrics]
    sm = [x['SM'] for x in per_program_metrics]
    sm_diff = [x['SM_DIFF'] for x in per_program_metrics]
    # calculate objective:
    objective = [x['objective'] for x in per_program_metrics]
    print("objective", np.mean(objective), np.median(objective))
    print('iou', np.mean(ious), np.median(ious))
    print('iou_64', np.mean(ious_64), np.median(ious_64))
    print('chamfer_dist', np.mean(chamfer_dist), np.median(chamfer_dist))
    print("program_length", np.mean(program_lengths), np.median(program_lengths))
    print("SM-ALL", np.mean(sm), np.median(sm))
    print("SM-diff-ALL", np.mean(sm_diff), np.median(sm_diff))
        
def measure_code_cohesion(program_list):
    """
    Average Frequency of unique expressions
    Average num. unique expressions.
    Important parameters:
        * Quantization
        * Count Primitives or not
    """

def measure_all_metrics():
    ...
    

def measure_spatial_modularity(program_list):
    ...
    
def feature_usage(program_list):
    # Does it use the features we depend on?
    ...
    
    
