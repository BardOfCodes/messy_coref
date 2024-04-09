
import os
import torch as th
from CSG.utils.train_utils import arg_parser, load_config, prepare_model_config_and_env, load_all_weights
from CSG.evaluator import Evaluator
from yacs.config import CfgNode as CN
import _pickle as cPickle
import CSG.env as csg_env
from pathlib import Path
import numpy as np
import time
import CSG.bc_trainers as bc_trainers
from stable_baselines3.common import utils
from CSG.bc_trainers.rewrite_engines.subexpr_cache import FaissIVFCache
from CSG.bc_trainers.rewrite_engines.train_state import PladTrainState
from CSG.env.csg3d.sub_parsers import RotationFCSG3DParser
from CSG.env.csg3d.compiler_utils import convert_sdf_samples_to_ply, get_center_offset
from CSG.env.csg3d.compiler import MCSG3DCompiler
from CSG.env.csg3d_shapenet_env import ShapeNetGenerateData
import mcubes
import open3d as o3d
import sys
# from chamfer_distance import ChamferDistance
from chamferdist import ChamferDistance

COMPUTE_ON_TRAIN = False


if __name__ == "__main__":
    args = arg_parser.parse_args()
    config = load_config(args)
    iterative_do_cs = config.EVAL.ITERATIVE_DO_CS
    max_iteration = config.EVAL.N_ITERATIVE
    fcsg_mode = config.BC.CS.FCSG_MODE

    LANGUAGE_NAME = config.LANGUAGE_NAME
    train_env, eval_env, model_info = prepare_model_config_and_env(config)
    model_info['train_state'] = PladTrainState
    logger = utils.configure_logger(1, config.MACHINE_SPEC.LOG_DIR , "EVAL_%s" % config.EXP_NAME, False)
    if args.bs_only:
        policy, lr_scheduler, _, _ = load_all_weights(load_path=config.MODEL.LOAD_WEIGHTS, train_env=train_env, instantiate_model=True, 
                                                    model_info=model_info, device="cuda")
        eval_list = []
        for beam_size in [10]:
            eval = Evaluator(eval_env=eval_env, beam_search=True, beam_k=beam_size, beam_state_size=beam_size, 
                                n_eval_episodes=config.TRAIN.EVAL_EPISODES, eval_freq=1, log_path=config.LOG_DIR + "_beam_%d" % beam_size, 
                                best_model_save_path=config.LOG_DIR + "_beam_%d" % beam_size, beam_selector=config.EVAL.BEAM_SELECTOR, 
                                save_predictions=True,
                                beam_n_proc=config.EVAL.BEAM_N_PROC, beam_n_batch=config.EVAL.BEAM_BATCH_SIZE, exhaustive=config.EVAL.EXHAUSTIVE)
            eval_list.append(eval)
        # It requires the BC trainer? - No simply run the evaluation by itself.

        # Setup the logger
        # model.set_env(train_env)
        # model._setup_learn(1, None)
        policy.optimizer.zero_grad(set_to_none=True)
        policy.eval()
        policy.set_training_mode(False)
        policy.action_dist.distribution = None
        print("starting Eval!")
        for eval in eval_list:
            print("eval for", eval)
            # Assigns the model and logger
            # eval.init_callback(model)
            eval.logger = logger
            eval.n_calls = 1
            # starts eval
            eval._on_step(policy, None)
        
        policy.cpu()
        del policy, lr_scheduler

    # Some basic settings: 

    bc_config = config.BC.clone()
    bc_config.BS.BEAM_SIZE = 10
    bc_config.DO.EXHAUSTIVE = True
    bc_config.GS.EXHAUSTIVE = True
    bc_config.CS.EXHAUSTIVE = True
    quantize = False
    bc_config.DO.ENABLE = False
    bc_config.GS.ENABLE = False
    # bc_config.CS.ENABLE = False
    # bc_config.DO.SAMPLE_COUNT = 100
    # bc_config.DO.N_PROC = 1
    # bc_config.CS.SAMPLE_COUNT = 100
    # bc_config.ENV.MODE = "EVAL"
    # bc_config.ENV.MODE = "TRAIN"

    # Conver the save Dir:
    bc_class = getattr(bc_trainers, bc_config.TYPE)
    bc_trainer = bc_class(
        bc_config=bc_config,
        save_dir=config.SAVE_DIR,
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        seed=config.SEED,
        config=config,
        model_info=model_info,
        demonstrations=None,
        train_env=train_env,
        custom_logger=logger
    )
    bc_trainer.init_data_loader()
    bc_trainer.randomize_rewriters = False
    bc_trainer.best_program_init = True
    bc_trainer.training_data_selection = "BEST"
    bc_trainer.wake_sleep_generator.enable = False
    bc_trainer.code_splice_rewriter.eval_mode = True
    # bc_trainer.code_splice_rewriter.eval_mode = False

    epoch = 1
    if bc_config.CS.ENABLE:
        if COMPUTE_ON_TRAIN:
            bc_trainer.code_splice_rewriter.enable = False
            # bc_trainer.diff_opt_rewriter.enable = False
            epoch = 0

    save_path = config.MODEL.LOAD_WEIGHTS
    train_state = bc_trainer.model_info['train_state']()
    train_state.cur_epoch = epoch

    save_file = config.LOG_DIR + "/eval_predictions_DO_GS"
    if bc_trainer.code_splice_rewriter.enable:
        save_file += "_CS"
    save_file = save_file + ".pkl"
    # load One time value:
    # bc_trainer.best_program_dict, program_list = cPickle.load(open(save_file, "rb"))
    # program_list = bc_trainer.construct_training_data(train_state.tensorboard_step)
    # cPickle.dump([bc_trainer.best_program_dict, program_list], open(save_file, "wb"))
    # print("saving output at %s" % save_file)
    # rewards = [x['reward'] for x in program_list]
    # print("Final Reward", np.mean(rewards))
    # bc_trainer.beam_search_generator.enable = False
    
    bc_trainer.update_best_programs(save_path, train_state, quantize=quantize, log_interval=100)
    
    # bc_trainer.best_program_dict, program_list = cPickle.load(open(save_file, "rb"))
    
    program_list = bc_trainer.construct_training_data(train_state)
    # expression_file = "/home/aditya/projects/rl/weights/iccv/base_main/pcsg3d_srt_2/beam_do_gs_cs_no_tax.pkl"
    # expression_file = "/home/aditya/projects/rl/weights/iccv/base_main/pcsg3d_srt_2/beam_do_gs_cs_3_no_tax.pkl"
    # program_list = cPickle.load(open(expression_file, "rb"))
    # here for each expression save the render mcsg version as well:
    parser = eval_env.envs[0].program_generator.parser
    for ind, program in enumerate(program_list):
        expression = program['expression']
        if "CSG3D" in LANGUAGE_NAME:
            if "MCSG" not in LANGUAGE_NAME:
                mcsg_expr = parser.convert_to_mcsg(expression)
            else:
                mcsg_expr = expression
            program["render_expr"] = mcsg_expr
        else:
            mcsg_expr = None
            program["render_expr"] = mcsg_expr
        if "SA3D" in LANGUAGE_NAME:
            # save the hcsg expression as render expr: 
            
            program["render_expr"] = expression
        
        if "CSG2D" in LANGUAGE_NAME:
            program["render_expr"] = mcsg_expr
            
    rewards = [x['reward'] + 0.001 * len(x['expression']) for x in program_list]
    print("Final Reward", np.mean(rewards))
    rewards = [len(x['expression']) for x in program_list]
    print("Final Length", np.mean(rewards))
    # Save the program file:
    dir_name = os.path.dirname(save_path)
    # file_name = os.path.join(dir_name, "beam_do_gs_cs_max.pkl")
    
    # file_name = os.path.join(dir_name, "beam_demo.pkl")
    # file_name = os.path.join(dir_name, "beam_cs.pkl")
    # cPickle.dump(program_list, open(file_name, "wb"))
    asdf
    
    if iterative_do_cs:
        # pass
        train_state.cur_epoch = 1
        # Get the expressions train
        # Now the train dict is saved:
        bc_trainer.beam_search_generator.enable = False
        bc_trainer.code_splice_rewriter.enable = True
        bc_trainer.code_splice_rewriter.eval_mode = False # so that it loads the recently created exprs
        for iteration in range(max_iteration):
            if iteration == max_iteration -1:
                bc_trainer.code_splice_rewriter.enable = False
            bc_trainer.update_best_programs(save_path, train_state, quantize=quantize, log_interval=100)
            print("Average Score of best training programs is %f" % bc_trainer.bpds.mean_reward)
            program_list = bc_trainer.construct_training_data(train_state)
            rewards = [x['reward'] for x in program_list]
            print("ITER %d: Final Eval Reward" % (iteration+1), np.mean(rewards))
    
    program_list = bc_trainer.construct_training_data(train_state)
    
    # here for each expression save the render mcsg version as well:
    parser = eval_env.envs[0].program_generator.parser
    for ind, program in enumerate(program_list):
        expression = program['expression']
        if "CSG3D" in LANGUAGE_NAME:
            if "MCSG" not in LANGUAGE_NAME:
                mcsg_expr = parser.convert_to_mcsg(expression)
            else:
                mcsg_expr = expression
        program["render_expr"] = mcsg_expr
        
    rewards = [x['reward'] for x in program_list]
    print("Final Reward", np.mean(rewards))
    # Save the program file:
    dir_name = os.path.dirname(save_path)
    file_name = os.path.join(dir_name, "beam_do_gs_cs_3_max.pkl")
    cPickle.dump(program_list, open(file_name, "wb"))
    file_name = os.path.join(dir_name, "beam_do_gs_cs_3_bpd_max.pkl")
    cPickle.dump(bc_trainer.bpds.bpd, open(file_name, "wb"))
    
    asdf 
    ## Save all the outputs in ply form: 
    compiler = eval_env.envs[0].program_generator.compiler
    parser = eval_env.envs[0].program_generator.parser
    # save_dir = "/home/aditya/projects/rl/CSGStumpNet/samples/plane_pred_256"
    # filename = "/home/aditya/projects/rl/CSGStumpNet/data/ShapeNet/03001627/test.lst"
    filename = "/home/aditya/projects/rl/CSGStumpNet/data/ShapeNet/03001627/test.lst"
    dirname = "/home/aditya/projects/rl/CSGStumpNet/data/ShapeNet/03001627/"
    folder_names = open(filename, "r").readlines()
    folder_names = [x.strip() for x in folder_names]
    import sys
    # sys.path.insert(0, "/home/aditya/projects/rl/binvox-rw-py")
    import binvox
    # Create a compiler with a much larger resolution:
    res = 256
    higher_res_compiler = MCSG3DCompiler(resolution=res, scale=res, device=compiler.device, 
                                         draw_mode=compiler.draw_mode)
    phase_config = config.EVAL
    old_program_gen = eval_env.envs[0].program_generator
    new_CSG_CONF = phase_config.ENV.CSG_CONF.clone()
    new_CSG_CONF.RESOLUTION = res
    
    # higher_res_program_gen = ShapeNetGenerateData(data_dir=config.MACHINE_SPEC.DATA_PATH, mode=phase_config.ENV.MODE, n_proc=1, proc_id=0,
    #                                                proportion=config.TRAIN_PROPORTION, program_lengths=old_program_gen.program_lengths, 
    #                                                csg_config=new_CSG_CONF, proportions=old_program_gen.proportions,
    #                                                sampling=phase_config.ENV.SAMPLING, project_root=config.MACHINE_SPEC.PROJECT_ROOT)
    iou_list = []
    distances_1 = []
    distances_2 = []
    chamferDist = ChamferDistance()
    unique_slots = np.unique([x['slot_id'] for x in program_list])
    shapenet_location = "/home/aditya/data/3d_csg/data/"
    names = {}
    for cur_class in unique_slots:
        
        filename = f"/home/aditya/projects/rl/CSGStumpNet/data/ShapeNet/{cur_class.split('_')[0]}/test.lst"
        # dirname = "/home/aditya/projects/rl/CSGStumpNet/data/ShapeNet/03001627/"
        folder_names = open(filename, "r").readlines()
        folder_names = [x.strip() for x in folder_names]
    #     save_file = os.path.join(shapenet_location,"%s/%s_val_names.txt" % (cur_class, cur_class.split('_')[0]) )
    #     cur_names = open(save_file, "r").readlines()
    #     cur_names = [x.strip() for x in cur_names if x.strip() != ""]
        names[cur_class] = folder_names
    # names = folder_names
    
    
    save_dir = os.path.dirname(save_path)
    save_dir = os.path.join(save_dir, "eval_meshes")
    # os.mkdir(save_dir)
    # file_name = os.path.join(dir_name, "beam_10.pkl")
    # save_dir = "/home/aditya/eval_meshes_pr/"
    eval_points_dir = "/home/aditya/eval_points/"
    # gt_points_dir = "/home/aditya/fcsg_gt_points/"
    point_cloud_dir = "/home/aditya/data/csgstump/ShapeNet/"
    final_files = []
    csg_location = []
    for ind, program in enumerate(program_list): 
        # get the action id:
        slot_id = program['slot_id']
        train_id = program['target_id']
        # expression = program['expression']
        # save_name = os.path.join(save_dir, "%d.ply" % train_id)
        # command_list =parser.parse(expression)
        # compiler.march_to_ply(command_list, save_name)
        # compiler._compile(command_list)
        # voxels = (compiler._output <=0)
        # higher_res_compiler._compile(command_list)
        # pred_voxels = (higher_res_compiler._output <=0)
        
        # flip the voxel:
        # pred_voxels = th.flip(pred_voxels, [0])
        # pred_voxels = th.stack([pred_voxels, pred_voxels], -1)
        # pred_voxels = compiler.draw.shape_rotate([0, -90, 0], pred_voxels)
        # pred_voxels = pred_voxels[:, :, :, 0]
        # sdf_values = -(pred_voxels.float() - 1e-4)
        
        # Alternative, load the gt and save the gt:
        # voxels, expr = higher_res_program_gen.get_executed_program(slot_id, train_id, return_numpy=False)
        # voxels, expr = old_program_gen.get_executed_program(slot_id, train_id, return_numpy=False)

        # voxels = th.flip(voxels, [0])
        # voxels = th.stack([voxels, voxels], -1)
        # voxels = higher_res_compiler.draw.shape_rotate([0, -90, 0], voxels)
        # voxels = voxels[:, :, :, 0]
        
        # sdf_values_orig = -(voxels.float() - 1e-4)
        # iou = th.logical_and(voxels, pred_voxels).sum()/th.logical_or(voxels, pred_voxels).sum()
        # iou_list.append(iou)
        # folder = folder_names[train_id]
        # cur_file = os.path.join(dirname, folder, "model.binvox")
        # with open(cur_file, 'rb') as f:
        #     # model = binvox.read_as_3d_array(f)
        #     model = binvox.read_as_3d_array(f)
        
        # res = compiler.draw.grid_shape[0]
        # final_scale = res
        # voxel_grid_origin =  [0.5, 0.5, 0.5]
        # offset = [-x for x in model.translate]
        # offset = np.array(offset)
        # bbox_center, bbox_scale = get_center_offset(pytorch_3d_sdf_tensor=sdf_values_orig.cpu(), 
        #                            voxel_grid_origin=voxel_grid_origin,
        #                            voxel_size=1,
        #                            scale=final_scale
        #                            )
        
        # res = higher_res_compiler.draw.grid_shape[0]
        # final_scale = res/model.scale
        # # offset += center
        # cur_name = names[slot_id][train_id]
        cur_name = names[slot_id][train_id]
        # pred_name = os.path.join(save_dir, "%s_%s.ply" % (slot_id, cur_name))
        # gt_name = os.path.join(save_dir, "gt_%s_%s.ply" % (slot_id, cur_name))
        
        # convert_sdf_samples_to_ply(pytorch_3d_sdf_tensor=sdf_values_orig.cpu(), 
        #                            voxel_grid_origin=voxel_grid_origin,
        #                            voxel_size=1,
        #                            scale=final_scale,
        #                            centered=False,
        #                            bbox_center=bbox_center,
        #                            bbox_scale=bbox_scale,
        #                            ply_filename_out=gt_name)
        # convert_sdf_samples_to_ply(pytorch_3d_sdf_tensor=sdf_values.cpu(), 
        #                            voxel_grid_origin=voxel_grid_origin,
        #                            voxel_size=1,
        #                            scale=res,
        #                            centered=False,
        #                            bbox_center=bbox_center,
        #                            bbox_scale=bbox_scale,
        #                            ply_filename_out=pred_name)
        # Also save the point clouds:
        file_name = os.path.join(point_cloud_dir, "%s/%s/pointcloud.npz" % ( slot_id.split('_')[0], cur_name))
        surface_pointcloud = np.load(file_name)['points']
        
        min_bound = surface_pointcloud.min(axis=0)
        max_bound = surface_pointcloud.max(axis=0)
        loc = (min_bound + max_bound)/2
        scale = np.linalg.norm(max_bound - min_bound)
        # # scale = surface_data['scale']
        # # loc = surface_data['loc']
        surface_pointcloud = surface_pointcloud - loc
        surface_pointcloud = surface_pointcloud / scale
        # # # save as pcd:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(surface_pointcloud)
        pred_name = os.path.join(eval_points_dir, "%s_%s.ply" % (slot_id, cur_name))
        # pred_name = os.path.join(eval_points_dir, "%s_%s.ply" % (slot_id, train_id))
        o3d.io.write_point_cloud(pred_name, pcd)
        
        # # Save the gt points: 
        # file_name = os.path.join(point_cloud_dir, "%s/%s/points.npz" % ( slot_id.split('_')[0], cur_name))
        # load_obj = np.load(file_name)
        # surface_pointcloud = load_obj['points']
        # min_bound = surface_pointcloud.min(axis=0)
        # max_bound = surface_pointcloud.max(axis=0)
        # # scale = surface_data['scale']
        # # loc = surface_data['loc']
        # surface_pointcloud = surface_pointcloud - loc
        # surface_pointcloud = surface_pointcloud / scale
        
        # final_dict = dict(points=surface_pointcloud, occupancies=load_obj['occupancies'])
        # pred_name = os.path.join(gt_points_dir, "%s_%s.pkl" % (slot_id, train_id))
        # cPickle.dump(final_dict, open(pred_name, 'wb'))
        
        
        

        
        # Compute CD
        # Save the meshes
        
        # mesh 2
        
        # load points at random
    #     mesh = o3d.io.read_triangle_mesh(gt_name)
    #     pcd = mesh.sample_points_poisson_disk(4096)
    #     gt_points = np.asarray(pcd.points, dtype=np.float32)
        
    #     mesh = o3d.io.read_triangle_mesh(pred_name)
    #     pcd = mesh.sample_points_poisson_disk(4096)
    #     pred_points = np.asarray(pcd.points, dtype=np.float32)
        
    #     # calculate CD
    #     source_cloud = th.tensor(gt_points).unsqueeze(0).cuda()
    #     target_cloud = th.tensor(pred_points).unsqueeze(0).cuda()
    #     # chamferDist = ChamferDistance()
    #     # distance_1, distance_2 = chamferDist(source_cloud, target_cloud)
    #     # distance_1 = distance_1.mean()
    #     # distance_2 = distance_2.mean()
    #     dist_forward = chamferDist(source_cloud, target_cloud)
    #     dist_backward = chamferDist(source_cloud, target_cloud, reverse=True)
    #     # dist_bidirectional = chamferDist(source_cloud, target_cloud, bidirectional=True)
    #     distances_1.append(dist_forward)
    #     distances_2.append(dist_backward)
        
        
    # cd1 = th.sum(th.stack(distances_1))/len(distances_1)
    # cd2 = th.sum(th.stack(distances_2))/len(distances_2)
    # cd = (cd1+cd2)*0.5
    
    iou = np.mean(iou_list)
    
    print("IOU", iou)
    # print("CD", cd)
        
    program_list = bc_trainer.construct_training_data(train_state)
    rewards = [x['reward'] for x in program_list]
    print("Final Reward", np.mean(rewards))
    
    if COMPUTE_ON_TRAIN:
        bc_config = config.BC.clone()
        bc_config.DO.EXHAUSTIVE = True
        bc_config.GS.EXHAUSTIVE = True
        bc_config.CS.EXHAUSTIVE = True
        # bc_config.CS.MAX_BOOL_COUNT = 15
        bc_config.ENV.MODE = "TRAIN"

        train_logger = utils.configure_logger(1, config.MACHINE_SPEC.LOG_DIR , "BC_EVAL_%s" % config.EXP_NAME, False)
        train_bc_trainer = bc_class(
            bc_config=bc_config,
            save_dir=config.SAVE_DIR,
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            seed=config.SEED,
            config=config,
            model_info=model_info,
            demonstrations=None,
            train_env=train_env,
            custom_logger=train_logger
        )
        # Delete the previous cache:
        train_bc_trainer.code_splice_rewriter.enable = False
        train_bc_trainer.training_data_selection = "BEST"
        train_bc_trainer.randomize_rewriters = False
        train_bc_trainer.best_program_init = True
        train_bc_trainer.wake_sleep_generator.enable = False
        # Only bs mode:
        train_bc_trainer.diff_opt_rewriter.enable = False
        train_bc_trainer.graph_sweep_rewriter.enable = False


        save_file = config.LOG_DIR + "/train_predictions_DO_GS.pkl"
        # train_bc_trainer.init_data_loader():
        train_bc_trainer.update_best_programs(save_path, train_state, quantize=False, log_interval=100)
        program_list = train_bc_trainer.construct_training_data(train_state)
        cPickle.dump([train_bc_trainer.best_program_dict, program_list], open(save_file, "wb"))
        print("Saved the train predictions at %s" % save_file)
        # else:
        #     train_bc_trainer.best_program_dict, program_list = cPickle.load(open(save_file, "rb"))
        rewards = [x['reward'] for x in program_list]
        print("Final Train Reward", np.mean(rewards))
                # Get the different parser
        # Generate cache before cleaning up program graph.
        subexpr_cache = FaissIVFCache(train_bc_trainer.save_dir, train_bc_trainer.code_splice_rewriter.cache_config, eval_mode=False)
        bc_env_class = getattr(csg_env, bc_config.ENV.TYPE)
        temp_env = bc_env_class(config=config, phase_config=bc_config, seed=0, n_proc=1, proc_id=0)
        temp_env.mode = "EVAL"
        if fcsg_mode:
            temp_env.program_generator.parser = RotationFCSG3DParser(temp_env.program_generator.parser.module_path, temp_env.program_generator.parser.device)
        subexpr_cache.load_previous_subexpr_cache = False
        subexpr_cache.generate_cache_and_index(train_bc_trainer.best_program_dict, temp_env)
            
        
        # If perform CS from Train:
        

    
    
    save_dir = "/home/aditya/projects/rl/CSGStumpNet/samples/plane_srt_bs_3ttr"
    
    for ind, program in enumerate(program_list): 
        # get the action id:
        slot_id = program['slot_id']
        train_id = program['target_id']
        expression = program['expression']
        save_name = os.path.join(save_dir, "%d.ply" % train_id)
        command_list =parser.parse(expression)
        # compiler.march_to_ply(command_list, save_name)
        compiler._compile(command_list)
        voxels = (compiler._output <=0)
        
        voxels = th.stack([voxels, voxels], -1)
        voxels = compiler.draw.shape_rotate([0, -90, 0], voxels)
        voxels = voxels[:, :, :, 0]
        sdf_values = -(voxels.float() - 1e-4)
        
        # Alternative, load the gt and save the gt:
        voxels, expr = eval_env.envs[0].program_generator.get_executed_program(slot_id, train_id, return_numpy=False)

        voxels = th.stack([voxels, voxels], -1)
        voxels = compiler.draw.shape_rotate([0, -90, 0], voxels)
        voxels = voxels[:, :, :, 0]
        sdf_values_orig = -(voxels.float() - 1e-4)
        
        
        folder = folder_names[train_id]
        cur_file = os.path.join(dirname, folder, "model.binvox")
        with open(cur_file, 'rb') as f:
            # model = binvox.read_as_3d_array(f)
            model = binvox.read_as_3d_array(f)
        
        res = compiler.draw.grid_shape[0]
        final_scale = res/model.scale
        
        voxel_grid_origin =  [0.5, 0.5, 0.5]
        offset = [-x for x in model.translate]
        offset = np.array(offset)
        center = get_center_offset(pytorch_3d_sdf_tensor=sdf_values_orig.cpu(), 
                                   voxel_grid_origin=voxel_grid_origin,
                                   voxel_size=1,
                                   scale=final_scale,
                                   offset=offset,
                                   )
        offset += center
        convert_sdf_samples_to_ply(pytorch_3d_sdf_tensor=sdf_values.cpu(), 
                                   voxel_grid_origin=voxel_grid_origin,
                                   voxel_size=1,
                                   scale=final_scale,
                                   offset=offset,
                                   centered=False,
                                   ply_filename_out=save_name)
        
    
    # Now if we have to save:
    save_file = config.LOG_DIR + "/eval_predictions_DO_GS_CS.pkl"
    if bc_trainer.code_splice_rewriter.enable:
        save_file += "_CS"
    save_file = save_file + ".pkl"
    bc_trainer.training_data_selection = "BEST"
    program_list = bc_trainer.construct_training_data(train_state)
    rewards = [x['reward'] for x in program_list]
    print("Final Reward", np.mean(rewards))
    ## Save all the outputs in ply form: 
    cPickle.dump([bc_trainer.bpds.best_program_dict, program_list], open(save_file, "wb"))
    print("saving output at %s" % save_file)
