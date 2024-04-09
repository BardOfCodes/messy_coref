
import open3d as o3d
import sys
import os
from chamfer_distance import ChamferDistance
import numpy as np
import CSG.env as csg_env
import torch as th
from CSG.utils.train_utils import arg_parser, load_config, prepare_model_config_and_env


from CSG.env.reward_function import chamfer, iou_func
from CSG.env.csg3d.compiler_utils import convert_sdf_samples_to_ply, get_center_offset
import _pickle as cPickle
# import metric_funcs
from scripts.metric_funcs import load_pointcloud_names, get_language_specific_generators, get_point_clouds, show_logs

POINT_CLOUD_LOC = "/home/aditya/eval_points/"
# POINT_CLOUD_LOC = "/home/aditya/fcsg_eval_points/"



if __name__ == "__main__":
    # measure_cd()
    
    th.backends.cudnn.benchmark = True
    try:
        th.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    args = arg_parser.parse_args()
    config = load_config(args)
    print(config)
    
    multiple_program_names = config.EVAL.PROGRAM_LISTS
    multiple_program_lists = []
    print(multiple_program_names)
    for prog_list_name in multiple_program_names:
        program_list = cPickle.load(open(prog_list_name, 'rb'))
        multiple_program_lists.append(program_list)
    
    LANGUAGE_NAME = config.LANGUAGE_NAME
    
    LENGTH_TAXES = config.EVAL.LENGTH_ALPHAS
    # create env:
    eval_env_class = getattr(csg_env, config.EVAL.ENV.TYPE)
    eval_env = eval_env_class(config=config, phase_config=config.EVAL, seed=0, n_proc=1, proc_id=0)
    
    parser = eval_env.program_generator.parser
    compiler = eval_env.program_generator.compiler
    program_generator = eval_env.program_generator
    
    compiler.set_to_cuda()
    compiler.set_to_full()
    compiler.reset()
    
    if LANGUAGE_NAME == "FCSG3D":
        skip_64 = True
    else:
        skip_64 = False
    res_64_compiler, res_256_compiler, res_64_program_gen, \
        graph_compiler, mcsg_parser, action_class = get_language_specific_generators(config, LANGUAGE_NAME, compiler, program_generator, skip_64)
    
    if '3D' in LANGUAGE_NAME:
        # names = load_pointcloud_names(data_type="test")
        names = load_pointcloud_names(data_type="val")
        
    cd_1 = []
    cd_2 = []
    # per program annotation
    for top_ind, program_list in enumerate(multiple_program_lists):
        
        top_metric_set = {}
        per_program_metrics = []
        # per program metrics
        for ind, program in enumerate(program_list):
            if ind!= 0 and ind % 100 == 0:
                print("cur ind %d" % ind)
                show_logs(per_program_metrics)
            new_metrics = {}
            slot_id = program['slot_id']
            target_id = program['target_id']
            expression = program['expression']
            
            new_metrics['slot_id'] = slot_id
            new_metrics['target_id'] = target_id
            new_metrics['expression'] = expression
            # Reconstruction metrics:
            
            # get IOU:
            gt_voxels, expr = eval_env.program_generator.get_executed_program(slot_id, target_id, return_numpy=False)
            command_list =parser.parse(expression)
            compiler._compile(command_list)
            pred_voxels = (compiler._output <=0)
            
            iou = th.logical_and(pred_voxels, gt_voxels).sum() / th.logical_or(pred_voxels, gt_voxels).sum()
            new_metrics['iou'] = iou.item()
            
            # per_program_metrics.append(new_metrics)
            # continue
             
            if "3D" in LANGUAGE_NAME and "F" not in LANGUAGE_NAME:
                # get IOU 64
                
                gt_voxels_64, expr = res_64_program_gen.get_executed_program(slot_id, target_id, return_numpy=False)
                gt_voxels_64 = th.stack([gt_voxels_64, gt_voxels_64], -1)
                gt_voxels_64 = res_64_compiler.draw.shape_rotate([0, -90, 0], gt_voxels_64)
                gt_voxels_64 = gt_voxels_64[:, :, :, 0]
                res_64_compiler._compile(command_list)
                pred_voxels_64 = (res_64_compiler._output <=0)
                               
                iou_64 = th.logical_and(pred_voxels_64, gt_voxels_64).sum() / th.logical_or(pred_voxels_64, gt_voxels_64).sum()
            
            else:
                iou_64 = iou
                
            new_metrics['iou_64'] = iou_64.item()
                
            # CD
            if "3D" in LANGUAGE_NAME:
                # gt_pc_name = os.path.join(POINT_CLOUD_LOC, "%s_%s.ply" % (slot_id, target_id))
                cur_name = names[slot_id][target_id]
                gt_pc_name = os.path.join(POINT_CLOUD_LOC, "%s_%s.ply" % (slot_id, cur_name))
                res = 32
                try:
                    res_256_compiler._compile(command_list)
                    pred_voxels = (res_256_compiler._output <=0)
                    higher_res = 256
                    pred_points, gt_points = get_point_clouds(pred_voxels, gt_voxels, res, higher_res, gt_pc_name)
                except:
                    try:
                        compiler._compile(command_list)
                        pred_voxels = (compiler._output <=0)
                        higher_res = 32
                        pred_points, gt_points = get_point_clouds(pred_voxels, gt_voxels, res, higher_res, gt_pc_name)
                    except:
                        # create a dummy in the center: or skip?
                        print("FAILED HERE!!")
                        tmp_expression = ["sphere"]
                        tmp_command_list =parser.parse(tmp_expression)
                        compiler._compile(tmp_command_list)
                        pred_voxels = (compiler._output <=0)
                        higher_res = 32
                        pred_points, gt_points = get_point_clouds(pred_voxels, gt_voxels, res, higher_res, gt_pc_name)
                        
                
                # calculate CD
                source_cloud = th.tensor(gt_points).unsqueeze(0).cuda()
                target_cloud = th.tensor(pred_points).unsqueeze(0).cuda()
                chamferDist = ChamferDistance()
                distance_1, distance_2 = chamferDist(source_cloud, target_cloud)
                distance_1 = distance_1.mean()
                distance_2 = distance_2.mean()
                # cd_1.append(distance_1.item())
                # cd_2.append(distance_2.item())
                # print("average: %f" % ((sum(cd_1)/len(cd_1) + sum(cd_2)/len(cd_2))*500))
                chamfer_dist = (distance_1+distance_2) * 500
                chamfer_dist = chamfer_dist.item()
        
            else:
                target_canvas = gt_voxels.data.cpu().numpy()
                predicted_canvas = pred_voxels.data.cpu().numpy()
                chamfer_dist = chamfer(target_canvas[None, :, :], predicted_canvas[None, :, :])[0]

                
            new_metrics['chamfer_dist'] = chamfer_dist
            # print(np.mean([x['chamfer_dist'] for x in per_program_metrics]), )
            # print(np.median([x['chamfer_dist'] for x in per_program_metrics]), )
            # print(np.mean([x['iou'] for x in per_program_metrics]), )
            # CQ metrics: 
            # ABV:
            # Extract all subexpressions: 
            if "CSG3D" in LANGUAGE_NAME:
                if "MCSG" not in LANGUAGE_NAME:
                    mcsg_expr = program_generator.parser.convert_to_mcsg(expression)
                else:
                    mcsg_expr = expression
                new_metrics['program_length'] = len(expression)
                command_list = mcsg_parser.parse(mcsg_expr)
                graph = graph_compiler.command_tree(command_list, target=None, enable_subexpr_targets=False,
                                                        add_splicing_info=True)
                volume, num_exprs = 0, 0
                # CONSIDER ALL
                for node_id in range(1, len(graph.nodes)):
                    cur_node = graph.nodes[node_id]
                    # Execute expression and get the bbox
                    cmd_list = graph_compiler.tree_to_command(graph, cur_node)
                    graph_compiler.reset()
                    graph_compiler._compile(cmd_list)
                    bbox = graph_compiler.draw.return_bounding_box(graph_compiler._output)
                    dim = (bbox[1] - bbox[0]) / 32
                    volume += dim.prod(0) # dim[0] * dim[1] * dim[2]
                    num_exprs += 1
                new_metrics['abv_all'] = volume / num_exprs
                new_metrics['all_volumes'] = volume
                new_metrics['all_n_subexprs'] = num_exprs
                new_metrics['SM_DIFF'] = num_exprs / volume
                
                volume, num_exprs = 0, 0
                # DONT CONSIDER FULL EXPRESSION or LEAF EXPRESSIONS
                for node_id in range(1, len(graph.nodes)):
                    cur_node = graph.nodes[node_id]
                    if cur_node['type'] in ["B", "M", "T"]:
                        # Execute expression and get the bbox
                        cmd_list = graph_compiler.tree_to_command(graph, cur_node)
                        graph_compiler.reset()
                        graph_compiler._compile(cmd_list)
                        bbox = graph_compiler.draw.return_bounding_box(graph_compiler._output)
                        dim = (bbox[1] - bbox[0]) # / 32
                        volume += dim.prod(0) # dim[0] * dim[1] * dim[2]
                        num_exprs += 1
                new_metrics['abv_valid_subsets'] = volume / num_exprs
                new_metrics['valid_volumes'] = volume
                new_metrics['valid_n_subexprs'] = num_exprs
                new_metrics['SM'] = num_exprs / volume
                
            elif "SA3D" in LANGUAGE_NAME:
                new_metrics['program_length'] = len(expression)
                command_list = mcsg_parser.parse(expression)
                graph = graph_compiler.command_tree(command_list, target=None, enable_subexpr_targets=False,
                                                        add_splicing_info=True)
                volume, num_exprs = 1, 1
                # CONSIDER ALL
                # for node_id in range(1, len(graph.nodes)):
                #     cur_node = graph.nodes[node_id]
                #     # Execute expression and get the bbox
                #     cmd_list = graph_compiler.tree_to_command(graph, cur_node)
                #     graph_compiler.reset()
                #     graph_compiler._compile(cmd_list)
                #     bbox = graph_compiler.draw.return_bounding_box(graph_compiler._output)
                #     dim = (bbox[1] - bbox[0]) / 32
                #     volume += dim.prod(0) # dim[0] * dim[1] * dim[2]
                #     num_exprs += 1
                new_metrics['abv_all'] = volume / num_exprs
                new_metrics['all_volumes'] = volume
                new_metrics['all_n_subexprs'] = num_exprs
                new_metrics['SM_DIFF'] = num_exprs / volume
                # TBD
                new_metrics['abv_valid_subsets'] = volume / num_exprs
                new_metrics['valid_volumes'] = volume
                new_metrics['valid_n_subexprs'] = num_exprs
                new_metrics['SM'] = num_exprs / volume
                
            # Inter-Node Distance 
            # IF REQUIRED
            LENGTH_TAX = LENGTH_TAXES[top_ind]
            new_metrics['objective'] = new_metrics['iou'] + LENGTH_TAX * new_metrics['program_length']
            per_program_metrics.append(new_metrics)
    
    
        
        print("-------------SET %d metrics----------------" % top_ind)
        # Reconstruction metrics
        show_logs(per_program_metrics)
        
        ious = [x['iou'] for x in per_program_metrics]
        ious_64 = [x['iou_64'] for x in per_program_metrics]
        chamfer_dist = [x['chamfer_dist'] for x in per_program_metrics]
        program_lengths = [x['program_length'] for x in per_program_metrics]
        sm = [x['SM'] for x in per_program_metrics]
        sm_diff = [x['SM_DIFF'] for x in per_program_metrics]
        
        top_metric_set['iou_mean'] = np.mean(ious)
        top_metric_set['iou_64_mean'] = np.mean(ious_64)
        top_metric_set['chamfer_dist_mean'] = np.mean(chamfer_dist)
        top_metric_set['program_length_mean'] = np.mean(program_lengths)
        top_metric_set['SM_mean'] = np.mean(sm)
        top_metric_set['SM_DIFF_mean'] = np.mean(sm_diff)
        
        
        top_metric_set['iou_median'] = np.median(ious)
        top_metric_set['iou_64_median'] = np.median(ious_64)
        top_metric_set['chamfer_dist_median'] = np.median(chamfer_dist)
        top_metric_set['program_length_median'] = np.median(program_lengths)
        top_metric_set['SM_median'] = np.median(sm)
        top_metric_set['SM_DIFF_median'] = np.median(sm_diff)
        
        for threshold in [0.0, 0.5, 0.75, 0.9]:
            sm = [x['SM'] for x in per_program_metrics if x['iou'] > threshold]
            sm_diff = [x['SM_DIFF'] for x in per_program_metrics if x['iou'] > threshold]
            top_metric_set['SM_mean T@%.2f' % threshold] = np.mean(sm)
            top_metric_set['SM_DIFF_mean T@%.2f' % threshold] = np.mean(sm_diff)
            top_metric_set['SM_median T@%.2f' % threshold] = np.median(sm)
            top_metric_set['SM_DIFF_median T@%.2f' % threshold] = np.median(sm_diff)
            print("MEAN_SM T@%.2f: %.4f, SM_DIFF: %.4f" % (threshold, np.mean(sm), np.mean(sm_diff)))
            print("MEDIAN_SM T@%.2f: %.4f, SM_DIFF: %.4f" % (threshold, np.median(sm), np.median(sm_diff)))
        
        # Code Cohesion:
        # Gather valid programs
        # for threshold in [0.0, 0.5, 0.75, 0.9]:
        # for threshold in [0.0]:
        #     valid_programs = [x for x in per_program_metrics if x['iou'] > threshold]
        #     # Quantize the programs
        #     for quantization in [2, 4, 8]:
        #         action_space = action_class(quantization)
            
        #         all_quantized_subexprs = []
                
        #         for program in valid_programs:
        #             expression = program['expression']
        #             if "CSG" in LANGUAGE_NAME:
        #                 if "MCSG" not in LANGUAGE_NAME:
        #                     mcsg_expr = program_generator.parser.convert_to_mcsg(expression)
        #                 else:
        #                     mcsg_expr = expression
        #             else:
        #                 ...
        #             command_list = mcsg_parser.parse(mcsg_expr)
        #             graph = graph_compiler.command_tree(command_list, target=None, enable_subexpr_targets=False,
        #                                                     add_splicing_info=True)
                    
        #             for node_id in range(2, len(graph.nodes)):
        #                 cur_node = graph.nodes[node_id]
        #                 if cur_node['type'] in ["B", "M", "T"]:
        #                     cmd_list = graph_compiler.tree_to_command(graph, cur_node)
        #                     cur_expr = mcsg_parser.get_expression(cmd_list)
        #                     action_seq = action_space.expression_to_action(cur_expr)
        #                     action_str = "_".join([str(x) for x in action_seq])
        #                     all_quantized_subexprs.append(action_str)
                
        #         unique_exprs, n_counts = np.unique(all_quantized_subexprs, return_counts=True)
                
        #         top_metric_set["MEAN_CC_T@%.2f_Q@%d" % (threshold, quantization)] = np.mean(n_counts)
        #         top_metric_set["MEDIAN_CC_T@%.2f_Q@%d" % (threshold, quantization)] = np.median(n_counts)
        #         print("MEAN CC T@%.2f, Q@%d: %.4f" % (threshold, quantization, np.mean(n_counts)))
        #         print("MEDIAN CC T@%.2f, Q@%d: %.4f" % (threshold, quantization, np.median(n_counts)))
        #         print("N Unique Exprs", len(unique_exprs)/1000.0)
        
         
        
        cur_program_name = multiple_program_names[top_ind]
        save_file = os.path.join(os.path.dirname(cur_program_name), "cumulative_metrics.pkl")
        cPickle.dump(top_metric_set, open(save_file, "wb"))
        save_file = os.path.join(os.path.dirname(cur_program_name), "programs_with_cd.pkl")
        cPickle.dump(per_program_metrics, open(save_file, "wb"))
            