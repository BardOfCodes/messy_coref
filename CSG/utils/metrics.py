from ast import operator
from collections import defaultdict
import numpy as np
import torch
from CSG.env.reward_function import chamfer, iou_func
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
# from chamferdist import ChamferDistance

def return_zero(*args):
    return 0

BASE_METRICS = ['iou', 'chamfer', "true_chamfer", 'p_match', 'p_match_ratio', 'draw_entropy', 'log_prob', 'operator_entropy', 'reward', 'ep_length', 'first_step']
DIFF_OPT_METRICS = BASE_METRICS + ['diff_opt_reward']

def get_episode_stats(info, gt_program, stop_expression, mode="2D"):
    
    # Mainly for evaluation
    # Visualize actions:
    
    if gt_program:
        target_tokens = info['target_expression']
        predicted_tokens = info['predicted_expression']
        target_tokens = [x for x in target_tokens if not x == stop_expression]
        predicted_tokens = [x for x in predicted_tokens if not x == stop_expression]
        correctness = np.array([x==y for x, y in zip(target_tokens, predicted_tokens)])
        program_correctness = correctness.all()
        program_correctness_ratio = correctness.mean()
        first_step_correctness = correctness[0]
    else:
        program_correctness = 0.0
        program_correctness_ratio = 0.0
        first_step_correctness = 0.0
    
    if mode=="2DCSG":
        target_canvas = info['target_canvas']
        predicted_canvas = info['predicted_canvas']
        real_c_dist = chamfer(target_canvas[None, :, :], predicted_canvas[None, :, :])[0]
        c_dist = 100 - real_c_dist
        iou = iou_func(target_canvas, predicted_canvas)

    else:
        # reduce_factor = np.sqrt(3 * 64 **2)
        # chamferDist = ChamferDistance()
        # target_cloud = np.stack(np.where(target_canvas), -1).astype(np.float32)
        # target_cloud = torch.from_numpy(target_cloud).unsqueeze(0)/ reduce_factor
        # predicted_cloud = np.stack(np.where(predicted_canvas), -1).astype(np.float32)
        # # predicted_cloud = predicted_cloud[::50, :]
        # # target_cloud = target_cloud[::50, :]
        # predicted_cloud = torch.from_numpy(predicted_cloud).unsqueeze(0) / reduce_factor
        # dist_forward = chamferDist(target_cloud, predicted_cloud, bidirectional=True)
        # c_dist = dist_forward.detach().cpu().item()
        c_dist = 0
        real_c_dist = 0
        iou = info['reward']
        target_canvas = info['target_canvas']
        predicted_canvas = info['predicted_canvas']
        iou = iou_func(target_canvas, predicted_canvas)
    
    return iou, c_dist, real_c_dist, program_correctness, program_correctness_ratio, first_step_correctness


def get_entropy(item_dict):
    
    item_sum = np.sum(list(item_dict.values()))
    token_counts = np.array([x/(float(item_sum) + 1e-9) for x in item_dict.values()])
    entropy = np.sum(- token_counts  * np.log(token_counts))
    return entropy

class DefaultMetricExtractor:
    
    def __init__(self, gt_program, env, *args, **kwargs):
        
        self.gt_program = gt_program
        
        # Action Space
        self.predicted_tokens = defaultdict(return_zero)
        
        if not isinstance(env, VecEnv):
            action_space = env.action_space
        else:
            action_space = env.envs[0].action_space
        self.stop_expression = action_space.stop_expression

        self.is_draw = action_space.is_draw
        self.is_operation = action_space.is_operation

        self.metric_extraction_func = get_episode_stats
        self.mode="CSG2D"
    
    def __call__(self, info, current_reward, current_length, *args, **kwargs):
        
        iou, c_dist, real_c_dist, program_correctness, program_correctness_ratio, first_step_correction = self.metric_extraction_func(info, 
                                                                                        self.gt_program, 
                                                                                        self.stop_expression,
                                                                                        mode=self.mode)
        
        pred_tokens = info['predicted_expression']
        if "CSG" in self.mode:
            self.csg3d_update(pred_tokens)
        elif "SA" in self.mode:
            self.hsa3d_update(pred_tokens)
        for token in pred_tokens:
            self.predicted_tokens[token] += 1
        output_dict = {
            'iou': iou,
            'chamfer': c_dist,
            "true_chamfer": real_c_dist, 
            'p_match': program_correctness,
            'p_match_ratio': program_correctness_ratio,
            'reward': current_reward,
            'ep_length': current_length,
            'log_prob': info['log_prob'],
            'first_step': first_step_correction
            
        }
        return output_dict

    def csg3d_update(self, pred_tokens):
        raise ValueError("Not defined for 2D CSG")
    def hsa3d_update(self, pred_tokens):
        raise ValueError("Not defined for 2D CSG")

    def fuse(self, metric_extractor):
        
        for key, value in metric_extractor.predicted_tokens.items():
            self.predicted_tokens[key] += value

    def extract_final_metrics(self):
        
        op_list, draw_list = [], []
        for key, value in self.predicted_tokens.items():
            if self.is_operation(key):
                op_list.append(value)
            elif self.is_draw(key):
                draw_list.append(value)
        op_list = np.array(op_list)
        draw_list = np.array(draw_list)
        
        operator_entropy = get_entropy(op_list)
        draw_entropy = get_entropy(draw_list)
        
        output_dict = dict(draw_entropy=draw_entropy, operator_entropy=operator_entropy)
        
        action_count_dict = dict(prediction_histogram={})
        
        for key, value in self.predicted_tokens.items():
            action_count_dict['prediction_histogram'][key] = value
        return output_dict, action_count_dict

class CSG3DMetricExtractor(DefaultMetricExtractor):
    def __init__(self, gt_program, env, *args, **kwargs):
        super(CSG3DMetricExtractor, self).__init__(gt_program, env, *args, **kwargs)
        self.mode="3DCSG"
        if not isinstance(env, VecEnv):
            action_space = env.action_space
        else:
            action_space = env.envs[0].action_space
        self.is_draw = action_space.is_draw
        self.is_bool = action_space.is_bool
        self.is_transform = action_space.is_transform
        self.is_stop = action_space.is_stop
        self.is_macro = action_space.is_macro
        self.is_fixed_macro = action_space.is_fixed_macro
    
    def csg3d_update(self, pred_tokens):
        pred_tokens = [x.split("(")[0] for x in pred_tokens]
        # Also record higher order transforms: transforms above boolean
        higher_transform_count = 0
        leaf_transforms = 0 
        for cur_token in pred_tokens:
            if cur_token in ["union", "intersection", "difference"]:
                leaf_transforms = 0
            elif cur_token in ["scale", "rotate", "translate"]:
                higher_transform_count += 1
                leaf_transforms += 1
            elif cur_token in ["cuboid", "ellipsoid", "cylinder", "sphere"]:
                higher_transform_count -= leaf_transforms
                leaf_transforms = 0
        self.predicted_tokens['higher_transforms'] += higher_transform_count 
        if higher_transform_count > 0:
            self.predicted_tokens["HAS_HIGHER"] += 1    
        if "macro" in pred_tokens:
            self.predicted_tokens["HAS_FIXED_MACRO"] += 1    

    def extract_final_metrics(self):
        total_tokens = 1e-9
        draws, bools, transforms, macros, fixed_macros, stops = [defaultdict(return_zero) for i in range(6)]
        for key, value in self.predicted_tokens.items():
            if self.is_draw(key):
                draws[key] += value
            elif self.is_bool(key):
                bools[key] += value
            elif self.is_transform(key):
                transforms[key] += value
            elif self.is_macro(key):
                macros[key] += value
            elif self.is_fixed_macro(key):
                fixed_macros[key] += value
            elif self.is_stop(key):
                stops[key] += value
            total_tokens += value
                
        draw_entropy = get_entropy(draws)
        bool_entropy = get_entropy(bools)
        transform_entropy = get_entropy(transforms)
        fixed_macro_entropy = get_entropy(fixed_macros)
        
        draw_usage = np.sum(list(draws.values())) / total_tokens
        bool_usage = np.sum(list(bools.values())) / total_tokens
        transform_usage = np.sum(list(transforms.values())) / total_tokens
        fixed_macro_usage = np.sum(list(fixed_macros.values())) / total_tokens


        output_dict = dict(D_entropy=draw_entropy, 
                           B_entropy=bool_entropy,
                           T_entropy=transform_entropy,
                           FM_entropy=fixed_macro_entropy,
                           D_usage=draw_usage, 
                           B_usage=bool_usage,
                           T_usage=transform_usage,
                           FM_usage=fixed_macro_usage,
                           HT_count=self.predicted_tokens['higher_transforms'],
                           HT_ratio=self.predicted_tokens['higher_transforms']/total_tokens,
                           p_HT=self.predicted_tokens['HAS_HIGHER'],
                           p_FM=self.predicted_tokens['HAS_FIXED_MACRO'],
                           )

        action_count_dict = dict(prediction_histogram={})
        
        for key, value in self.predicted_tokens.items():
            action_count_dict['prediction_histogram'][key] = value
        return output_dict, action_count_dict


class CSG2DMetricExtractor(CSG3DMetricExtractor):
    def __init__(self, gt_program, env, *args, **kwargs):
        super(CSG2DMetricExtractor, self).__init__(gt_program, env, *args, **kwargs)
        self.mode="2DCSG"

class DiffOptMetricExtractor(DefaultMetricExtractor):
    
    def __call__(self, info, current_reward, current_length, *args, **kwargs):
        output_dict = super(DiffOptMetricExtractor, self).__call__(info, current_reward, current_length, *args, **kwargs)
        output_dict['diff_opt_reward'] = kwargs['diff_opt_reward']
        return output_dict
        
class HSA3DMetricExtractor(DefaultMetricExtractor):
    
    def __init__(self, gt_program, env, *args, **kwargs):
        
        self.gt_program = gt_program
        
        # Action Space
        self.predicted_tokens = defaultdict(return_zero)
        
        if not isinstance(env, VecEnv):
            action_space = env.action_space
        else:
            action_space = env.envs[0].action_space
        self.stop_expression = action_space.stop_expression

        self.metric_extraction_func = get_episode_stats
        self.mode="HSA3D"

        self.is_cuboid = action_space.is_cuboid
        self.is_attach = action_space.is_attach
        self.is_squeeze = action_space.is_squeeze
        self.is_translate = action_space.is_translate
        self.is_reflect = action_space.is_reflect
        self.is_stop = action_space.is_stop
        self.n_cubes = []
        self.n_hierarchy = []
    
    def hsa3d_update(self, pred_tokens):
        # Also record higher order transforms: transforms above boolean
        n_cubes = 0
        n_hierarchy = -1
        for cur_token in pred_tokens:
            if self.is_cuboid(cur_token):
                n_cubes += 1
            elif self.is_stop(cur_token):
                n_hierarchy += 1
        self.n_cubes.append(n_cubes) 
        self.n_hierarchy.append(n_hierarchy)

    def extract_final_metrics(self):
        total_tokens = 1e-9
        cuboids, attaches, squeezes, translates, reflects, stops = [defaultdict(return_zero) for i in range(6)]
        for key, value in self.predicted_tokens.items():
            if self.is_cuboid(key):
                cuboids[key] += value
            elif self.is_attach(key):
                attaches[key] += value
            elif self.is_squeeze(key):
                squeezes[key] += value
            elif self.is_translate(key):
                translates[key] += value
            elif self.is_reflect(key):
                reflects[key] += value
            elif self.is_stop(key):
                stops[key] += value
            total_tokens += value
                
        cuboids_entropy = get_entropy(cuboids)
        attaches_entropy = get_entropy(attaches)
        squeezes_entropy = get_entropy(squeezes)
        translates_entropy = get_entropy(translates)
        reflects_entropy = get_entropy(reflects)
        
        cuboids_usage = np.sum(list(cuboids.values())) / total_tokens
        attaches_usage = np.sum(list(attaches.values())) / total_tokens
        squeezes_usage = np.sum(list(squeezes.values())) / total_tokens
        translates_usage = np.sum(list(translates.values())) / total_tokens
        reflects_usage = np.sum(list(reflects.values())) / total_tokens

        output_dict = dict(cuboids_entropy=cuboids_entropy, 
                           attaches_entropy=attaches_entropy,
                           squeezes_entropy=squeezes_entropy,
                           translates_entropy=translates_entropy,
                           reflects_entropy=reflects_entropy, 
                           cuboids_usage=cuboids_usage,
                           attaches_usage=attaches_usage,
                           squeezes_usage=squeezes_usage,
                           translates_usage=translates_usage,
                           reflects_usage=reflects_usage,
                           avg_cubes=np.nanmean(self.n_cubes),
                           avg_hier=np.nanmean(self.n_hierarchy),
                           )

        action_count_dict = dict(prediction_histogram={})
        
        # for key, value in self.predicted_tokens.items():
        #     action_count_dict['prediction_histogram'][key] = value
        return output_dict, action_count_dict
        
        
EXTRACTOR = dict(DefaultMetricExtractor=DefaultMetricExtractor,
                 DiffOptMetricExtractor=DiffOptMetricExtractor,
                 CSG2DMetricExtractor=CSG2DMetricExtractor,
                 CSG3DMetricExtractor=CSG3DMetricExtractor,
                 HSA3DMetricExtractor=HSA3DMetricExtractor)


class MetricObj:
    def __init__(self, metrics=None, metric_extractor=None, env=None, gt_program=None, save_predictions=False):
        if metrics is None:
            raise ValueError("Need to pass a list of metrics")
        if metric_extractor is None:
            raise ValueError("Need to pass a metric_extractor")
        
        if isinstance(metric_extractor, str):
            extractor_cls = EXTRACTOR[metric_extractor]
            self.metric_extractor = extractor_cls(gt_program, env)
        else:
            self.metric_extractor = metric_extractor(gt_program, env)
            
        
        self.metric_dict = {x:[] for x in metrics}
        self.save_predictions = save_predictions
        self.predictions = []
        
        self.predicted_tokens = defaultdict(return_zero)
    
    def update_metrics(self, info, *args, **kwargs):
        metrics = self.metric_extractor(info, *args, **kwargs)
        if self.save_predictions:
            self.predictions.append([info['predicted_expression'], info['slot_id'], info['target_id']])
            if "current_reward" in kwargs.keys():
                self.predictions[-1].append(kwargs['current_reward'])
            if "log_prob" in kwargs.keys():
                self.predictions[-1].append(kwargs['log_prob'])
            else:
                self.predictions[-1].append(0)

        for key, value in metrics.items():
            self.metric_dict[key].append(value)
    
    def return_metrics(self):
        mean_metrics = dict()
        hist_key = "prediction_histogram"
        if hist_key in self.metric_dict.keys():
            _hist = self.metric_dict.pop(hist_key)
        for key, value in self.metric_dict.items():
            mean_metrics[key] = np.nanmean(value)
            # mean_metrics["%s median" % key] = np.nanmedian(value)
        
        final_mean_metrics, bulk_metrics = self.metric_extractor.extract_final_metrics()
        
        for key, value in final_mean_metrics.items():
            mean_metrics[key] = value
            
        for key, value in bulk_metrics.items():
            self.metric_dict[key] = value
        
        return mean_metrics, self.metric_dict, self.predictions
    def fuse(self, metric_obj):
        ## ADD assert checkers: 
        for key, value in metric_obj.metric_dict.items():
            self.metric_dict[key].extend(value)
        ## ADD assert checkers: 
        for value in metric_obj.predictions:
            self.predictions.append(value)
        
        for key, value in metric_obj.predicted_tokens.items():
            self.predicted_tokens[key] += value
            
        self.metric_extractor.fuse(metric_obj.metric_extractor)
        
    def update_refactor(self, pred_expression, refactored_expression):
        
        pred_len = len(pred_expression)
        
        
        if refactored_expression:
            
            refactored_len = len(refactored_expression)
            if refactored_len == pred_len:
                self.refactor.append(0)
                self.refactor_length.append(pred_len)
            else:
                self.refactor.append(1)
                self.refactor_length.append(refactored_len)
                self.refactor_ratio.append(refactored_len/pred_len)
        else:
            self.refactor.append(0)
            self.refactor_length.append(pred_len)
            # self.refactor_ratio.append()
            
        