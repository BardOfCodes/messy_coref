import cv2
import numpy as np
from collections import defaultdict
import torch as th
from typing import Tuple

class Reward():

    def __init__(self, config):

        self.frequency = config.FREQUENCY
        self.power = config.POWER
        self.type = config.TYPE
        self.canvas_shape = config.CANVAS_SHAPE
        self.false_reward = config.FALSE_PROGRAM_REWARD
        self.active_reward_func = None
        self.cdist_neg = config.CDIST_NEG

        if self.type == "IOU":
            self.active_reward_func = self.iou_func
        elif self.type == "CHAMFER":
            self.active_reward_func = self.chamfer_func
        elif self.type == "THRESH_CHAMFER":
            self.active_reward_func = self.thres_chamfer_func
        elif self.type == "BOTH":
            self.active_reward_func = self.both_func
        elif self.type == "3DIOU":
            self.active_reward_func = self.iou_3d
        else:
            raise Exception("Reward Type not recognized!")
        
        self.use_exploration_reward = config.USE_EXPLORATION_REWARD
        self.exploration_beta = config.EXPLORATION_BETA
        # self.state_counter = defaultdict(lambda: 0)
        self.use_cr_reward = False
        self.mod_env = None
        if config.USE_CR_REWARD:
            self.use_cr_reward = True
            self.cr_reward_coef = config.CR_REWARD_COEF
            # somehow assign the 
        self.previous_reward = 0

    @staticmethod
    def iou_func(prediction, target):
        R = iou_func(prediction, target)
        return R

    def iou_3d(self, prediction, target, *args, **kwargs):
        # convert sdf to bool:
        R =  iou_func(prediction, target)
        return R
        
    def chamfer_func(self, prediction, target):

        distance = chamfer(target, prediction)
        # CHAMFER_CONSTANT = chamfer(target, prediction * 0)
        image_size = self.canvas_shape[-1]
        # normalize the distance by the diagonal of the image
        # HACK max seems to be 18, 
        R = (1.0 - (1 * distance) / image_size / (2**0.5))
        R = np.clip(R, a_min=0.0, a_max=1.0)
        R[R > 1.0] = 0
        return R
    
    def thres_chamfer_func(self, prediction, target):
        R = self.chamfer_func(prediction, target)
        thres = 0.985
        R[R>=thres] = 1
        R[R<thres] = 0
        return R
        

    def both_func(self, prediction, target):
        R1 = self.iou_func(prediction, target)
        R2 = self.chamfer_func(prediction, target)
        R = (R1 + R2)/ 2.0
        return R

    def exploration_reward(self, pred_expression):
        expr_str = "".join(pred_expression)
        self.state_counter[expr_str] += 1
        exploration_reward = self.exploration_beta / np.sqrt(self.state_counter[expr_str]) 
        return exploration_reward
    
    def cr_reward(self, pred_expression, target):
        expr_failed, new_expression = self.mod_env.refactor_expression(pred_expression, target)
        # if not expr_failed:
        reward = 0.0
        if not expr_failed:
            if len(new_expression) < len(pred_expression):
                reward = -1.0 * self.cr_reward_coef
        return reward

    def __call__(self, prediction, target, done, pred_expression=None):
        R = self.active_reward_func(prediction, target)
        R = R**self.power
        R = float(R)
        self.per_step_reward = R
        
        if self.use_exploration_reward:
            self.per_step_reward += self.exploration_reward(pred_expression)
        # Return frequency
        if self.frequency == 'EPISODIC':
            if done:
                if self.use_cr_reward:
                    assert self.mod_env, "Need to have a mod env"
                    self.per_step_reward += self.cr_reward(pred_expression, target)
                return self.per_step_reward
            else:
                return float(0.0)
        elif self.frequency == "PER_STEP_DELTA":
            cur_delta = self.per_step_reward - self.previous_reward
            self.previous_reward = self.per_step_reward
            # print(cur_delta)
            if done:
                self.previous_reward = 0
            return cur_delta
        else:
            return self.per_step_reward
    

    def false_program_reward(self):
        return self.false_reward ** self.power
    
    def reset(self, previous_reward=0):
        self.previous_reward = previous_reward

def iou_func(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    R = np.sum(np.logical_and(target, prediction)) / \
        (np.sum(np.logical_or(target, prediction)) + 1e-6)
    return R


def chamfer(images1: np.ndarray, images2: np.ndarray) -> np.ndarray:
    """Taken from:https://git.io/JfIpC"""
    # Convert in the opencv data format
    images1 = (images1 * 255).astype(np.uint8)
    images2 = (images2 * 255).astype(np.uint8)
    N = images1.shape[0]
    size = images1.shape[-1]

    D1 = np.zeros((N, size, size))
    E1 = np.zeros((N, size, size))

    D2 = np.zeros((N, size, size))
    E2 = np.zeros((N, size, size))
    summ1 = np.sum(images1, (1, 2))
    summ2 = np.sum(images2, (1, 2))

    # sum of completely filled image pixels
    filled_value = int(255 * size ** 2)
    defaulter_list = []
    for i in range(N):
        img1 = images1[i, :, :]
        img2 = images2[i, :, :]

        if (
            (summ1[i] == 0)
            or (summ2[i] == 0)
            or (summ1[i] == filled_value)
            or (summ2[i] == filled_value)
        ):
            # just to check whether any image is blank or completely filled
            defaulter_list.append(i)
            continue
        edges1 = cv2.Canny(img1, 1, 3)
        sum_edges = np.sum(edges1)
        if (sum_edges == 0) or (sum_edges == size ** 2):
            defaulter_list.append(i)
            continue
        dst1 = cv2.distanceTransform(
            ~edges1, distanceType=cv2.DIST_L2, maskSize=3
        )

        edges2 = cv2.Canny(img2, 1, 3)
        sum_edges = np.sum(edges2)
        if (sum_edges == 0) or (sum_edges == size ** 2):
            defaulter_list.append(i)
            continue

        dst2 = cv2.distanceTransform(
            ~edges2, distanceType=cv2.DIST_L2, maskSize=3
        )
        D1[i, :, :] = dst1
        D2[i, :, :] = dst2
        E1[i, :, :] = edges1
        E2[i, :, :] = edges2
    distances = np.sum(D1 * E2, (1, 2)) / (np.sum(E2, (1, 2)) + 1) + np.sum(
        D2 * E1, (1, 2)
    ) / (np.sum(E1, (1, 2)) + 1)
    # TODO make it simpler
    distances = distances / 2.0
    # This is a fixed penalty for wrong programs
    distances[defaulter_list] = 16
    return distances

def to_tensor(data: np.ndarray) -> th.Tensor:
    data = th.from_numpy(data).float()
    if th.cuda.is_available():
        return data.cuda()
    return data


def get_chamfer_distance_and_normal_consistency(
    gt_points: th.Tensor,
    pred_points: th.Tensor,
    gt_normals: th.Tensor,
    pred_normals: th.Tensor,
) -> Tuple[float, float]:
    gt_num_points = gt_points.shape[0]
    pred_num_points = pred_points.shape[0]

    points_gt_matrix = gt_points.unsqueeze(1).expand(
        [gt_points.shape[0], pred_num_points, gt_points.shape[-1]]
    )
    points_pred_matrix = pred_points.unsqueeze(0).expand(
        [gt_num_points, pred_points.shape[0], pred_points.shape[-1]]
    )

    distances = (points_gt_matrix - points_pred_matrix).pow(2).sum(dim=-1)
    match_pred_gt = distances.argmin(dim=0)
    match_gt_pred = distances.argmin(dim=1)

    dist_pred_gt = (pred_points - gt_points[match_pred_gt]).pow(2).sum(dim=-1).mean()
    dist_gt_pred = (gt_points - pred_points[match_gt_pred]).pow(2).sum(dim=-1).mean()

    normals_dot_pred_gt = (
        (pred_normals * gt_normals[match_pred_gt]).sum(dim=1).abs().mean()
    )

    normals_dot_gt_pred = (
        (gt_normals * pred_normals[match_gt_pred]).sum(dim=1).abs().mean()
    )
    chamfer_distance = dist_pred_gt + dist_gt_pred
    normal_consistency = (normals_dot_pred_gt + normals_dot_gt_pred) / 2

    return chamfer_distance.item(), normal_consistency.item()
