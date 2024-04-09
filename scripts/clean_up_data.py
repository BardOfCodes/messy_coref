
import os 
import numpy as np
import torch as th
import torch.nn as nn
import torch.multiprocessing as tmp
# import meshplot as mp
import torch
import h5py
import random

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout

import sys

import seaborn as sns
import seaborn as sb
from pathlib import Path
import json

sns.set_theme()
# Load the dataset

from yacs.config import CfgNode as CN
from CSG.utils.train_utils import arg_parser, load_config
import CSG.env as csg_env
import _pickle as cPickle
import random
import time
from itertools import product
import torch.nn.functional as F

import argparse

import logging

from CSG.utils.train_utils import load_config, arg_parser
from CSG.env.csg3d.mixed_len_generator import DATA_PATHS

SIZE_LIMITS = {
    "PCSG3D": 11 + 12 * 7 + 1,
    "FCSG3D": 11 + 12 * 10 + 1,
    "HCSG3D": 192,
    "MCSG3D": 192
}
def main():
    config_file = 'configs/ablations/emerald/pretrain_baseline.py'
    args = arg_parser.parse_args(['--config-file', config_file])

    config = load_config(args)
    lang_type = config.TRAIN.ENV.CSG_CONF.LANG_TYPE
    size_limit = SIZE_LIMITS[lang_type]
    # config.MODEL.LOAD_WEIGHTS = os.path.join('..', config.MODEL.LOAD_WEIGHTS) 
    if lang_type == "FCSG3D":
        EXPRESSION_CORRECTION = True
    else:
        EXPRESSION_CORRECTION = False
    seed = 0
    config.POLICY.PPO.N_ENVS = 1
    bc_config = config.BC
    # bc_config.ENV.MODE = "EVAL" # For the eval set:
    bc_env_class = getattr(csg_env, config.TRAIN.ENV.TYPE)
    temp_env = bc_env_class(config=config, phase_config=bc_config, seed=seed, n_proc=1, proc_id=0)
    prog_gen = temp_env.program_generator
    action_space = temp_env.action_space

    data_paths = {}
    data_dir = os.path.join(config.MACHINE_SPEC.DATA_PATH)    
    for key, value in DATA_PATHS.items():
        if key in prog_gen.program_lengths:
            data_paths[key] = os.path.join(data_dir, value)
    # Check length
    for key, program_list in prog_gen.programs.items():
        new_list = []
        reject_count = 0
        for program in program_list:
            expression = program.strip()
            expression = expression.split("__")
            target_actions = temp_env.action_space.expression_to_action(expression)
            action_len = len(target_actions)
            if action_len < size_limit:
                if EXPRESSION_CORRECTION:
                    for ind, expr in enumerate(expression):
                        if "(" in expr:
                            command_symbol = expr.split("(")[0]
                            param_str = expr.split("(")[1][:-1]
                            param = np.array([float(x.strip()) for x in param_str.split(",")])
                            new_param = np.zeros(param.shape)
                            new_param[:3] = param[:3]
                            new_param[3:6] = param[6:9]
                            new_param[6:9] = param[3:6]
                            param_str = ", ".join(["%f" % x for x in new_param])
                            expression[ind] = "%s(%s)" %(command_symbol, param_str)
                
                new_list.append(expression)
            else:
                reject_count += 1

        path = data_paths[key]
        folder_path = os.path.dirname(path)
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        print('Saving %d programs at %s' % (len(new_list), path))
        print("%d programs rejected." % reject_count)
        with open(path, "w") as f:
            for cur_program in new_list:
                strng = "__".join(cur_program) + "\n"
                f.write(strng)

if __name__ == "__main__":
    main()
    # parse_file()