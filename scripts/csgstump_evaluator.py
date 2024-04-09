
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
from CSG.bc_trainers.rewrite_engines.code_splice_utils import get_scores, get_2d_scores



# objs_file = "/home/aditya/projects/rl/weights/iccv/csgstump_converted/csgstump_train_converted.pkl"
objs_file = "/home/aditya/projects/rl/weights/iccv/csgstump_converted/csgstump_converted.pkl"

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

    # Some basic settings: 
    # get it to get the train mcsg:
    bc_config = config.BC.clone()
    bc_config.BS.BEAM_SIZE = 10
    bc_config.DO.EXHAUSTIVE = True
    bc_config.GS.EXHAUSTIVE = True
    bc_config.CS.EXHAUSTIVE = True
    # bc_config.DO.ENABLE = False
    # bc_config.GS.ENABLE = False
    # bc_config.CS.ENABLE = False
    bc_config.BS.ENABLE = False
    bc_config.WS.ENABLE = False
    # bc_config.DO.SAMPLE_COUNT = 100
    # bc_config.DO.N_PROC = 1
    # bc_config.CS.SAMPLE_COUNT = 100
    # bc_config.ENV.MODE = "TRAIN"
    # config.EVAL.ENV.MODE = "TRAIN"
    # config.TRAIN.ENV.MODE = "TRAIN"
    bc_config.ENV.MODE = "TEST"
    config.EVAL.ENV.MODE = "TEST"
    config.TRAIN.ENV.MODE = "TEST"

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

    epoch = 1

    save_path = config.MODEL.LOAD_WEIGHTS
    train_state = bc_trainer.model_info['train_state']()
    train_state.cur_epoch = epoch
    
    # Get the train programs:
    prog_objs = cPickle.load(open(objs_file, "rb"))
    bc_env_class = getattr(csg_env, bc_config.ENV.TYPE)
    temp_env = bc_env_class(
        config=config, phase_config=bc_config, seed=0, n_proc=1, proc_id=0)
    temp_env.mode = "EVAL"
    # update the reward and othe details: 
    compiler = temp_env.program_generator.compiler
    parser = temp_env.program_generator.parser
    resolution = temp_env.action_space.resolution
    
    device = th.device("cuda")
    dtype = th.float32
    compiler.set_to_cuda()
    compiler.set_to_full()
    compiler.reset()

    parser.set_device(device)
    parser.set_tensor_type(dtype)
    for ind, obj in enumerate(prog_objs):
        # if ind % 100 == 0:
        #     print("Processing %d/%d" % (ind, len(prog_objs)))
        expression = obj['expression']
        # # expression.append("$")
        slot_id = obj['slot_id']
        target_id = obj['target_id']
        # obj['log_prob'] = 0
        # obj['origin'] = "BS"
        # obj['do_fail'] = False
        # obj['cs_fail'] = False
        obs = temp_env.reset_to_target(slot_id, target_id)
        target_np = obs['obs']
        target = th.from_numpy(target_np).cuda()
        target_bool = target.bool()
        target = target.half()
        command_list = parser.parse(expression)
        compiler._compile(command_list)
        output_sdf = compiler._output
        pred_canvas = (output_sdf.detach() <= 0)# .cpu().data.numpy()
        reward = get_scores(pred_canvas, target_bool)
        # obj['reward'] = reward + config.BC.PLAD.LENGTH_ALPHA * len(obj['expression'])
        obj['reward'] = reward + config.BC.PLAD.LENGTH_ALPHA * len(obj['expression'])
    # # save with this update:
    cPickle.dump(prog_objs, open(objs_file, "wb"))
    bc_trainer.bpds.max_length = 1024
    bc_trainer.bpds.set_best_programs(prog_objs, temp_env)
    
    # csr = bc_trainer.code_splice_rewriter
    # best_program_dict = bc_trainer.bpds.bpd
    # subexpr_cache = FaissIVFCache(csr.save_dir, csr.cache_config, eval_mode=False, language_name=csr.language_name)
    # subexpr_cache.generate_cache_and_index(
    #     best_program_dict, temp_env, csr.cs_use_canonical)
    # asdf
    bc_trainer.best_program_init = False
    bc_trainer.code_splice_rewriter.cs_max_bool_count = 1000
    bc_trainer.code_splice_rewriter.max_length = 1024
    bc_trainer.update_best_programs(save_path, train_state, quantize=False, log_interval=100)
    # bc_trainer.best_program_dict, program_list = cPickle.load(open(save_file, "rb"))
   
    program_list = bc_trainer.construct_training_data(train_state)
    for ind, program in enumerate(program_list):
        expression = program['expression']
        program["render_expr"] = expression
    
    dir_name = os.path.dirname(save_path)
    file_name = os.path.join(dir_name, "beam_do_gs_cs.pkl")
    # file_name = os.path.join(dir_name, "beam_cs.pkl")
    cPickle.dump(program_list, open(file_name, "wb"))
    
    if iterative_do_cs:
        train_state.cur_epoch = 1
        # Get the expressions train
        # Now the train dict is saved:
        bc_trainer.beam_search_generator.enable = False
        bc_trainer.code_splice_rewriter.enable = True
        bc_trainer.code_splice_rewriter.eval_mode = False # so that it loads the recently created exprs
        for iteration in range(max_iteration):
            if iteration == max_iteration -1:
                bc_trainer.code_splice_rewriter.enable = False
            bc_trainer.update_best_programs(save_path, train_state, quantize=False, log_interval=100)
            print("Average Score of best training programs is %f" % bc_trainer.bpds.mean_reward)
            program_list = bc_trainer.construct_training_data(train_state)
            rewards = [x['reward'] for x in program_list]
            print("ITER %d: Final Eval Reward" % (iteration+1), np.mean(rewards))
    
    program_list = bc_trainer.construct_training_data(train_state)
    
    for ind, program in enumerate(program_list):
        expression = program['expression']
        program["render_expr"] = expression
    
    rewards = [x['reward'] for x in program_list]
    print("Final Reward", np.mean(rewards))
    # Save the program file:
    dir_name = os.path.dirname(save_path)
    file_name = os.path.join(dir_name, "beam_do_gs_cs_3.pkl")
    cPickle.dump(program_list, open(file_name, "wb"))