

from typing import List, Dict
import os
from pathlib import Path
import torch.multiprocessing as mp
import torch as th
import numpy as np

from CSG.env.csg3d.mixed_len_generator import MixedGenerateData, DATA_PATHS, TOTAL_GENERATED_SAMPLES
from CSG.env.csg3d.shapenet_generator import ShapeNetGenerateData, DATA_PATHS as SA_DATA_PATHS
from .parser import SAParser
from .compiler import SACompiler
from .random_generator import generate_programs_func


class SAMixedGenerateData(MixedGenerateData):


    def __init__(self,
                 data_dir: str,
                 mode: str,
                 n_proc: int,
                 proc_id: int,
                 proportion: float,
                 sa_config,
                 program_lengths: Dict[int, float],
                 proportions: List[float],
                 sampling: str,
                 set_loader_limit=False,
                 loader_limit=1000,
                 project_root="",
                 action_space=None
                 ):
        
        self.program_lengths = program_lengths
        self.proportions = proportions
        self.program_types = len(program_lengths)
        self.mode = mode
        self.proportion = proportion
        self.sampling = sampling
        self.n_proc = n_proc
        self.proc_id = proc_id
        self.set_loader_limit = set_loader_limit
        self.loader_limit = loader_limit
        self.n_generate_procs = sa_config.GENERATOR_N_PROCS
        self.language_mode = sa_config.LANGUAGE_NAME
        if "PSA" in self.language_mode:
            self.gen_subprogram = False
        else:
            self.gen_subprogram = True

        self.set_parser_compiler(sa_config, project_root)
        
        
        data_paths = {}
        for key, value in DATA_PATHS.items():
            if key in self.program_lengths:
                data_paths[key] = os.path.join(data_dir, value)

        self.programs = {}
        generate_programs = False
        for key, path in data_paths.items():
            if not os.path.exists(path):
                print("%s does not exists!" % path)
                generate_programs = True
                break
            else:
                print("%s exists" % path)
        
        if generate_programs:
            # Create all the paths:
            per_proc_plens = [list() for x in range(self.n_generate_procs)]
            per_proc_save_files = [dict() for x in range(self.n_generate_procs)]
            for ind, (key, path) in enumerate(data_paths.items()): 
                folder_path = os.path.dirname(path)
                Path(folder_path).mkdir(parents=True, exist_ok=True)
                proc_id = ind % self.n_generate_procs
                per_proc_plens[proc_id].append(key)
                per_proc_save_files[proc_id][key] = path

            processes= []
            for proc_id in range(self.n_generate_procs):
                p = mp.Process(target=generate_programs_func, args=(proc_id, per_proc_plens[proc_id], 
                               self.parser, self.compiler, action_space, per_proc_save_files[proc_id],
                               TOTAL_GENERATED_SAMPLES, self.gen_subprogram))
                p.start()
                processes.append(p) 
            
            for p in processes:
                p.join()
        
        self.data_paths = data_paths
        self.reload_data()
        self.hsa_to_hcsg_exprs = dict()

    def set_parser_compiler(self, sa_config, project_root):
        self.parser = SAParser(module_path=project_root)
        self.compiler = SACompiler(sa_config.RESOLUTION, sa_config.SCALE, sa_config.SCAD_RESOLUTION, varied_bbox_mode=sa_config.VARIED_SUBPROG_BBX)
    
    def boolean_count(self, expression):
        raise ValueError("Not defined for SA")
    
    def get_expression(self, slot, index):
        raise ValueError("NOT valid in SA")

    def get_executed_program(self, slot, index, return_numpy=True, return_bool=False):
        
        expression = self.programs[slot][index]#.strip()
        # expression = expression.split("__")
        with th.no_grad():
            # with th.cuda.amp.autocast():
            program = self.parser.parse(expression)
            self.compiler._compile(program)
            output = self.compiler.get_output_shape(return_bool)
        if return_numpy:
            output = output.cpu().numpy()
        
        return output, expression

class SAShapeNetGenerateData(ShapeNetGenerateData):

    def __init__(self,
                 data_dir: str,
                 mode: str,
                 n_proc: int,
                 proc_id: int,
                 proportion: float,
                 sa_config,
                 program_lengths: Dict[int, float],
                 proportions: List[float],
                 sampling: str,
                 set_loader_limit=False,
                 loader_limit=1000,
                 project_root=""
                 ):
        
        self.program_lengths = program_lengths
        self.proportions = proportions
        self.program_types = len(program_lengths)
        self.mode = mode
        self.proportion = proportion
        self.sampling = sampling
        self.n_proc = n_proc
        self.proc_id = proc_id

        self.lang_type = "ShapeAssembly"
        self.resolution = sa_config.RESOLUTION
        self.datamode = sa_config.DATAMODE
        self.restrict_datasize = sa_config.RESTRICT_DATASIZE
        self.datasize = sa_config.DATASIZE

        self.set_parser_compiler(sa_config, project_root)
        
        self.data_paths = {}
        for key, value in SA_DATA_PATHS.items():
            if key in self.program_lengths:
                if mode == "TRAIN":
                    self.data_paths[key] = os.path.join(data_dir, value % self.mode.lower())
                elif mode == "EVAL":
                    self.data_paths[key] = os.path.join(data_dir, value % "val")
                elif mode == "TEST":
                    self.data_paths[key] = os.path.join(data_dir, value % "test")

        self.programs = {}
        generate_programs = False
        self.reload_data()

        self.cumalative_prop = [sum(self.proportions[:x+1])
                                for x in range(len(self.proportions))]
        # linear sampling: y = m x + b; 
        self.slopes = [-0.5 * (i+1)/float(self.program_types)
                       for i in range(self.program_types)]
        # y = train_progress; x = sample cut. 
        # At y = 0; x = 1 for i = 1; 0 for rest
        # At y = 1; x = 1 for i = n; o for rest
        # at y = 0.5; x = i/n 
        # non-linear sampling; y = (1 -x) ^ n
        xes = [(i+1)/float(self.program_types) for i in range(self.program_types-1)]
        self.etas = [np.log(0.5)/np.log(1 - x) for x in xes]
        self.etas.append(np.inf)
        self.slot_counter = 0

    def set_parser_compiler(self, sa_config, project_root):
        self.parser = SAParser(module_path=project_root)
        self.compiler = SACompiler(sa_config.RESOLUTION, sa_config.SCALE, sa_config.SCAD_RESOLUTION)
