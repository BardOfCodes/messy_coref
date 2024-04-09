
from multiprocessing.sharedctypes import Value
from CSG.env.csg3d.mixed_len_generator import MixedGenerateData, DATA_PATHS, TOTAL_GENERATED_SAMPLES
from CSG.env.csg3d.shapenet_generator import ShapeNetGenerateData, DATA_PATHS as SA_DATA_PATHS
from .generator_helper import generate_programs_func
import h5py

# Training and testing dataset generator

from typing import List, Dict
from pathlib import Path
import numpy as np
from collections import defaultdict
import os
import torch
from .languages import boolean_commands, language_map
import torch.multiprocessing as mp
CAD_FILE = 'cad/cad.h5' 

class MixedGenerateData2D(MixedGenerateData):

    def __init__(self,
                 data_dir: str,
                 mode: str,
                 n_proc: int,
                 proc_id: int,
                 proportion: float,
                 csg_config,
                 program_lengths: Dict[int, float],
                 proportions: List[float],
                 sampling: str,
                 set_loader_limit=False,
                 loader_limit=1000,
                 project_root="",
                 action_space=None,
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
        self.n_generate_procs = csg_config.GENERATOR_N_PROCS
        self.valid_draws = csg_config.VALID_DRAWS
        self.valid_transforms = csg_config.VALID_TRANFORMS
        
        self.lang_type = csg_config.LANG_TYPE
        self.restrict_datasize = csg_config.RESTRICT_DATASIZE
        self.datasize = csg_config.DATASIZE
        
        self.set_parser_compiler(csg_config, project_root)
        
        
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

            
            # self.programs = self.generate_programs(program_lengths, proportions)
            # self.save_programs(data_paths)
            # Do this in parallel:
            allow_mirror = self.lang_type == "MNRCSG2D"
            allow_macro_mirror = self.lang_type == "MCSG2D"
                
            processes= []
            for proc_id in range(self.n_generate_procs):
                p = mp.Process(target=generate_programs_func, args=(proc_id, per_proc_plens[proc_id], 
                               self.parser, self.compiler, action_space, per_proc_save_files[proc_id],
                               TOTAL_GENERATED_SAMPLES, self.valid_draws, self.valid_transforms, self.lang_type, 
                               allow_mirror, allow_macro_mirror))
                p.start()
                processes.append(p) 
            
            for p in processes:
                p.join()
        
        self.data_paths = data_paths
        self.reload_data()
        self.add_stop_to_expr = True
        self.set_execution_mode(torch.device('cuda'), torch.float32)

    def set_parser_compiler(self, csg_config, project_root):
        parser_class = language_map[self.lang_type]['parser']
        compiler_class = language_map[self.lang_type]['compiler']
        self.parser = parser_class(module_path=project_root)
        self.compiler = compiler_class(csg_config.RESOLUTION, csg_config.SCALE, csg_config.SCAD_RESOLUTION,
                                        draw_type=csg_config.DRAW_TYPE, draw_mode=csg_config.DRAW_MODE)
                                        


class ShapeNetGenerateData2D(ShapeNetGenerateData):

    def __init__(self, *args, **kwargs):
        super(ShapeNetGenerateData2D, self).__init__(*args, **kwargs)

    
    def set_data_path(self):
        self.data_paths = {"CAD": os.path.join(self.data_dir, CAD_FILE)}

    def set_parser_compiler(self, csg_config, project_root):
        parser_class = language_map[self.lang_type]['parser']
        compiler_class = language_map[self.lang_type]['compiler']
        self.parser = parser_class(module_path=project_root)
        self.compiler = compiler_class(csg_config.RESOLUTION, csg_config.SCALE, csg_config.SCAD_RESOLUTION,
                                        draw_type=csg_config.DRAW_TYPE, draw_mode=csg_config.DRAW_MODE)

    def reload_data(self):

        for key, path in self.data_paths.items():
            if not os.path.exists(path):
                print("%s does not exists!" % path)
                raise ValueError("The vox file does not exists")
            else:
                print("%s exists" % path)
                hf = h5py.File(path, "r")
                if self.mode == "TRAIN":
                    data = np.array(hf.get(name="%s_images" % "train"))
                elif self.mode == "EVAL":
                    data = np.array(hf.get(name="%s_images" % "val"))
                elif self.mode == "TEST":
                    data = np.array(hf.get(name="%s_images" % "test"))
                # data = self.preprocess_data(data, self.resolution) 
                hf.close()
                if self.restrict_datasize: 
                    res_len = int(data.shape[0] * self.datasize)
                    data = data[:res_len]
                self.programs[key] = data
                

        self.current_index_list = {i:0 for i in self.programs.keys()}
        self.set_data_distr()
    
    def load_specific_data(self, target_slots, target_ids):
        # Remove data
        del self.programs
        self.programs = dict()
        
        target_slot_set = set(target_slots)
        slot_to_targets = defaultdict(list)
        for ind, cur_id in enumerate(target_ids):
            cur_slot = target_slots[ind]
            slot_to_targets[cur_slot].append(cur_id)
        
        for slot, targets in slot_to_targets.items():
            path = self.data_paths[slot]
            if not os.path.exists(path):
                print("%s does not exists!" % path)
                raise ValueError("The vox file does not exists")
            else:
                print("%s exists" % path)
                hf = h5py.File(path, "r")
                if self.mode == "TRAIN":
                    data = np.array(hf.get(name="%s_images" % "train"))
                elif self.mode == "EVAL":
                    data = np.array(hf.get(name="%s_images" % "val"))
                elif self.mode == "TEST":
                    data = np.array(hf.get(name="%s_images" % "test"))
                # data = self.preprocess_data(data, self.resolution) 
                hf.close()
                
                # data_list = []
                # # id_to_index = dict()
                # for ind, cur_id in enumerate(targets):
                #     data_list.append(data[cur_id])
                # data_list = np.stack(data_list, 0)
                self.programs[slot] = data
        self.current_index_list = {i:0 for i in self.programs.keys()}
        
        self.set_data_distr(mode="np_array")


    def preprocess_data(self, data, resolution):
        raise ValueError("Not for 2D CSG")