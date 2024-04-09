# Training and testing dataset generator

from typing import List, Dict
from pathlib import Path
import numpy as np
from collections import defaultdict
import os
import torch
from .languages import boolean_commands, language_map
import torch.multiprocessing as mp
from .generator_helper import generate_programs_func

# TODO: More lengths possible.
DATA_PATHS = {
    1: "synthetic/one_ops/expressions.txt",
    2: "synthetic/two_ops/expressions.txt",
    3: "synthetic/three_ops/expressions.txt",
    4: "synthetic/four_ops/expressions.txt",
    5: "synthetic/five_ops/expressions.txt",
    6: "synthetic/six_ops/expressions.txt",
    7: "synthetic/seven_ops/expressions.txt",
    8: "synthetic/eight_ops/expressions.txt",
    9: "synthetic/nine_ops/expressions.txt",
    10: "synthetic/ten_ops/expressions.txt",
    11: "synthetic/eleven_ops/expressions.txt",
    12: "synthetic/twelve_ops/expressions.txt",
    13: "synthetic/thirteen_ops/expressions.txt",
    14: "synthetic/fourteen_ops/expressions.txt",
    15: "synthetic/fifteen_ops/expressions.txt",
}
# This is per generator
TOTAL_GENERATED_SAMPLES = int(1e5)

class MixedGenerateData:

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
        assert not self.restrict_datasize, "Not configured for synthetic programs"
        
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
            allow_mirror = self.lang_type == "MNRCSG3D"
            allow_macro_mirror = self.lang_type == "MCSG3D"
                
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
    
    def set_execution_mode(self, device, dtype):
        self.parser.set_device(device)
        self.parser.set_tensor_type(dtype)

        if device == torch.device("cuda"):
            self.compiler.set_to_cuda()
        else:
            self.compiler.set_to_cpu()

        if dtype == torch.float32:
            self.compiler.set_to_full()
        else:
            self.compiler.set_to_half()

        self.compiler.reset()

    def reload_data(self):

        for index in self.data_paths.keys():
            with open(self.data_paths[index]) as data_file:
                programs = data_file.readlines()
                programs = [x.strip().split("__") for x in programs]
                self.programs[index] = programs
        
        self.current_index_list = {i:0 for i in self.programs.keys()}
        self.set_data_distr()

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


    def set_parser_compiler(self, csg_config, project_root):
        parser_class = language_map[self.lang_type]['parser']
        compiler_class = language_map[self.lang_type]['compiler']
        self.parser = parser_class(module_path=project_root)
        self.compiler = compiler_class(csg_config.RESOLUTION, csg_config.SCALE, csg_config.SCAD_RESOLUTION,
                                        draw_type=csg_config.DRAW_TYPE, draw_mode=csg_config.DRAW_MODE)
    def save_programs(self, data_paths):
        print("Saving programs!")
        for key, path in data_paths.items():
            gen_programs = self.programs[key]
            print('Saving %s' % path)
            with open(path, "w") as f:
                for cur_program in gen_programs:
                    strng = "__".join(cur_program) + "\n"
                    f.write(strng)
    
        
    def execute(self, expr, return_numpy=False, return_shape=True, add_noise=False, return_bool=False):
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            if add_noise:
                program = self.parser.noisy_parse(expr)
            else:
                program = self.parser.parse(expr)

            self.compiler._compile(program)
            if return_shape:
                primitive = self.compiler.get_output_shape(return_bool)
            else:
                primitive = self.compiler.get_output_sdf()
            if return_numpy:
                primitive = primitive.cpu().numpy().copy()
        # cleanup:

        return primitive
    
    def reset_data_distr(self):
        self.set_data_distr()

    def set_data_distr(self, mode="np_array"):
        # find how many for each
        dataset_sizes = [len(y) for x, y in self.programs.items()]

        if self.mode == "TRAIN":
            # Use the first 70%.
            lims = [(0, int(x * self.proportion)) for x in dataset_sizes]
            # lims = [(0, 600) for x in dataset_sizes]
            randomize = True
            # lims = [(0,8)]
        elif self.mode == "EVAL":
            # Use the last 30%
            # Split it based on number of procs
            lims = [(int(x * self.proportion), x) for x in dataset_sizes]
            divs = [(x[1]-x[0])// self.n_proc for x in lims]
            lims = [(x[0] + self.proc_id * divs[i], x[0] + (self.proc_id + 1) * divs[i]) for i, x in enumerate(lims)]
            randomize = False
        elif self.mode == "TEST":
            randomize = False
        self.active_gen_objs = {}
        if self.set_loader_limit:
            lims[1] = min(lims[0] + self.loader_limit, lims[1])
        print("Loader Limits", lims)
        for ind, p_len in enumerate(self.program_lengths):
            self.active_gen_objs[p_len] = self.get_data(
                p_len, data_lims=lims[ind], if_randomize=randomize)

    def get_executed_program(self, slot, index, return_numpy=True, return_bool=False):
        
        expression = self.programs[slot][index]#.strip()
        # expression = expression.split("__")
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
                try:
                    program = self.parser.parse(expression)
                except Exception as ex:
                    print(ex)
                    print(slot, index)
                self.compiler._compile(program)
                
                output = self.compiler.get_output_shape(return_bool)
        if return_numpy:
            output = output.cpu().numpy()
        # program_len = len(program)
        # Add stop sign
        expression.append("$")
        
        return output, expression
    
    def boolean_count(self, expression):
        count = 0
        for expr in expression:
            if expr in boolean_commands:
                count += 1    
        return count
    
    def get_executed_program_by_IDS(self, slot, index):
        index = self.active_gen_objs[slot][0][index]
        return self.get_executed_program(slot, index)
        
    def get_expression(self, slot, index):
        raise ValueError("NOT valid in CSG3D")
        
    def get_data(self,
                 program_len: int,
                 data_lims=None,
                 if_randomize=True):
        """
        This is a special generator that can generate dataset for any length.
        Since, this is a generator, you need to make a generator different object for
        different program length and use them as required. Training data is shuffled
        once per epoch.
        :param num_train_images: Number of training examples from a particular program
        length.
        :param jitter_program: whether to jitter the final output or not
        :param if_randomize: If randomize the training dataset
        :param program_len: which program length dataset to sample
        :param stack_size: program_len  // 2 + 1
        :return data: image and label pair for a minibatch
        """
        # The last label corresponds to the stop symbol and the first one to
        # labels = np.zeros((1, program_len + 1), dtype=np.int64)
        # while True:
            # Random things to select random indices
        # self.programs[program_len] = self.programs[program_len][data_lims[0] : data_lims[1]]
        # IDS = np.arange(data_lims[0], data_lims[1])
        IDS = np.arange(data_lims[0], data_lims[1])
        if if_randomize:
            np.random.shuffle(IDS)
        return [IDS, len(IDS)]    
        
    def get_next_slot_and_target(self, train_progress=0):
        
        slot_id = self.get_next_slot(train_progress)
        ID_list, max = self.active_gen_objs[slot_id]
        target_id = ID_list[self.current_index_list[slot_id]]
        return slot_id, target_id, max
        
    def get_next_sample(self, train_progress=0, return_numpy=True):
        slot_id, target_id, max = self.get_next_slot_and_target(train_progress)
        output, labels = self.get_executed_program(slot=slot_id, index=target_id, return_numpy=return_numpy)
        self.current_index_list[slot_id] += 1
        if self.current_index_list[slot_id] == max:
            np.random.shuffle(self.active_gen_objs[slot_id][0])
            self.current_index_list[slot_id] = 0
        return output, labels, slot_id, target_id

    def get_next_slot(self, train_progress=0):
        rand_sample = np.random.sample()
        if self.mode == "EVAL":
            ind = self.slot_counter%len(self.program_lengths)
            self.slot_counter = ind  + 1
        elif self.mode == "TEST":
            ind = self.slot_counter%len(self.program_lengths)
            self.slot_counter = ind  + 1
        else:
            if self.sampling == "RANDOM":
                # Sample a program from a random length
                cur_props = self.cumalative_prop
            elif self.sampling == "LINEAR":
                # sample from the lower, and then the higher eventually as training goes on.
                cur_props = [1 - (train_progress * x) for x in self.slopes]
            elif self.sampling == "NON_LINEAR":
                # sample from the lower, and then the higher eventually as training goes on.
                cur_props = [1 - (train_progress ** x) for x in self.etas]

            # TODO: Could have been binary search
            ind = 0
            while(True):
                if rand_sample < cur_props[ind]:
                    bin = ind
                    break
                else:
                    ind += 1

        return self.program_lengths[ind]


    def load_specific_data(self, target_slots, target_ids):
        # Reload data
        self.reload_data()
        
        slot_to_targets = defaultdict(list)
        for ind, cur_id in enumerate(target_ids):
            cur_slot = target_slots[ind]
            slot_to_targets[cur_slot].append(cur_id)
        
        for slot, targets in slot_to_targets.items():
            data_dict = dict()
            for ind, cur_id in enumerate(targets):
                data_dict[cur_id] = self.programs[slot][cur_id]
            self.programs[slot] = data_dict