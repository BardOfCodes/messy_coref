from collections import defaultdict
from .mixed_len_generator import MixedGenerateData
import os
from typing import List, Dict
import h5py
import torch as th
import numpy as np

# Figure out No split later.
no_split_dirs = ['04530566_vessel', '04090263_rifle', '02958343_car', '02828884_bench', '04256520_couch']
split_dirs = ['03790512_motorbike', '03001627_chair', '03261776_earphone', '03624134_knife', '04379243_table', 
             '03642806_laptop', '03797390_mug', '04225987_skateboard', '02773838_bag', '02954340_cap', 
             '03467517_guitar', '03636649_lamp', '02691156_airplane', '04099429_rocket', '03948459_pistol',
             '02828884_bench', '04256520_couch', "00000000_temp", "ucsgnet_data", "02691156_plane"]

DATA_PATHS = {x : os.path.join( "3d_csg", x, x.split('_')[0]+"_%s_vox.hdf5") for x in split_dirs}

class ShapeNetGenerateData(MixedGenerateData):

    def __init__(self,
                 data_dir: str,
                 mode: str,
                 n_proc: int,
                 proc_id: int,
                 proportion: float,
                 csg_config,
                 program_lengths,
                 proportions: List[float],
                 sampling: str,
                 project_root=""):
        
        self.program_lengths = program_lengths
        self.proportions = proportions
        self.program_types = len(program_lengths)
        self.mode = mode
        self.proportion = proportion
        self.sampling = sampling
        self.n_proc = n_proc
        self.proc_id = proc_id
        
        self.lang_type = csg_config.LANG_TYPE
        self.resolution = csg_config.RESOLUTION
        
        self.datamode = csg_config.DATAMODE
        self.restrict_datasize = csg_config.RESTRICT_DATASIZE
        self.datasize = csg_config.DATASIZE
        
        
        self.set_parser_compiler(csg_config, project_root)
        self.data_dir = data_dir
        
        
        self.set_data_path()

        self.programs = {}
        
        
        self.reload_data()
        
        # Regarding sampling:

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

    def set_data_path(self):
        self.data_paths = {}
        if self.datamode == "CSGSTUMP":
            additional = "csgstump_"
        elif self.datamode == "NEW":
            additional = "new_"
        else:
            additional = ""
        for key, value in DATA_PATHS.items():
            if key in self.program_lengths:
                if self.mode == "TRAIN":
                    self.data_paths[key] = os.path.join(self.data_dir, value % (additional + "train"))
                elif self.mode == "EVAL":
                    self.data_paths[key] = os.path.join(self.data_dir, value % (additional + "val"))
                elif self.mode == "TEST":
                    self.data_paths[key] = os.path.join(self.data_dir, value % (additional + "test"))
    
    def reload_data(self):

        for key, path in self.data_paths.items():
            if not os.path.exists(path):
                print("%s does not exists!" % path)
                raise ValueError("The vox file does not exists")
            else:
                print("%s exists" % path)
                hf_loader = h5py.File(path, 'r')
                data = hf_loader.get('voxels')
                data = self.preprocess_data(data, self.resolution, self.restrict_datasize, self.datasize)
                hf_loader.close()   

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
                hf_loader = h5py.File(path, 'r')
                data = hf_loader.get('voxels')
                data_list = []
                # id_to_index = dict()
                for ind, cur_id in enumerate(targets):
                    data_list.append(data[cur_id, :, :, :, :])
                    # id_to_index[cur_id] = ind
                hf_loader.close()  
                data_list = np.stack(data_list, 0)
                data_list = self.preprocess_data(data_list, self.resolution)
                data_dict = dict()
                for ind, cur_id in enumerate(targets):
                    # real_ind = id_to_index[cur_id] 
                    data_dict[cur_id] = data_list[ind]
                self.programs[slot] = data_dict


        self.current_index_list = {i:0 for i in self.programs.keys()}
        
        self.set_data_distr(mode="dict")


    def preprocess_data(self, data, resolution, restrict_datasize=False, datasize=0):
        # Also subsample/ resize based on size: 
        data_res = data.shape[1]
        if not resolution == data_res:
            data = data[:,:,:,:,:]
            data = th.from_numpy(data)# .unsqueeze(1)
            # data = th.nn.functional.interpolate(data, scale_factor=(resolution/64.0, 
            #                                     resolution/64.0, resolution/64.0), mode='nearest')
            # data = data[:,0].cpu().numpy().astype(np.float32)
            # B_DIM = V_DIM * 2
            # flat_voxel_inds = th.from_numpy(voxel_inds.reshape(-1, 3)).float()
            V_DIM = 32
            voxel_inds = ((np.indices((V_DIM, V_DIM, V_DIM)).T + .5) / (V_DIM//2)) -1.
            ndim = V_DIM
            a = th.arange(ndim).view(1,1,-1) * 2 * 1 + th.arange(ndim).view(1,-1,1) * 2 * ndim * 2 + th.arange(ndim).view(-1,1,1) * 2 * ((ndim * 2) ** 2)
            b = a.view(-1,1).repeat(1, 8)
            rs_inds = b + th.tensor([[0,1,ndim*2,ndim*2+1,(ndim*2)**2,((ndim*2)**2)+1, ((ndim*2)**2)+(ndim*2), ((ndim*2)**2)+(ndim*2)+1]])
            data_list = []
            for i in range(data.shape[0]):
                new_data = data[i].flatten()[rs_inds].max(dim=1).values.view(V_DIM, V_DIM, V_DIM).T 
                # new_data = new_data.bool()
                data_list.append(new_data) 
            data = th.stack(data_list, 0).numpy()
        else:
            
            data = data[:,:,:,:,0]
        # set data limits here:
        if len(data.shape) == 5:
            data = data[:, :, :, :, 0]
        if restrict_datasize: 
            res_len = int(data.shape[0] * datasize)
            data = data[:res_len]
        return data

    def generate_programs(self, program_lengths, proportions):
        raise ValueError("This is not implemented for this class.")

    def set_data_distr(self, mode="np_array"):
        ## FOr non split titles just use the entire size
        if mode == "np_array":
            dataset_sizes = {x: y[:].shape[0] for x, y in self.programs.items()}
        else:
            dataset_sizes = {x: len(list(y.keys())) for x, y in self.programs.items()}

        lims = {x: (0, y) for x, y in dataset_sizes.items()}
        # lims = [(0, 1) for x in dataset_sizes]

        if self.mode == "TRAIN":
            randomize = True
        elif self.mode == "EVAL":
            randomize = False
        elif self.mode == "TEST":
            randomize = False

        self.active_gen_objs = {}
        print("Loader Limits", lims)
        for ind, p_len in enumerate(lims.keys()):
            self.active_gen_objs[p_len] = self.get_data(
                p_len, data_lims=lims[p_len], if_randomize=randomize)

    def get_data(self,
                 program_len: int,
                 data_lims=None,
                 if_randomize=True):
        IDS = np.arange(data_lims[0], data_lims[1])
        if if_randomize:
            np.random.shuffle(IDS)
        return [IDS, len(IDS)]    
        

    def get_executed_program(self, slot, index, return_numpy=True, return_bool=False):
        # index = 0
        # real_index = self.index_2_id_mapper[index]
        output = self.programs[slot][index].copy()
        if not return_numpy:
            output = th.tensor(output, device=self.compiler.device)
            if return_bool:
                output = output.bool()
            else:
                output = output.to(self.compiler.tensor_type)
        else:
            if not return_bool:
                output = output.astype(np.float32)

        # print(slot, index)

        expression = self.parser.trivial_expression

        return output, expression
    
    def get_expression(self, slot, index):
        raise ValueError("Not defined for CSG3D")
        expression = self.parser.trivial_expression
        return expression

