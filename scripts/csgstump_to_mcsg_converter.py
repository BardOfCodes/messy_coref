import os
import _pickle as cPickle
import numpy as np
from CSG.external.csgstump_converter import CSGStumpConverter


# # save the predictions as pkl in a folder:
# NOTE: Only "correct" for MIX subset

target_folder = "/home/aditya/projects/rl/CSGStumpNet/outputs/mix_small_32_32_predictions"
# target_folder = "/home/aditya/projects/rl/CSGStumpNet/outputs/mix_small_train_32_32_predictions"
save_folder = "/home/aditya/projects/rl/weights/iccv/csgstump_converted/"
converter = CSGStumpConverter(32, 8)

slot_to_name_mapper = {
    "03001627": "03001627_chair",
    "04379243": "04379243_table",
    "02828884": "02828884_bench",
    "04256520": "04256520_couch"
}
files = os.listdir(target_folder)

prog_objs = []
for ind, cur_file in enumerate(files):
    if ind % 100 == 0:
        print("Processing file: %d/%d" % (ind, len(files)))
    cur_slot = slot_to_name_mapper[cur_file.split("_")[0]]
    cur_target = cur_file.split("_")[1].split(".")[0]
    cur_target = int(cur_target)
    cur_load_path = os.path.join(target_folder, cur_file)
    information = cPickle.load(open(cur_load_path, "rb"))
    primitive_parameters, intersection_layer_connections, \
        union_layer_connections = information['primitive_parameters'], \
            information['intersection_layer_connections'], information['union_layer_connections']
    mcsg_expr = converter.convert_to_mcsg(primitive_parameters, intersection_layer_connections, union_layer_connections)
    
    prog_obj = dict(
        slot_id=cur_slot,
        target_id=cur_target,
        expression=mcsg_expr,
        render_expr=mcsg_expr,
    )
    prog_objs.append(prog_obj)

# save it: 
save_file = os.path.join(save_folder, "csgstump_converted.pkl")
cPickle.dump(prog_objs, open(save_file, "wb"))
print("saved_file %s" % save_file)
