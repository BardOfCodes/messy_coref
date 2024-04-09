import os
import h5py
import random
import numpy as np

# Load the vox, 
# Create the splits
# If previous exists - move it. 

shapenet_classes = ['03001627_chair', '04379243_table', '02828884_bench', '04256520_couch']

shapenet_location = "../../../data/3d_csg/data/"

train_percent = 10000/13998.
val_percent = 1000/13998.
test_percent = 1000/13998. 

for cur_class in shapenet_classes:
    cur_input_file = os.path.join(shapenet_location, "%s/%s_vox.hdf5" % (cur_class, cur_class.split('_')[0]))
    hf_loader = h5py.File(cur_input_file, 'r')
    data = hf_loader.get('voxels')
    data = data[:,:,:,:,]
    hf_loader.close()

    # Now create the partitions:
    random_indices = random.sample(range(data.shape[0]), data.shape[0])
    train_indices = random_indices[:int(train_percent * len(random_indices))]
    val_indices = random_indices[int(train_percent * len(random_indices)):int((train_percent + val_percent) * len(random_indices))]
    test_indices = random_indices[int((train_percent + val_percent) * len(random_indices)):int((train_percent + val_percent + test_percent) * len(random_indices))]

    train_shapes = np.take(data, train_indices, 0)
    val_shapes = np.take(data, val_indices, 0)
    test_shapes = np.take(data, test_indices, 0)
    shapes = [train_shapes, val_shapes, test_shapes]
    # Now save them using hdf5
    for ind, file_type in enumerate(['train', 'val', 'test']):
        save_file = os.path.join(shapenet_location, "%s/%s_%s_vox.hdf5" % (cur_class, cur_class.split('_')[0], file_type))
        if os.path.exists(save_file):
            diff_file_name = save_file.replace("_%s_vox" % file_type, "_old_%s_vox" % file_type)
            os.system("mv %s %s" % (save_file, diff_file_name))
        # save it
        with h5py.File(save_file, "w") as f:
            f.create_dataset('voxels', data=shapes[ind])

        




