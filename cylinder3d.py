import numpy as np
import torch
import os
import pathlib
import yaml

from config.config import load_config_data
from builder import model_builder
from utils.load_save_util import load_checkpoint
from dataloader.dataset_semantickitti import nb_process_label, cart2polar, polar2cat

"""
Label map: config/label_mapping/semantic-kitti.yaml
"""

def create_segm_model(pytorch_device):
    cylinder_path = pathlib.Path(__file__).parent.absolute()
    print("Cylinder3D path:", cylinder_path)

    config_path = os.path.join(cylinder_path, "config/semantickitti.yaml")
    model_load_path = os.path.join(cylinder_path, "./model_load_dir/model_load.pt")

    configs = load_config_data(config_path)
    model_config = configs['model_params']
    dataset_config = configs['dataset_params']

    pcl_sem_segm_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        pcl_sem_segm_model = load_checkpoint(model_load_path, pcl_sem_segm_model)
    
    pcl_sem_segm_model.to(pytorch_device)

    label_mapping = os.path.join(cylinder_path, dataset_config["label_mapping"])
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    inv_learning_map = semkittiyaml['learning_map_inv']
    return pcl_sem_segm_model, inv_learning_map

def pcl_to_cylinder(pcl):
    """
    Converts numpy a XYZ point cloud to an input format supported by the framework
      pcl: [N, 3] 
      device: target device
    """
    fixed_volume_space = True
    max_volume_space = [50, np.pi, 2]
    min_volume_space = [0, -np.pi, -4]
    grid_size = np.asarray([480, 360, 32])
    ignore_label = 0

    raw_data = np.concatenate([pcl, np.zeros((pcl.shape[0], 1), dtype=pcl.dtype)], axis=1).astype(np.float32)

    annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
    data = (raw_data[:, :3], annotated_data.astype(np.uint8), raw_data[:, 3])

    if len(data) == 2:
        xyz, labels = data
    elif len(data) == 3:
        xyz, labels, sig = data
        if len(sig.shape) == 2: sig = np.squeeze(sig)
    else:
        raise Exception('Return invalid data tuple')
    
    xyz_pol = cart2polar(xyz)

    max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
    min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
    max_bound = np.max(xyz_pol[:, 1:], axis=0)
    min_bound = np.min(xyz_pol[:, 1:], axis=0)
    max_bound = np.concatenate(([max_bound_r], max_bound))
    min_bound = np.concatenate(([min_bound_r], min_bound))
    if fixed_volume_space:
        max_bound = np.asarray(max_volume_space)
        min_bound = np.asarray(min_volume_space)
    # get grid index
    crop_range = max_bound - min_bound
    cur_grid_size = grid_size
    intervals = crop_range / (cur_grid_size - 1)

    if (intervals == 0).any(): print("Zero interval!")
    grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

    voxel_position = np.zeros(grid_size, dtype=np.float32)
    dim_array = np.ones(len(grid_size) + 1, int)
    dim_array[0] = -1
    voxel_position = np.indices(grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
    voxel_position = polar2cat(voxel_position)

    processed_label = np.ones(grid_size, dtype=np.uint8) * ignore_label
    label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
    label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
    processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
    data_tuple = (voxel_position, processed_label)

    # center data on each voxel for PTnet
    voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
    return_xyz = xyz_pol - voxel_centers
    return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

    if len(data) == 2:
        return_fea = return_xyz
    elif len(data) == 3:
        return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

    data_tuple += (grid_ind, labels, return_fea)

    return grid_ind, return_fea


def sem_segm_predict_single(model, device, grid_ind, return_fea):
    demo_grid = [grid_ind]
    demo_pt_fea = [return_fea]

    demo_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(device)
                       for i in demo_pt_fea]
    demo_grid_ten = [torch.from_numpy(i).to(device) for i in demo_grid]

    demo_batch_size = 1
    predict_labels = model(demo_pt_fea_ten, demo_grid_ten, demo_batch_size)
    return predict_labels


def process_labels(predict_labels, grid_ind, inv_learning_map):
    demo_grid = [grid_ind]
    predict_labels = torch.argmax(predict_labels, dim=1)
    predict_labels = predict_labels.cpu().detach().numpy()
    batch_idx = 0
    demo_grid_b = demo_grid[batch_idx]
    invert_labels = np.vectorize(inv_learning_map.__getitem__)
    inv_labels = invert_labels(predict_labels[batch_idx, demo_grid_b[:, 0], demo_grid_b[:, 1], demo_grid_b[:, 2]]) 
    inv_labels = inv_labels.astype('uint32')
    return inv_labels


def save_pcl_labels(path, pcl, labels):
    if pcl.shape[1] == 3:
        pcl = np.concatenate([pcl, np.zeros((pcl.shape[0], 1), dtype=pcl.dtype)], axis=1).astype(np.float32)
    pcl.tofile(f"{path}/.bin")
    pcl.tofile(f"{path}/.bin")