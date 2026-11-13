"""Loading and saving of datasets and trained autoencoders."""
import os
import torch as th
import matplotlib.pyplot as plt
import numpy as np
import json
from ae import AutoEncoderClass

def compute_hidden_dims(flattened_dim, z_dim):
    """Return two hidden sizes that shrink geometrically from flattened_dim to z_dim."""
    mid1 = int(np.sqrt(flattened_dim * z_dim))
    mid2 = int(np.sqrt(mid1 * z_dim))
    return [mid1, mid2]

def load_data_subset(folder_path):
    """Load data.npy from folder_path and keep the 20 upper-body joint groups used for training, as a (30, -1) array."""
    arr = np.load(os.path.join(folder_path, 'data.npy'))  # assumes 'data.npy' exists
    data = arr[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], :]
    return data.reshape((30, -1))

def load_all_npy_from_folder(folder_path):
    """
    Load all .npy files in the given folder into a list

    :param folder_path: path to the folder containing .npy files
    :return: list of numpy arrays
    """
    npy_list = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith('.npy'):
            full_path = os.path.join(folder_path, fname)
            arr = np.load(full_path)
            npy_list.append(arr)
    return npy_list

def load_npy_as_dict(folder_path, remove_ext=True):
    """
    Load all .npy files in a folder into a dictionary.

    :param folder_path: Path to folder containing .npy files
    :param remove_ext: Whether to remove '.npy' from the keys
    :return: dict {filename(str): numpy array}
    """
    npy_dict = {}
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith('.npy'):
            key = fname[:-4] if remove_ext else fname  # remove ".npy"
            full_path = os.path.join(folder_path, fname)
            npy_dict[key] = np.load(full_path)
    return npy_dict

def show_data_matrix(data, cmap='viridis'):
    """
    Show 2D matrix as an image using matshow.
    
    :param data: 2D numpy array (e.g., shape: 30 x 180)
    :param cmap: colormap to use (e.g., 'viridis', 'plasma', 'magma', 'gray')
    """
    plt.figure(figsize=(8, 6))
    plt.matshow(data, cmap=cmap, fignum=1)
    plt.colorbar()
    plt.title("Matrix Visualization (matshow)")
    plt.xlabel("Features")
    plt.ylabel("Samples")
    plt.show()

def save_model(model, path, extra_info=None):
    """
    Save AutoEncoder model with architecture info and optional metadata.

    :param model: AutoEncoderClass instance
    :param path: full path to save file (e.g., 'models/ae_model.pt')
    :param extra_info: optional dict for experiment settings (e.g., loss_type, actv name)
    """
    save_dict = {
        "state_dict": model.state_dict(),
        "x_dim": model.input_shape,
        "z_dim": model.z_dim,
        "h_dims": model.h_dims,
        "loss_type": model.loss_type,
        "actv": model.actv.__class__.__name__,  # store only the name of the activation function
    }

    if extra_info is not None:
        save_dict["extra_info"] = extra_info

    os.makedirs(os.path.dirname(path), exist_ok=True)
    th.save(save_dict, path)
    print(f"Model saved to {path}")

def load_saved_model(model_name, device='cpu', root="../../result"):
    """
    Load an AutoEncoderClass instance from the saved config.json and model.pt.

    Args:
        model_name (str): folder name under root
        device (str): 'cpu', 'cuda:0', etc.
        root (str): directory containing the model folders (default: ../../result, where the pretrained models are)

    Returns:
        ae (AutoEncoderClass): restored model instance
        config (dict): configuration
    """
    save_dir = os.path.join(root, model_name)
    config_path = os.path.join(save_dir, "config.json")
    model_path = os.path.join(save_dir, "model.pt")

    # load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # initialize the model
    actv_map = {
        'Tanh': th.nn.Tanh(),
        'ReLU': th.nn.ReLU(),
        'SiLU': th.nn.SiLU(),
        'GELU': th.nn.GELU(),
        # add if needed
    }
    actv_fn = actv_map[config['actv']]

    ae = AutoEncoderClass(
        x_dim=config['x_dim'][0],  # should later be set from the data
        z_dim=config['z_dim'],
        h_dims=config['h_dims'],
        loss_type=config['loss_type'],
        actv=actv_fn,
        iso=config['iso'],
        iso_reg=config['iso_reg'],
    ).to(device)

    # load model parameters
    ae.load_state_dict(th.load(model_path, map_location=device))
    ae.eval()

    return ae, config
