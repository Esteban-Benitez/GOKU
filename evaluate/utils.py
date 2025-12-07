import torch
import inspect
import numpy as np
import matplotlib.pyplot as plt

from models.GOKU import create_goku_double_pendulum
from models.Latent_ODE import create_latent_ode_double_pendulum
from models.LSTM import create_lstm_double_pendulum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the directory where the script is located
script_dir =  os.path.dirname(__file__) + "/"

def load_checkpoint(path, device=None):
    with open(script_dir + path, "rb") as f:
        return torch.load(script_dir + script_dir + f, map_location=device, weights_only=False)

def build_model_from_ckpt(constructor, ckpt, device=None):
    sig = inspect.signature(constructor)
    valid_keys = set(sig.parameters.keys())
    model_kwargs = {k: v for k, v in vars(ckpt['args']).items() if k in valid_keys}
    model = constructor(**model_kwargs)
    model.load_state_dict(ckpt['model'])
    if device is not None:
        model.to(device)
    model.eval()
    return model

def safe_load(path):
    with open(script_dir + path, "rb") as f:
        return torch.load(script_dir + f, weights_only=False)

def to_tensor_on_device(obj):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).float().to(device)
    else:
        raise TypeError(f"Unexpected data type: {type(obj)}")

def zero_to_one(x, norm_params):
    min_ = norm_params['min']
    max_ = norm_params['max']
    if isinstance(min_, np.ndarray): min_ = torch.from_numpy(min_).float().to(device)
    elif isinstance(min_, torch.Tensor): min_ = min_.to(device)
    if isinstance(max_, np.ndarray): max_ = torch.from_numpy(max_).float().to(device)
    elif isinstance(max_, torch.Tensor): max_ = max_.to(device)
    return (x - min_) / (max_ - min_)