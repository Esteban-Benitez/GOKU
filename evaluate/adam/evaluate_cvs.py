import torch
import inspect
import numpy as np
import matplotlib.pyplot as plt

from models.GOKU import create_goku_cvs
from models.Latent_ODE import create_latent_ode_cvs
from models.LSTM import create_lstm_cvs

import os

# Get the directory where the script is located
script_dir =  os.path.dirname(__file__) + "/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint(path, device=None):
    with open(path, "rb") as f:
        return torch.load(f, map_location=device, weights_only=False)

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
    with open(path, "rb") as f:
        return torch.load(f, weights_only=False)

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

# --- Load checkpoints and models for CVS ---
goku_ckpt = load_checkpoint("checkpoints/cvs/goku_model.pkl", device=device)
latent_ode_ckpt = load_checkpoint("checkpoints/cvs/latent_ode_model.pkl", device=device)
lstm_ckpt = load_checkpoint("checkpoints/cvs/lstm_model.pkl", device=device)
goku_model = build_model_from_ckpt(create_goku_cvs, goku_ckpt, device=device)
latent_ode_model = build_model_from_ckpt(create_latent_ode_cvs, latent_ode_ckpt, device=device)
lstm_model = build_model_from_ckpt(create_lstm_cvs, lstm_ckpt, device=device)

# --- Load and normalize CVS data ---
norm_params = safe_load("data/cvs/data_norm_params.pkl")
processed_data = safe_load("data/cvs/processed_data.pkl")
test_data_raw = processed_data['test'] # expected [batch, 100, ...]
test_data = to_tensor_on_device(test_data_raw)
if getattr(goku_ckpt['args'], "norm", None) == "zero_to_one":
    test_data = zero_to_one(test_data, norm_params)

# --- Setup for extrapolation evaluation ---
extrap_start, extrap_end = 50, 100
x_in = test_data[:, :extrap_start]           # [batch, 50, ...]
x_extrap_target = test_data[:, extrap_start:extrap_end]    # [batch, 50, ...]
delta_t = getattr(goku_ckpt['args'], 'delta_t', 1.0)
t_extrap = torch.arange(extrap_start, extrap_end, device=device).float() * delta_t

results = {}

with torch.no_grad():
    print("\n=== CVS Extrapolation Evaluation (timesteps 50-99) ===")

    print("\nEvaluating GOKU Model:")
    goku_out = goku_model(x_in, t_extrap)
    goku_pred = goku_out[0] if isinstance(goku_out, tuple) else goku_out
    l1_goku = None
    if goku_pred.shape == x_extrap_target.shape:
        l1_goku = (goku_pred - x_extrap_target).abs().mean().item()
        print("GOKU Extrapolation L1 error (pixels):", l1_goku)
    else:
        print("Shape mismatch. Cannot calculate extrapolation L1 error for GOKU.")
    results['GOKU'] = {"extrap_l1": l1_goku}

    print("\nEvaluating Latent ODE Model:")
    l1_latent = None
    latent_pred = None
    try:
        latent_out = latent_ode_model(x_in, t_extrap)
        latent_pred = latent_out[0] if isinstance(latent_out, tuple) else latent_out
        if latent_pred.shape == x_extrap_target.shape:
            l1_latent = (latent_pred - x_extrap_target).abs().mean().item()
            print("Latent ODE Extrapolation L1 error (pixels):", l1_latent)
        else:
            print("Shape mismatch. Cannot calculate extrapolation L1 error for Latent ODE.")
        results['Latent-ODE'] = {"extrap_l1": l1_latent}
    except Exception as e:
        print("Latent ODE evaluation failed:", str(e))
        results['Latent-ODE'] = {"extrap_l1": None}

    print("\nEvaluating LSTM Model (with autoregressive rollout):")
    l1_lstm = None
    lstm_pred = None
    target_shape = x_extrap_target.shape
    # Flexible handling for image [batch, 50, H, W] or feature [batch, 50, F] CVS
    if len(target_shape) == 4:
        H, W = target_shape[2], target_shape[3]
        frame_shape = (H, W)
    elif len(target_shape) == 3:
        F = target_shape[2]
        frame_shape = (F,)
    else:
        raise ValueError(f"x_extrap_target shape not recognized: {target_shape}")
    try:
        lstm_frames = []
        input_frame = x_in[:, -1]
        h = None
        for step in range(extrap_end - extrap_start):
            out, h = lstm_model(input_frame.unsqueeze(1), h)
            if out.ndim == 2 and out.shape[1] == np.prod(frame_shape):
                next_frame = out.view(out.shape[0], *frame_shape)
            elif out.ndim == 3:
                next_frame = out
            elif out.ndim == 4:
                next_frame = out[:, 0]
            else:
                raise ValueError(f"Unexpected LSTM output ndim: {out.ndim}")
            lstm_frames.append(next_frame.unsqueeze(1))
            input_frame = next_frame
        lstm_pred = torch.cat(lstm_frames, dim=1)
        if lstm_pred.shape == x_extrap_target.shape:
            l1_lstm = (lstm_pred - x_extrap_target).abs().mean().item()
            print("LSTM Extrapolation L1 error (pixels):", l1_lstm)
        else:
            print("Still shape mismatch for LSTM.")
    except Exception as e:
        print("LSTM autoregressive evaluation failed:", str(e))
    results['LSTM'] = {"extrap_l1": l1_lstm}

print("\nCVS Evaluation complete.")

# --- Result Table (CVS) ---
print("\n" + "="*45)
print("{:<16s} | {:>20s} |".format(
    "Method", "X extrap. L1 (x1e3)"
))
print("-"*45)
paper_numbers = {
    'GOKU':      29,  # Replace with actual paper values
    'Latent-ODE':2.8,
    'LSTM':      90,
    'DMM':       85,
    'DI 1%':     "NA",
    'DI 5%':     11,
}
for name in ["GOKU", "Latent-ODE", "LSTM"]:
    extrap_l1_mine = results.get(name, {}).get("extrap_l1")
    extrap_l1_paper = paper_numbers.get(name, None)
    print("{:<16s} | {:>8.2f} (mine){:>8.2f} (paper) |".format(
        name,
        1000*extrap_l1_mine if extrap_l1_mine is not None else float('nan'),
        extrap_l1_paper if extrap_l1_paper is not None else float('nan'),
    ))
for bname in ['DMM','DI 1%','DI 5%']:
    extrap = paper_numbers[bname]
    print("{:<16s} | {:>8} (paper)          |".format(bname, extrap))
print("="*45)

# --- Safe .npy Saving ---
np.save(script_dir + "goku_pred_cvs.npy", goku_pred.cpu().numpy() if torch.is_tensor(goku_pred) else goku_pred)
if latent_pred is not None:
    np.save(script_dir + "latent_pred_cvs.npy", latent_pred.cpu().numpy() if torch.is_tensor(latent_pred) else latent_pred)
else:
    print("WARNING: latent_pred was not computed, skipping save.")

if lstm_pred is not None:
    np.save(script_dir + "lstm_pred_cvs.npy", lstm_pred.cpu().numpy() if torch.is_tensor(lstm_pred) else lstm_pred)
else:
    print("WARNING: lstm_pred was not computed, skipping save.")

np.save(script_dir+"x_extrap_target_cvs.npy", x_extrap_target.cpu().numpy() if torch.is_tensor(x_extrap_target) else x_extrap_target)

# =================== Plot curves ===================
try:
    goku_pred_np = np.load(script_dir + "goku_pred_cvs.npy")
    latent_pred_np = np.load(script_dir + "latent_pred_cvs.npy")
    lstm_pred_np = np.load(script_dir + "lstm_pred_cvs.npy")
    x_extrap_target_np = np.load(script_dir + "x_extrap_target_cvs.npy")

    def mean_l1_curve(pred, target):
        abs_err = np.abs(pred - target)
        err_curve = abs_err.reshape(abs_err.shape[0], abs_err.shape[1], -1).mean(axis=(0,2))
        return err_curve

    steps = np.arange(extrap_start, extrap_end)
    goku_curve = mean_l1_curve(goku_pred_np, x_extrap_target_np)
    latent_curve = mean_l1_curve(latent_pred_np, x_extrap_target_np)
    lstm_curve = mean_l1_curve(lstm_pred_np, x_extrap_target_np)
    allblack_curve = np.abs(x_extrap_target_np).reshape(x_extrap_target_np.shape[0], x_extrap_target_np.shape[1], -1).mean(axis=(0,2))

    plt.figure(figsize=(7,5))
    plt.plot(steps, 1000*goku_curve, label='GOKU')
    plt.plot(steps, 1000*latent_curve, label='Latent ODE')
    plt.plot(steps, 1000*lstm_curve, label='LSTM')
    plt.plot(steps, 1000*allblack_curve, label='All black', linestyle='dashed', color='black')
    plt.xlabel("Time step")
    plt.ylabel("Mean pixel L1 error (x$10^3$)")
    plt.title("CVS: Mean Extrapolation Error over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(script_dir+"cvs_extrapolation_error.png")
    plt.show()
except Exception as e:
    print("WARNING: Could not plot results due to missing or invalid npy files:", str(e))