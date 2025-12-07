import os
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
        return torch.load( f, weights_only=False)

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

# --- Load checkpoints and models for double pendulum ---
goku_ckpt = load_checkpoint("checkpoints_adamw/double_pendulum/goku_model.pkl", device=device)
latent_ode_ckpt = load_checkpoint("checkpoints_adamw/double_pendulum/latent_ode_model.pkl", device=device)
lstm_ckpt = load_checkpoint("checkpoints_adamw/double_pendulum/lstm_model.pkl", device=device)
goku_model = build_model_from_ckpt(create_goku_double_pendulum, goku_ckpt, device=device)
latent_ode_model = build_model_from_ckpt(create_latent_ode_double_pendulum, latent_ode_ckpt, device=device)
lstm_model = build_model_from_ckpt(create_lstm_double_pendulum, lstm_ckpt, device=device)

# --- Load and normalize double pendulum data ---
norm_params = safe_load("data/double_pendulum/data_norm_params.pkl")
processed_data = safe_load("data/double_pendulum/processed_data.pkl")
test_data_raw = processed_data['test'] # expected [batch, 100, 28, 28]
test_data = to_tensor_on_device(test_data_raw)
if getattr(goku_ckpt['args'], "norm", None) == "zero_to_one":
    test_data = zero_to_one(test_data, norm_params)

# --- Setup for extrapolation evaluation ---
extrap_start, extrap_end = 50, 100
x_in = test_data[:, :extrap_start]            # [batch, 50, 28, 28]
x_extrap_target = test_data[:, extrap_start:extrap_end]    # [batch, 50, 28, 28]
delta_t = getattr(goku_ckpt['args'], 'delta_t', 1.0)
t_extrap = torch.arange(extrap_start, extrap_end, device=device).float() * delta_t  # [50..99]

results = {}

with torch.no_grad():
    print("\n=== Double Pendulum Extrapolation Evaluation (timesteps 50-99) ===")

    print("\nEvaluating GOKU Model:")
    goku_out = goku_model(x_in, t_extrap)
    if isinstance(goku_out, tuple):
        goku_pred = goku_out[0]
    else:
        goku_pred = goku_out
    l1_goku = None
    if goku_pred.shape == x_extrap_target.shape:
        l1_goku = (goku_pred - x_extrap_target).abs().mean().item()
        print("GOKU Extrapolation L1 error (pixels):", l1_goku)
    else:
        print("Shape mismatch. Cannot calculate extrapolation L1 error for GOKU.")
    results['GOKU'] = {"extrap_l1": l1_goku}

    print("\nEvaluating Latent ODE Model:")
    l1_latent = None
    try:
        latent_out = latent_ode_model(x_in, t_extrap)
        if isinstance(latent_out, tuple):
            latent_pred = latent_out[0]
        else:
            latent_pred = latent_out
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
    try:
        lstm_frames = []
        input_frame = x_in[:, -1]  # last known input frame, [batch, 28, 28]
        for step in range(50):
            out = lstm_model(input_frame.unsqueeze(1))  # shape [batch, 1, 28, 28] or [batch, 28, 28]
            if isinstance(out, tuple):
                out = out[0]
            if out.ndim == 4:
                next_frame = out[:, 0] # take the new prediction [batch, 28, 28]
            elif out.ndim == 3:
                next_frame = out  # [batch, 28, 28]
            else:
                raise ValueError(f"Unexpected LSTM output ndim: {out.ndim}")
            lstm_frames.append(next_frame.unsqueeze(1))  # [batch, 1, 28, 28]
            input_frame = next_frame
        lstm_pred = torch.cat(lstm_frames, dim=1)  # [batch, 50, 28, 28]
        if lstm_pred.shape == x_extrap_target.shape:
            l1_lstm = (lstm_pred - x_extrap_target).abs().mean().item()
            print("LSTM Extrapolation L1 error (pixels):", l1_lstm)
        else:
            print("Still shape mismatch for LSTM.")
    except Exception as e:
        print("LSTM autoregressive evaluation failed:", str(e))
    results['LSTM'] = {"extrap_l1": l1_lstm}

print("\nDouble Pendulum Evaluation complete.")

# --- Result Table (Double Pendulum) ---
print("\n" + "="*45)
print("{:<16s} | {:>20s} |".format(
    "Method", "X extrap. L1 (x1e3)"
))
print("-"*45)
paper_numbers = {
    'GOKU':      12,
    'Latent-ODE':16,
    'LSTM':      32,
    'DMM':       26,
    'DI 1%':    37,
    'DI 5%':    37,
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
    print("{:<16s} | {:>8} (paper)          |".format(
        bname, extrap
    ))
print("="*45)

# Save predictions and targets to npy files for plotting
np.save(script_dir + "goku_pred_doublependulum.npy", goku_pred.cpu().numpy() if torch.is_tensor(goku_pred) else goku_pred)
np.save(script_dir + "latent_pred_doublependulum.npy", latent_pred.cpu().numpy() if torch.is_tensor(latent_pred) else latent_pred)
np.save(script_dir + "lstm_pred_doublependulum.npy", lstm_pred.cpu().numpy() if torch.is_tensor(lstm_pred) else lstm_pred)
np.save(script_dir + "x_extrap_target_doublependulum.npy", x_extrap_target.cpu().numpy() if torch.is_tensor(x_extrap_target) else x_extrap_target)

# =================== Plot curves ===================
goku_pred = np.load(script_dir+"goku_pred_doublependulum.npy")           # shape [batch, 50, 28, 28]
latent_pred = np.load(script_dir+"latent_pred_doublependulum.npy")
lstm_pred = np.load(script_dir+"lstm_pred_doublependulum.npy")
x_extrap_target = np.load(script_dir+"x_extrap_target_doublependulum.npy")

goku_pred = np.asarray(goku_pred, dtype=np.float32)
latent_pred = np.asarray(latent_pred, dtype=np.float32)
lstm_pred = np.asarray(lstm_pred, dtype=np.float32)
x_extrap_target = np.asarray(x_extrap_target, dtype=np.float32)

def mean_l1_curve(pred, target):
    abs_err = np.abs(pred - target)
    err_curve = abs_err.reshape(abs_err.shape[0], abs_err.shape[1], -1).mean(axis=(0,2))
    return err_curve

steps = np.arange(50, 100)
goku_curve = mean_l1_curve(goku_pred, x_extrap_target)
latent_curve = mean_l1_curve(latent_pred, x_extrap_target)
lstm_curve = mean_l1_curve(lstm_pred, x_extrap_target)
allblack_curve = np.abs(x_extrap_target).reshape(x_extrap_target.shape[0], x_extrap_target.shape[1], -1).mean(axis=(0,2))

plt.figure(figsize=(7,5))
plt.plot(steps, 1000*goku_curve, label='GOKU')
plt.plot(steps, 1000*latent_curve, label='Latent ODE')
plt.plot(steps, 1000*lstm_curve, label='LSTM')
plt.plot(steps, 1000*allblack_curve, label='All black', linestyle='dashed', color='black')
plt.xlabel("Time step")
plt.ylabel("Mean pixel L1 error (x$10^3$)")
plt.title("Double Pendulum: Mean Extrapolation Error over Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(script_dir+"doublependulum_extrapolation_error.png")
plt.show()