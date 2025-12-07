
import os
import numpy as np
import torch
import torch.nn as nn

# ---------- Paths ----------
DATA_PATH       = "data/pendulum/processed_data.pkl"  # dict with 'test' as numpy array [S,T,28,28]
PRED_X_TEST     = "checkpoints/pixel_pendulum/di/di_baseline_x_test.pkl"     # saved DI predictions [S,T,28,28]
Z_TEST_PATH     = "checkpoints/pixel_pendulum/di/di_baseline_z_test.pkl"     # [S,T,2]
G_STATE_PATH    = "checkpoints/pixel_pendulum/di/di_baseline_generative.pkl" # MLP state_dict
PARAMS          = "checkpoints/pixel_pendulum/di/di_baseline_params_test.pkl"
EXTRAP_MASK_PATH = None   # optional: tensor/np bool mask of shape [S,T] to evaluate extrapolated frames only

H, W = 28, 28

# ---------- Helpers ----------
def to_torch_img4d(x_np_or_t):
    """Convert [S,T,28,28] numpy->torch and add channel -> [S,T,1,28,28]."""
    if isinstance(x_np_or_t, np.ndarray):
        x = torch.from_numpy(x_np_or_t).float()
    else:
        x = x_np_or_t.float()
    assert x.dim() == 4 and x.shape[2] == H and x.shape[3] == W, f"Expected [S,T,{H},{W}], got {tuple(x.shape)}"
    return x.unsqueeze(2)  # [S,T,1,28,28]

def mae_per_frame(x_hat_img, x_gt_img, mask=None):
    """
    x_hat_img, x_gt_img: [S,T,1,H,W]
    Returns: MAE per frame flattened to [S*T] (or masked subset).
    """
    diff = torch.abs(x_hat_img - x_gt_img)  # [S,T,1,H,W]
    per_frame = diff.view(diff.shape[0], diff.shape[1], -1).mean(dim=2)  # [S,T]
    if mask is not None:
        assert mask.shape == per_frame.shape, f"Mask must be [S,T], got {mask.shape}"
        per_frame = per_frame[mask]
    return per_frame.reshape(-1)  # [S*T_masked]

def summarize_x1e3(values_1d):
    """Return mean, std, SEM scaled by 1000, matching paper format."""
    n = values_1d.numel()
    mean = values_1d.mean().item()
    std  = values_1d.std(unbiased=True).item() if n > 1 else 0.0
    sem  = (std / (n ** 0.5)) if n > 1 else 0.0
    return 1000.0*mean, 1000.0*std, 1000.0*sem

# ---------- Load data ----------
data = torch.load(DATA_PATH,weights_only=False)
X_gt_np = data["test"]  # per your dump: numpy ndarray [S,T,28,28]
assert isinstance(X_gt_np, np.ndarray), type(X_gt_np)
X_gt = to_torch_img4d(X_gt_np)  # [S,T,1,28,28]
S, T = X_gt.shape[0], X_gt.shape[1]

mask = None
if EXTRAP_MASK_PATH and os.path.isfile(EXTRAP_MASK_PATH):
    m = torch.load(EXTRAP_MASK_PATH,weights_only=False)
    if isinstance(m, np.ndarray):
        m = torch.from_numpy(m.astype(np.bool_))
    assert m.shape == (S, T)
    mask = m.bool()

# ---------- Zero baseline (paper reference ~147 ×10³) ----------
X_zero = torch.zeros_like(X_gt)
mae_zero = mae_per_frame(X_zero, X_gt, mask=mask)
zero_mean, zero_std, zero_sem = summarize_x1e3(mae_zero)

print(f"[Zero baseline] X extrap L1: mean={zero_mean:.1f} ×10^3, std={zero_std:.1f}, sem={zero_sem:.1f}")

# ---------- Saved predictions evaluation ----------
if os.path.isfile(PRED_X_TEST):
    X_hat_saved = torch.load(PRED_X_TEST,weights_only=False)
    # Make sure shape is [S,T,1,28,28]
    if X_hat_saved.dim() == 4:
        X_hat_saved = X_hat_saved.unsqueeze(2)
    elif X_hat_saved.dim() == 5:
        pass
    else:
        raise ValueError(f"Unexpected saved pred shape: {tuple(X_hat_saved.shape)}")
    assert X_hat_saved.shape == X_gt.shape, f"Saved preds {tuple(X_hat_saved.shape)} != GT {tuple(X_gt.shape)}"

    mae_saved = mae_per_frame(X_hat_saved, X_gt, mask=mask)
    s_mean, s_std, s_sem = summarize_x1e3(mae_saved)
    print(f"[Saved DI]     X extrap L1: mean={s_mean:.1f} ×10^3, std={s_std:.1f}, sem={s_sem:.1f}")

# ---------- Recompute g(Z_test) properly (flatten S,T) ----------
if os.path.isfile(G_STATE_PATH) and os.path.isfile(Z_TEST_PATH):
    # Build g from state dict (names must match)
    state = torch.load(G_STATE_PATH,weights_only=False)
    latent_dim = state['first_layer.weight'].shape[1]
    h1         = state['first_layer.weight'].shape[0]
    h2         = state['second_layer.weight'].shape[0]
    h3         = state['third_layer.weight'].shape[0]
    x_dim      = state['fourth_layer.weight'].shape[0]
    assert latent_dim == 2 and x_dim == H*W, f"g expects latent_dim=2, x_dim={H*W}; got {latent_dim}, {x_dim}"

    class G(nn.Module):
        def __init__(self, latent_dim, x_dim, h1, h2, h3):
            super().__init__()
            self.first_layer  = nn.Linear(latent_dim, h1)
            self.second_layer = nn.Linear(h1, h2)
            self.third_layer  = nn.Linear(h2, h3)
            self.fourth_layer = nn.Linear(h3, x_dim)
            self.act = nn.ReLU()
        def forward(self, z):
            x = self.act(self.first_layer(z))
            x = self.act(self.second_layer(x))
            x = self.act(self.third_layer(x))
            x = self.fourth_layer(x)
            return x

    g = G(latent_dim, x_dim, h1, h2, h3).eval()
    missing, unexpected = g.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"State load mismatch: missing={missing}, unexpected={unexpected}")

    # Load Z_test and flatten (S,T) to batch
    Z_test = torch.load(Z_TEST_PATH,weights_only=False)  # [S,T,2]
    assert Z_test.shape[:2] == (S, T) and Z_test.shape[2] == latent_dim, f"Z_test shape {tuple(Z_test.shape)}"
    Z_flat = Z_test.reshape(S*T, latent_dim)              # [S*T,2]

    with torch.no_grad():
        X_hat_flat = g(Z_flat)                            # [S*T,784]
        X_hat_img  = X_hat_flat.view(S, T, 1, H, W)       # [S,T,1,28,28]

    mae_re = mae_per_frame(X_hat_img, X_gt, mask=mask)
    r_mean, r_std, r_sem = summarize_x1e3(mae_re)
    print(f"[Recomputed g(Z)] X extrap L1: mean={r_mean:.1f} ×10^3, std={r_std:.1f}, sem={r_sem:.1f}")

    # Optional: check consistency with saved predictions
    if os.path.isfile(PRED_X_TEST):
        X_hat_saved = torch.load(PRED_X_TEST,weights_only=False)
        if X_hat_saved.dim() == 4:
            X_hat_saved = X_hat_saved.unsqueeze(2)
        same = torch.allclose(X_hat_img, X_hat_saved.float(), atol=1e-6)
        print(f"[Consistency] recomputed vs saved allclose: {bool(same)}")
