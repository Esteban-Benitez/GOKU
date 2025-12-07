#!/usr/bin/env python3
"""
read_goku.py

Convert notebook logic to a CLI Python script that loads a saved GOKU checkpoint
and dataset, constructs the model for a given model type, runs evaluation on the
test set, and prints per-timestep MAE / RMSE summary.

Usage:
    python read_goku.py --model-name pendulum
    python read_goku.py --model-name pendulum --checkpoint checkpoints/pendulum/goku_model.pkl --data data/pendulum/processed_data.pkl
"""
from __future__ import annotations
import argparse
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import numpy as np


def load_torch(path: Path, map_location: Optional[torch.device] = None) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def build_goku_model(model_name: str, ckpt_args: Optional[Any]) -> Any:
    """
    Attempt to construct a GOKU model instance for the given model_name.
    Strategy:
      - import models.GOKU.GOKU (base class)
      - try to import models.GOKU_<model_name> module to get Encoder/Decoder
      - use saved args (ckpt_args) to pass relevant constructor kwargs (filtered).
    """
    try:
        from models.GOKU import GOKU
    except Exception as e:
        raise ImportError("Could not import models.GOKU.GOKU") from e

    # Try to import model-specific module (e.g., models.GOKU_pendulum)
    model_mod = None
    mod_name = f"models.GOKU_{model_name}"
    try:
        model_mod = importlib.import_module(mod_name)
    except Exception:
        # not fatal; some projects provide create_* factory in models.GOKU
        model_mod = None

    # Try to find a factory create_goku_{model_name} in models.GOKU
    try:
        goku_module = importlib.import_module("models.GOKU")
        factory_name = f"create_goku_{model_name}"
        factory = getattr(goku_module, factory_name, None)
    except Exception:
        factory = None

    # If factory exists, prefer calling it with saved args if compatible
    if factory is not None:
        sig = inspect.signature(factory)
        kwargs = {}
        if ckpt_args is not None:
            args_dict = vars(ckpt_args) if not isinstance(ckpt_args, dict) else ckpt_args
            for k in sig.parameters.keys():
                if k in args_dict:
                    kwargs[k] = args_dict[k]
        try:
            return factory(**kwargs)
        except Exception:
            # fallback to manual construction below
            pass

    # Otherwise, construct base GOKU by filtering saved args to accepted params
    goku_kwargs: Dict[str, Any] = {}
    if ckpt_args is not None:
        args_dict = vars(ckpt_args) if not isinstance(ckpt_args, dict) else ckpt_args
    else:
        args_dict = {}

    init_sig = inspect.signature(GOKU.__init__)
    for name in list(init_sig.parameters.keys()):
        if name == "self":
            continue
        if name in args_dict:
            goku_kwargs[name] = args_dict[name]

    # If model module provides Encoder/Decoder, supply them
    if model_mod is not None:
        if hasattr(model_mod, "Encoder"):
            goku_kwargs.setdefault("encoder", getattr(model_mod, "Encoder"))
        if hasattr(model_mod, "Decoder"):
            goku_kwargs.setdefault("decoder", getattr(model_mod, "Decoder"))

    # Instantiate
    return GOKU(**goku_kwargs)


def undo_norm(z: torch.Tensor, data_args: Optional[Dict[str, Any]]) -> torch.Tensor:
    if not data_args:
        return z
    norm = data_args.get("norm", None)
    if norm == "zscore" and "x_mean" in data_args and "x_std" in data_args:
        mean = torch.as_tensor(data_args["x_mean"], device=z.device).view(1, 1, -1)
        std = torch.as_tensor(data_args["x_std"], device=z.device).view(1, 1, -1)
        return z * std + mean
    if norm in ("zero_to_one", "minmax") and "x_min" in data_args and "x_max" in data_args:
        mn = torch.as_tensor(data_args["x_min"], device=z.device).view(1, 1, -1)
        mx = torch.as_tensor(data_args["x_max"], device=z.device).view(1, 1, -1)
        return z * (mx - mn) + mn
    return z


def eval_model(
    goku: Any,
    test_data: Any,
    ckpt_args: Optional[Any],
    data_args: Optional[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    goku.eval()
    x = torch.as_tensor(test_data).float().to(device)

    # Try to robustly infer time dimension
    # Expect x shape (N, T, ...) or (T, N, ...) or (T, ...) or (N, T)
    delta_t = getattr(ckpt_args, "delta_t", 1.0) if ckpt_args is not None else 1.0
    # prefer time on dim=1 if possible
    seq_len = x.shape[1] if x.ndim >= 2 else x.shape[0]
    t = torch.arange(0.0, float(seq_len) * float(delta_t), step=float(delta_t), device=device)

    with torch.no_grad():
        # Many GOKU implementations accept t= and variational= keyword; safe to try
        try:
            out = goku(x, t=t, variational=False)
        except TypeError:
            # fallback without variational or t
            try:
                out = goku(x, t=t)
            except TypeError:
                out = goku(x)
        pred = out[0] if isinstance(out, tuple) else out

    pred = undo_norm(pred, data_args)
    true = undo_norm(x, data_args)

    time_dim = 1 if pred.ndim > 1 and pred.shape[1] == seq_len else 0
    reduce_dims = tuple(i for i in range(pred.ndim) if i != time_dim)
    err = pred - true
    mae_per_t = err.abs().mean(dim=reduce_dims).cpu().numpy()
    rmse_per_t = err.pow(2).mean(dim=reduce_dims).sqrt().cpu().numpy()

    return {"mae_per_t": mae_per_t, "rmse_per_t": rmse_per_t}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a saved GOKU model with test data.")
    parser.add_argument("--model-name", type=str, default="", help="Model type / dataset name (used for default paths).")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to checkpoint .pkl file.")
    parser.add_argument("--data", type=Path, default=None, help="Path to processed_data.pkl file.")
    parser.add_argument("--data-args", type=Path, default=None, help="Path to data_args.pkl (optional).")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available.")
    parser.add_argument("--map-location", type=str, default=None, help="torch.load map_location override (e.g., cpu).")
    args = parser.parse_args(argv)

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    map_location = torch.device(args.map_location) if args.map_location else device

    # Default paths
    ckpt_path = args.checkpoint or Path(f"checkpoints/{args.model_name}/goku_model.pkl")
    data_path = args.data or Path(f"data/{args.model_name}/processed_data.pkl")
    data_args_path = args.data_args or Path(f"data/{args.model_name}/data_args.pkl")

    # Load checkpoint and data
    ckpt = load_torch(ckpt_path, map_location=map_location)
    processed = load_torch(data_path, map_location=map_location)
    data_args = None
    if data_args_path.exists():
        try:
            data_args = load_torch(data_args_path, map_location=map_location)
        except Exception:
            data_args = None

    # Extract fields from checkpoint robustly
    ckpt_args = ckpt.get("args", None)
    model_state = ckpt.get("model", ckpt.get("model_state", ckpt.get("state_dict", None)))
    opt_state = ckpt.get("opt", None)

    # Build model
    goku = build_goku_model(args.model_name, ckpt_args)
    # load state dict if available
    if model_state is not None:
        try:
            goku.load_state_dict(model_state)
        except Exception:
            # try non-strict if shapes / keys mismatch
            try:
                goku.load_state_dict(model_state, strict=False)
            except Exception as e:
                raise RuntimeError("Failed to load model state dict") from e

    goku = goku.to(device)

    test_data = processed.get("test") if isinstance(processed, dict) else processed
    if test_data is None:
        raise RuntimeError("Test data not found in processed data file.")

    metrics = eval_model(goku, test_data, ckpt_args, data_args, device)

    mae_per_t = metrics["mae_per_t"]
    rmse_per_t = metrics["rmse_per_t"]

    print(f"Mean MAE: {mae_per_t.mean():.6g}, Mean RMSE: {rmse_per_t.mean():.6g}")
    # also print shapes
    print(f"mae_per_t shape: {mae_per_t.shape}, rmse_per_t shape: {rmse_per_t.shape}")


if __name__ == "__main__":
    main()