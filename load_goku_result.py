import argparse
import os
import time
import math
import logging
from typing import Optional

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("read_goku")


def load_checkpoint(path: str, device: Optional[torch.device] = None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    map_loc = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=map_loc, weights_only=False)
    return ckpt


def construct_model_from_ckpt(ckpt: dict, model_name: str, device: torch.device):
    # Try to locate a factory function create_goku_<model_name> first
    import models

    args = ckpt.get("args")
    model_state = ckpt.get("model")
    if args is None or model_state is None:
        raise RuntimeError("Checkpoint does not include required 'args' or 'model' keys")

    kwargs = {}
    # args might be Namespace or dict
    if isinstance(args, dict):
        kwargs.update(args)
    else:
        try:
            kwargs.update(vars(args))
        except Exception:
            # fallback
            log.debug("args not a Namespace or dict; passing nothing")
    factory_name = f"create_goku_{model_name}"
    model = None

    if hasattr(models, factory_name):
        factory = getattr(models, factory_name)
        try:
            model = factory(**kwargs)
        except TypeError:
            # sometimes factory takes fewer args; call without kwargs
            model = factory()
    else:
        # try to import module e.g. models.GOKU_pendulum for encoder/decoder classes
        try:
            module = __import__(f"models.GOKU_{model_name}", fromlist=["*"])
            # Use the base create function if available, otherwise try GOKU constructor
            if hasattr(models, factory_name):
                model = getattr(models, factory_name)()
            else:
                from models.GOKU import GOKU, create_goku_pendulum
                # best-effort: create default and attempt to set encoder/decoder
                enc = getattr(module, "Encoder", None)
                dec = getattr(module, "Decoder", None)
                model = GOKU(**kwargs, encoder=enc, decoder=dec)
        except Exception as e:
            raise RuntimeError(f"Failed to construct model for '{model_name}': {e}")

    # load weights
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    return model, args


def detect_hardware():
    info = {"cpu_count": os.cpu_count(), "cuda_available": torch.cuda.is_available()}
    if info["cuda_available"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_names"] = [torch.cuda.get_device_name(i) for i in range(info["cuda_device_count"])]
        info["cuda_capability"] = [torch.cuda.get_device_capability(i) for i in range(info["cuda_device_count"])]
    else:
        info["cuda_device_count"] = 0
        info["cuda_names"] = []
    return info


def extract_training_stats(ckpt: dict, args: Optional[object]):
    """
    Try common checkpoint keys to find epoch timing / trial / total time information.
    Returns a dict with:
      - num_epochs
      - epoch_times (list) (seconds per epoch)
      - avg_epoch_time (sec)
      - total_training_time_sec
      - num_trials
    Missing values are None.
    """
    res = {
        "num_epochs": None,
        "epoch_times": None,
        "avg_epoch_time_sec": None,
        "total_training_time_sec": None,
        "num_trials": None,
    }

    # num_epochs candidate
    if args is not None:
        try:
            if hasattr(args, "num_epochs"):
                res["num_epochs"] = int(getattr(args, "num_epochs"))
            elif hasattr(args, "epochs"):
                res["num_epochs"] = int(getattr(args, "epochs"))
        except Exception:
            pass
    for k in ("num_epochs", "n_epochs", "epochs"):
        if res["num_epochs"] is None and k in ckpt:
            try:
                res["num_epochs"] = int(ckpt[k])
            except Exception:
                pass

    # epoch_times candidates
    candidates = ["epoch_times", "epoch_durations", "epoch_time", "epoch_seconds"]
    for k in candidates:
        v = ckpt.get(k)
        if isinstance(v, (list, tuple)) and len(v) > 0:
            res["epoch_times"] = [float(x) for x in v]
            break

    # total training time candidate
    for k in ("total_training_time_sec", "total_training_time_seconds", "total_time_sec", "total_time_seconds"):
        if k in ckpt:
            try:
                res["total_training_time_sec"] = float(ckpt[k])
                break
            except Exception:
                pass

    # fallback: if we have epoch_times but not total then compute
    if res["epoch_times"] is not None:
        res["avg_epoch_time_sec"] = float(np.mean(res["epoch_times"]))
        if res["total_training_time_sec"] is None:
            res["total_training_time_sec"] = float(np.sum(res["epoch_times"]))

    # if args contain timing info
    if res["total_training_time_sec"] is None and hasattr(args, "total_training_time_sec"):
        try:
            res["total_training_time_sec"] = float(getattr(args, "total_training_time_sec"))
        except Exception:
            pass

    # num_trials candidates
    for k in ("num_trials", "trials", "n_trials"):
        if k in ckpt:
            try:
                res["num_trials"] = int(ckpt[k])
                break
            except Exception:
                pass
    if res["num_trials"] is None and hasattr(args, "num_trials"):
        try:
            res["num_trials"] = int(getattr(args, "num_trials"))
        except Exception:
            pass

    return res


def pearson_per_feature(pred: np.ndarray, true: np.ndarray, time_dim: int = 1):
    """
    Compute Pearson R per feature.
    Both arrays expected to be shape (N, T, F) or (T, N, F) depending on layout.
    We'll flatten across N and T and compute per-feature correlation.
    """
    if pred.shape != true.shape:
        raise ValueError(f"pred and true shapes must match: {pred.shape} != {true.shape}")
    # Move time dimension to axis 1 if necessary, then flatten N and T into samples axis
    arr = pred
    tar = true
    # make sure data is shape (N*T, F)
    if arr.ndim == 3:
        flat_pred = arr.reshape(-1, arr.shape[-1])
        flat_true = tar.reshape(-1, tar.shape[-1])
    elif arr.ndim == 2:
        # (T, F) or (N, F) treated as (samples, features)
        flat_pred = arr.reshape(-1, arr.shape[-1])
        flat_true = tar.reshape(-1, tar.shape[-1])
    else:
        flat_pred = arr.flatten().reshape(-1, 1)
        flat_true = tar.flatten().reshape(-1, 1)

    # compute correlations robustly for each column
    r_list = []
    for i in range(flat_pred.shape[-1]):
        x = flat_pred[:, i]
        y = flat_true[:, i]
        # remove nan/inf pairs
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            r_list.append(np.nan)
            continue
        xr = x[mask]
        yr = y[mask]
        if np.std(xr) == 0 or np.std(yr) == 0:
            r_list.append(np.nan)
            continue
        r = np.corrcoef(xr, yr)[0, 1]
        r_list.append(r)
    return np.array(r_list)


def evaluate_model(
    model,
    ckpt: dict,
    data_path: str,
    model_name: str,
    device: torch.device,
    variational: bool = False,
):
    # load processed data
    processed_pkl = os.path.join(data_path, "processed_data.pkl")
    if not os.path.exists(processed_pkl):
        raise FileNotFoundError(f"Processed data not found: {processed_pkl}")
    processed = torch.load(processed_pkl, weights_only=False)
    test = torch.as_tensor(processed["test"]).float()
    train = torch.as_tensor(processed["train"]).float()
    data_args = ckpt.get("data_args", {})

    # move to device and align layout
    x = test.to(device)
    device_t = torch.device(device)
    # time array
    args = ckpt.get("args")
    delta_t = float(getattr(args, "delta_t", 1.0))
    seq_len = x.shape[1]
    t = torch.arange(0.0, seq_len * delta_t, step=delta_t, device=device)

    with torch.no_grad():
        out = model(x, t=t, variational=variational)
        pred = out[0] if isinstance(out, tuple) else out
        # some models return (pred_x, pred_z, ...) keep pred_x

    # undo normalization if present
    def undo_norm(z, data_args):
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

    pred = undo_norm(pred, data_args).cpu().numpy()
    true = undo_norm(x, data_args).cpu().numpy()

    # metrics
    err = pred - true
    l1 = np.mean(np.abs(err))
    mae_per_t = np.mean(np.abs(err), axis=tuple(i for i in range(pred.ndim) if i != 1))
    rmse_per_t = np.sqrt(np.mean(err ** 2, axis=tuple(i for i in range(pred.ndim) if i != 1)))

    # correlation per feature
    try:
        r_per_feat = pearson_per_feature(pred, true, time_dim=1)
        mean_r = np.nanmean(r_per_feat)
    except Exception:
        r_per_feat = None
        mean_r = float("nan")

    results = {
        "pred": pred,
        "true": true,
        "L1": float(l1),
        "mae_per_t": mae_per_t,
        "rmse_per_t": rmse_per_t,
        "r_per_feature": r_per_feat,
        "mean_r": mean_r,
        "raw_err": err,
    }
    return results


def format_time(sec: Optional[float]):
    if sec is None:
        return "N/A"
    if math.isfinite(sec):
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h}h {m}m {s:.1f}s"
    return "N/A"


def main():
    parser = argparse.ArgumentParser(description="Load GOKU checkpoint, run evaluation and print stats")
    parser.add_argument("--model", required=True, choices=["pendulum", "double_pendulum", "cvs", "pendulum_friction"])
    parser.add_argument("--checkpoint-dir", default="checkpoints/", help="directory with model subfolders")
    parser.add_argument("--data-dir", default="data/", help="dir that contains <model>/processed_data.pkl")
    parser.add_argument("--checkpoint-file", default="goku_model.pkl", help="checkpoint filename (default goku_model.pkl)")
    parser.add_argument("--device", default=None, help="cpu|cuda or device id (default autodetect)")
    parser.add_argument("--variational", action="store_true", help="run variational sampling when evaluating")
    args = parser.parse_args()

    # build paths and device
    ckpt_path = os.path.join(args.checkpoint_dir, args.model, args.checkpoint_file)
    data_path = os.path.join(args.data_dir, args.model) + os.sep
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Loading checkpoint from '{ckpt_path}'")
    ckpt = load_checkpoint(ckpt_path, device=device)

    # try to construct model and evaluate
    log.info("Reconstructing model from checkpoint")
    model, saved_args = construct_model_from_ckpt(ckpt, args.model, device=device)

    hw = detect_hardware()
    log.info("Hardware summary:")
    log.info(f"  CPU cores: {hw['cpu_count']}")
    log.info(f"  CUDA available: {hw['cuda_available']}")
    if hw["cuda_available"]:
        log.info(f"  CUDA device count: {hw['cuda_device_count']}")
        for i, nm in enumerate(hw["cuda_names"]):
            log.info(f"    device[{i}] name: {nm}")

    train_stats = extract_training_stats(ckpt, saved_args)
    log.info("Training / timing summary (from checkpoint if available):")
    log.info(f"  num_epochs: {train_stats['num_epochs']}")
    if train_stats["epoch_times"] is not None:
        log.info(f"  epochs recorded: {len(train_stats['epoch_times'])}")
        log.info(f"  avg epoch time: {train_stats['avg_epoch_time_sec']:.3f} sec ({format_time(train_stats['avg_epoch_time_sec'])})")
    else:
        log.info("  per-epoch timings: not available in checkpoint")

    log.info(f"  total training time: {format_time(train_stats['total_training_time_sec'])}")
    log.info(f"  num_trials: {train_stats['num_trials']}")

    # estimate GPU-hours if we have total time
    total_sec = train_stats["total_training_time_sec"]
    n_gpus = hw["cuda_device_count"] if hw["cuda_available"] else 0
    if total_sec is not None:
        if n_gpus > 0:
            gpu_hours = (total_sec / 3600.0) * n_gpus
        else:
            gpu_hours = None
    else:
        gpu_hours = None
    log.info(f"  estimated GPU-hours used: {gpu_hours if gpu_hours is not None else 'N/A'}")

    # evaluation
    log.info("Running evaluation on test set")
    eval_start = time.time()
    results = evaluate_model(model, ckpt, data_path, args.model, device=device, variational=args.variational)
    eval_time = time.time() - eval_start

    log.info("Evaluation results:")
    log.info(f"  L1 (mean abs): {results['L1']:.6f}")
    log.info(f"  Mean Pearson R (across features): {results['mean_r']:.6f}")
    if results["r_per_feature"] is not None:
        perf = results["r_per_feature"]
        log.info(f"  Pearson R per feature: {np.array2string(perf, precision=4, separator=', ')}")
    log.info(f"  Eval runtime: {eval_time:.3f} sec")

    # Basic per-time metrics (print first 10)
    log.info(f"  MAE per timestep (first 10): {np.array2string(results['mae_per_t'][:10], precision=6)}")
    log.info(f"  RMSE per timestep (first 10): {np.array2string(results['rmse_per_t'][:10], precision=6)}")

    # Print a compact summary
    print("\n" + "=" * 80)
    print("GOKU CHECKPOINT SUMMARY")
    print("=" * 80)
    print(f"Model         : {args.model}")
    print(f"Checkpoint    : {ckpt_path}")
    print(f"Data dir      : {data_path}")
    print(f"Device        : {device}")
    print(f"Num params    : {sum(p.numel() for p in model.parameters())}")
    print(f"Training epochs (saved): {train_stats['num_epochs']}")
    print(f"Avg epoch time: {train_stats['avg_epoch_time_sec']:.3f} sec" if train_stats["avg_epoch_time_sec"] else "Avg epoch time: N/A")
    print(f"Total training time: {format_time(train_stats['total_training_time_sec'])}")
    print(f"Estimated GPU hours: {gpu_hours if gpu_hours is not None else 'N/A'}")
    print(f"Num trials recorded: {train_stats['num_trials']}")
    print("-" * 80)
    print(f"L1 (mean abs): {results['L1']:.8f}")
    print(f"Mean Pearson R (feat): {results['mean_r']:.6f}")
    print(f"Eval time: {eval_time:.3f} sec")
    print("=" * 80)


if __name__ == "__main__":
    main()