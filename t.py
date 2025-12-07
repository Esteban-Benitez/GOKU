import torch
import os

# --- Model and data paths ---
model_names = ["goku_model", "latent_ode_model", "lstm_model"]
model_paths = [f"checkpoints/pendulum/{name}.pkl" for name in model_names]
data_files = {
    "norm_params": "data/pendulum/data_norm_params.pkl",
    "args": "data/pendulum/data_args.pkl",
    "test_params": "data/pendulum/test_params_data.pkl",
    "test_latent": "data/pendulum/test_latent_data.pkl",
    "train_params": "data/pendulum/train_params_data.pkl",
    "grounding": "data/pendulum/grounding_data.pkl",
    "train_latent": "data/pendulum/train_latent_data.pkl",
    "gt_test": "data/pendulum/gt_test_data.pkl",
    "processed": "data/pendulum/processed_data.pkl"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load data ---
data_dict = {}
for key, path in data_files.items():
    with open(path, 'rb') as f:
        try:
            data_dict[key] = torch.load(f, map_location=device)
        except Exception as e:
            print(f"Failed to load {key} from {path}: {e}")

# --- Function to load a model checkpoint ---
def load_model_ckpt(path):
    with open(path, 'rb') as f:
        ckpt = torch.load(f, map_location=device)
    args = ckpt['args']
    model_state = ckpt['model']
    # You may need to use a different constructor depending on the model type
    # Example:
    # from models.GOKU import create_goku_pendulum
    # model = create_goku_pendulum(**vars(args))
    # model.load_state_dict(model_state)
    # model.to(device)
    # Return model instance for evaluation
    return ckpt

# --- Example loading loop ---
for model_path in model_paths:
    print(f"\n--- Loading model: {model_path} ---")
    ckpt = load_model_ckpt(model_path)
    print("Checkpoint keys:", ckpt.keys())
    print("Args namespace:", ckpt['args'])
    print("Model state keys:", ckpt['model'].keys())
    print("Optimizer present:", 'opt' in ckpt)
    # Add: Model reconstruction/import by type.
    # Add: Evaluation logic below.

# --- Example: Evaluate on test data (extend as needed) ---
# The structure of your test data may look like:
# test_data = data_dict['processed']['test']
# You need model-specific evaluation code, e.g.:
# output = model(test_data)
# loss = loss_fn(output, test_targets)

# --- Print summary or metrics ---
# print('Test Loss:', loss.item())