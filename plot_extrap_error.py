import torch
import numpy as np
import matplotlib.pyplot as plt

# ----------- Load data -------------
# Set to True if you want to load from npy files, or False to generate random predictions for demo.
LOAD_FROM_FILE = True

if LOAD_FROM_FILE:
    goku_pred = np.load("goku_pred.npy")          # shape [batch, 50, 28, 28]
    latent_pred = np.load("latent_pred.npy")
    lstm_pred = np.load("lstm_pred.npy")
    x_extrap_target = np.load("x_extrap_target.npy")
else:
    # --- DEMO ONLY: Make random data of correct shape if you want to test the script ---
    goku_pred = np.random.rand(50, 50, 28, 28)
    latent_pred = np.random.rand(50, 50, 28, 28)
    lstm_pred = np.random.rand(50, 50, 28, 28)
    x_extrap_target = np.random.rand(50, 50, 28, 28)

# Make sure everything is numpy array, float32
goku_pred = np.asarray(goku_pred, dtype=np.float32)
latent_pred = np.asarray(latent_pred, dtype=np.float32)
lstm_pred = np.asarray(lstm_pred, dtype=np.float32)
x_extrap_target = np.asarray(x_extrap_target, dtype=np.float32)

# --------- Compute mean L1 error per time step [batch, 50, 28, 28] -------------
def mean_l1_curve(pred, target):
    abs_err = np.abs(pred - target)       # [batch, 50, 28, 28]
    err_curve = abs_err.reshape(abs_err.shape[0], abs_err.shape[1], -1).mean(axis=(0,2))  # [50]
    return err_curve

steps = np.arange(50, 100)

goku_curve = mean_l1_curve(goku_pred, x_extrap_target)
latent_curve = mean_l1_curve(latent_pred, x_extrap_target)
lstm_curve = mean_l1_curve(lstm_pred, x_extrap_target)
allblack_curve = np.abs(x_extrap_target).reshape(x_extrap_target.shape[0], x_extrap_target.shape[1], -1).mean(axis=(0,2))

# --------- Plot -------------
plt.figure(figsize=(7,5))
plt.plot(steps, 1000*goku_curve, label='GOKU')
plt.plot(steps, 1000*latent_curve, label='Latent ODE')
plt.plot(steps, 1000*lstm_curve, label='LSTM')
plt.plot(steps, 1000*allblack_curve, label='All black', linestyle='dashed', color='black')

plt.xlabel("Time step")
plt.ylabel("Mean pixel L1 error (x$10^3$)")
plt.title("Pixel Pendulum: Mean Extrapolation Error over Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.savefig("extrapolation_error.png")  # or .png