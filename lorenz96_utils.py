#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Module: Utilities for Lorenz 96 surrogate model
  Author: Rachid El Montassir
  Licensing: this code is distributed under the CeCILL-C license
  Copyright (c) 2025 CERFACS
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange
import cmocean
import os
import sys

if 'google.colab' in sys.modules:
    PATH = "lorenz96_ml/outputs"
else:
    PATH = "./outputs"


class Lorenz96:
    def __init__(self, Nx=40, dt=0.05, F=8.0, integrator="rk4"):
        self.Nx = Nx
        self.dt = dt
        self.F = F
        self.integrator = integrator

    def dxdt(self, x):
        dx = np.zeros_like(x)
        for i in range(self.Nx):
            dx[i] = (x[(i + 1) % self.Nx] - x[(i - 2) % self.Nx]) * \
                x[(i - 1) % self.Nx] - x[i] + self.F
        return dx

    def forward(self, x, steps=1):
        # Optionally advance multiple substeps (e.g., for dt=0.01 and shift every 5 steps)
        for _ in range(steps):
            if self.integrator == "rk4":
                k1 = self.dxdt(x)
                k2 = self.dxdt(x + 0.5 * self.dt * k1)
                k3 = self.dxdt(x + 0.5 * self.dt * k2)
                k4 = self.dxdt(x + self.dt * k3)
                x = x + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            elif self.integrator == "euler":
                x = x + self.dt * self.dxdt(x)
            elif self.integrator == "rk2":
                k1 = self.dxdt(x)
                k2 = self.dxdt(x + self.dt * k1)
                x = x + 0.5 * self.dt * (k1 + k2)
            else:
                raise ValueError(
                    "Invalid integration method. Use 'rk4', 'euler', or 'rk2'.")
        return x

    def generate_dataset(self, Nt_train=10000, Nt_spinup=100, Nt_shift=1, seed=42):
        path = f"{PATH}/lorenz96_dataset_Nt{Nt_train}_dt{self.dt}_Nx{self.Nx}_F{self.F}.npz"
        # Load the dataset
        if os.path.exists(f"{path}"):
            data = np.load(
                f"{path}")
            xt = data['xt']
            print(
                f"Dataset loaded from {path}")
            return xt
        rng = np.random.default_rng(seed)
        xt = np.zeros((Nt_train + 1, self.Nx))
        x0 = rng.normal(loc=3.0, scale=1.0, size=self.Nx)

        for _ in trange(Nt_spinup, desc='Spin-up integration'):
            x0 = self.forward(x0)

        xt[0] = x0
        for t in trange(Nt_train, desc='Model integration (train)'):
            x0 = self.forward(x0, steps=Nt_shift)
            xt[t + 1] = x0

        # save the dataset
        np.savez(
            f"{path}", xt=xt)
        print(
            f"Dataset saved as {path}")

        return xt


def preprocess_dataset(xt, val_fraction=0.1, normalize=True):
    x_raw = xt[:-1]  # input at time t
    y_raw = xt[1:]   # output at time t+1

    # Normalisation: standard score (z-score)
    if normalize:
        mean = x_raw.mean(axis=0)
        std = x_raw.std(axis=0)
    else:
        mean = np.zeros_like(x_raw[0])
        std = np.ones_like(x_raw[0])
    x_norm = (x_raw - mean) / std
    y_norm = (y_raw - mean) / std
    # Convert to torch dtype
    if xt.dtype == np.float32:
        x_tensor = torch.tensor(x_norm, dtype=torch.float32)
        y_tensor = torch.tensor(y_norm, dtype=torch.float32)
    elif xt.dtype == np.float64:
        x_tensor = torch.tensor(x_norm, dtype=torch.float64)
        y_tensor = torch.tensor(y_norm, dtype=torch.float64)
    else:
        raise ValueError("Unsupported data type")
    # Convert to tensors

    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)

    n_val = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val
    train_set = torch.utils.data.Subset(dataset, range(n_train))
    val_set = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))

    return train_set, val_set, mean, std


def plot_lorenz96_trajectory(dataset, Nt=500, Nx=40, mode='heatmap', grid_points=[0, 1, 2]):
    """
    Plots a Lorenz-96 trajectory over time using the dataset.

    Parameters:
        dataset: TensorDataset or Dataset returning (x, y)
        Nt: number of time steps to plot
        Nx: number of spatial grid points
        mode: 'heatmap' or 'lines'
        grid_points: which spatial indices to plot (for mode='lines')
    """
    # Extract trajectory from the dataset
    traj = np.zeros((Nt, Nx))
    for t in range(Nt):
        traj[t] = dataset[t]

    # Plotting
    if mode == 'heatmap':
        plt.figure(figsize=(10, 5))
        plt.imshow(traj.T, aspect='auto', cmap=cmocean.cm.balance,
                   origin='lower', interpolation='spline36')
        plt.colorbar(label='State value')
        plt.xlabel('Time step')
        plt.ylabel('Lorenz96 variables')
        plt.title('Lorenz-96 trajectory (space vs time)')
        plt.tight_layout()
        plt.savefig(PATH + '/lorenz96_trajectory.png', dpi=300)
        plt.show()

    elif mode == 'lines':
        plt.figure(figsize=(10, 4))
        for idx in grid_points:
            plt.plot(traj[:, idx], label=f'Grid point {idx}')
        plt.xlabel('Time step')
        plt.ylabel('State value')
        plt.title('Lorenz-96 time series at selected grid points')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PATH + '/lorenz96_trajectory_lines.png', dpi=300)
        plt.show()

    else:
        raise ValueError("mode must be 'heatmap' or 'lines'")
