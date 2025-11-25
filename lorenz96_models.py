#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Module: Surrogate models for Lorenz 96
  Author: Rachid El Montassir
  Licensing: this code is distributed under the CeCILL-C license
  Copyright (c) 2025 CERFACS
"""
import torch.nn as nn
import torch


class HybridNetwork(nn.Module):
    def __init__(self, Nx, num_filters=6, kernel_size=5, dt=0.01, integrator='euler'):
        super(HybridNetwork, self).__init__()
        self.Nx = Nx
        self.dt = dt
        self.border = kernel_size // 2
        self.integrator = integrator

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=num_filters, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=num_filters * 2,
                               out_channels=1, kernel_size=1)

    def pad(self, x):
        # Periodic padding
        left = x[..., -self.border:]
        right = x[..., :self.border]
        return torch.cat([left, x, right], dim=-1)

    def tendency(self, x):
        x = x.unsqueeze(1)  # (B, 1, Nx)
        x_pad = self.pad(x)  # Periodic padding

        x1 = self.conv1(x_pad)  # (B, F, Nx)
        x2 = x1 ** 2            # Quadratic nonlinearity

        x_cat = torch.cat([x1, x2], dim=1)  # (B, 2F, Nx)
        out = self.conv2(x_cat)             # (B, 1, Nx)

        return out.squeeze(1)               # (B, Nx)

    def forward(self, x):
        if self.integrator == 'euler':
            return self.euler_step(x)
        elif self.integrator == 'rk2':
            return self.rk2_step(x)
        elif self.integrator == 'rk4':
            return self.rk4_step(x)
        else:
            raise ValueError(
                "Invalid integrator. Use 'euler', 'rk2', or 'rk4'.")

    def euler_step(self, x):
        """
        Euler integration step.
        """
        dx = self.tendency(x)
        return x + self.dt * dx

    def rk2_step(self, x):
        """
        Runge-Kutta 2nd order integration
        """
        k1 = self.tendency(x)
        k2 = self.tendency(x + self.dt * k1)
        dx = (k1 + k2) / 2
        return x + self.dt * dx

    def rk4_step(self, x):
        """
        Runge-Kutta 4th order integration
        """
        k1 = self.tendency(x)
        k2 = self.tendency(x + 0.5 * self.dt * k1)
        k3 = self.tendency(x + 0.5 * self.dt * k2)
        k4 = self.tendency(x + self.dt * k3)
        dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return x + self.dt * dx

    def predict(self, x, steps=1):
        """
        Predict the next state using the model for a given number of steps.
        """
        for _ in range(steps):
            x = self.forward(x)
        return x


class NaiveNetwork(nn.Module):
    def __init__(self, Nx, num_layers=4):
        super(NaiveNetwork, self).__init__()
        self.Nx = Nx
        self.sequential = nn.Sequential(
            nn.Linear(Nx, 128),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(128, 128),
                    nn.ReLU()
                ) for _ in range(num_layers - 2)
            ])
        self.linear = nn.Linear(128, Nx)

    def forward(self, x):
        x = self.sequential(x)
        x = self.linear(x)
        return x

    def predict(self, x, steps=1):
        for _ in range(steps):
            x = self.forward(x)
        return x


class ConvolutionalNetwork(nn.Module):
    def __init__(self, Nx, num_filters=64, kernel_size=5):
        super(ConvolutionalNetwork, self).__init__()
        self.Nx = Nx

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters,
                               kernel_size=kernel_size, padding=kernel_size//2)
        self.conv4 = nn.Conv1d(in_channels=num_filters, out_channels=1,
                               kernel_size=kernel_size, padding=kernel_size//2)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Add channel dimension and apply convolutions
        x = x.unsqueeze(1)  # (B, 1, Nx) -- Add channel dimension
        x = self.relu(self.conv1(x))  # Apply Conv1
        x = self.conv4(x)  # Apply Conv4 to reduce to 1 channel
        return x.squeeze(1)  # Remove channel dimension and return (B, Nx)

    def predict(self, x, steps=1):
        """
        Predict the next state using the model for a given number of steps.
        """
        for _ in range(steps):
            x = self.forward(x)
        return x
