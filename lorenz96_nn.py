#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Module: Surrogate models for Lorenz 96
  Author: Rachid El Montassir
  Licensing: this code is distributed under the CeCILL-C license
  Copyright (c) 2025 CERFACS
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from da_training.MachineLearning.lorenz_96.lorenz96_models import NaiveNetwork, ConvolutionalNetwork, HybridNetwork
from da_training.MachineLearning.lorenz_96.lorenz96_utils import Lorenz96, preprocess_dataset, PATH
from da_training.MachineLearning.lorenz_96.lorenz96_trainer import Trainer


class Lorenz96Surrogate(nn.Module):
    def __init__(self, Nx=40, dt=0.01, F=8.0, model_type='hybrid', train_if_needed=True):
        """
        Initialize the Lorenz96 surrogate model.
        Parameters:
        Nx (int): Number of variables in the Lorenz96 model.
        dt (float): Time step for the model.
        F (float): Forcing parameter for the Lorenz96 model.
        model_type (str): Type of surrogate model ('hybrid', 'naive', 'conv').
        train_if_needed (bool): Whether to train the model if no pre-trained model is found.
        """

        super(Lorenz96Surrogate, self).__init__()
        # use double precision
        torch.set_default_dtype(torch.float64)
        self.Nx = Nx
        self.dt = dt
        self.F = F
        self.model_type = model_type
        self.train_if_needed = train_if_needed

        if model_type == 'hybrid':
            self.model = HybridNetwork(Nx, dt=dt)
        elif model_type == 'naive':
            self.model = NaiveNetwork(Nx)
        elif model_type == 'conv':
            self.model = ConvolutionalNetwork(Nx)
        else:
            raise ValueError(
                "Invalid model type. Use 'hybrid', 'naive', or 'conv'.")
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model_type = model_type
        self.load_model_state()
        self.model.eval()
        print(
            f"Model initialized: {model_type} with Nx={Nx}, dt={dt}, F={F}")

    def load_model_state(self):
        """
        Load the model state from a file.
        """
        model_path = f'{PATH}/{self.model_type}/best_dt{self.dt}_Nx{self.Nx}_F{self.F}_network.pth'
        # Previously stored model weights
        model_path_stored = f'{PATH}/stored/{self.model_type}/best_dt{self.dt}_Nx{self.Nx}_F{self.F}_network.pth'
        if os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, weights_only=True))
            print(f"Model state loaded from {model_path}")
        elif os.path.exists(model_path_stored):
            print("Model not found at the specified path. Loading from stored path.")
            self.model.load_state_dict(
                torch.load(model_path_stored, weights_only=True))
            print(f"Model state loaded from {model_path_stored}")
        else:
            print(
                f"No pre-trained model found at {model_path} or {model_path_stored}.")
            if self.train_if_needed:
                print("Training the model...")
                self.train()
            else:
                print(
                    "No pre-trained model found and training is not enabled. !!!!!! Using untrained model !!!!!!!.")

    def train(self):
        true_model = Lorenz96(Nx=self.Nx, dt=self.dt,
                              F=self.F, integrator='rk4')
        xt = true_model.generate_dataset(
            Nt_train=10000, Nt_spinup=1000, seed=42, Nt_shift=10)
        # change xt to double precision
        xt = xt.astype(np.float64)

        train_set, val_set, mean, std = preprocess_dataset(xt)

        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=64)

        # Train the model
        trainer = Trainer(self.model, train_loader, val_loader, mean, std,
                          device='cuda' if torch.cuda.is_available() else 'cpu',
                          model_type=self.model_type, dt=self.dt, F=self.F)
        trainer.train(epochs=150, lr=1e-3)

    def predict(self, x, steps=1):
        """
        Predict the next state using the model for a given number of steps.
        """
        for _ in range(steps):
            x = self.model(x)
        return x
