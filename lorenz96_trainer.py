#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Module: Trainer and Inference for Lorenz 96
  Author: Rachid El Montassir
  Licensing: this code is distributed under the CeCILL-C license
  Copyright (c) 2025 CERFACS
"""
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import cmocean
import os
from da_training.MachineLearning.lorenz_96.lorenz96_models \
   import NaiveNetwork, ConvolutionalNetwork, HybridNetwork
from da_training.MachineLearning.lorenz_96.lorenz96_utils \
   import Lorenz96, plot_lorenz96_trajectory, preprocess_dataset, PATH


class Trainer:
    def __init__(self, model, train_loader, val_loader, mean, std, device='cuda',
                 model_type='hybrid', patience=15, dt=0.01, F=8.0):
        self.dt = dt
        self.F = F
        self.Nx = model.Nx
        self.model_type = model_type
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mean = mean
        self.std = std
        self.device = device
        self.patience = patience
        self.model.to(self.device)

    def train(self, epochs=20, lr=1e-3):
        print(f"Training on device: {self.device}")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        sstep = epochs // 6
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=sstep, gamma=0.5)  # Reduce LR every 10 epochs
        losses = {'train': [], 'val': []}

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = criterion(pred - x, y - x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()
            losses['train'].append(train_loss / len(self.train_loader))

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in self.val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = self.model(x)
                    loss = criterion(pred - x, y - x)
                    val_loss += loss.item()
            val_loss /= len(self.val_loader)
            losses['val'].append(val_loss)

            print(f"[Epoch {epoch+1}] Train loss: {train_loss/len(self.train_loader):.4e} | "
                  f"Val loss: {val_loss:.4e}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(PATH,
                           f'{self.model_type}/best_dt{self.dt}_Nx{self.Nx}_F{self.F}_network.pth'))
                # print(
                #     f"Best model saved as {self.model_type}/best_dt{self.dt}_Nx{self.Nx}_F{self.F}_network.pth")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        self._plot_losses(losses)
        print(
            f"Final model saved as {self.model_type}/dt{self.dt}_Nx{self.Nx}_F{self.F}_network.pth")
        torch.save(self.model.state_dict(), os.path.join(PATH,
                   f'{self.model_type}/dt{self.dt}_Nx{self.Nx}_F{self.F}_network.pth'))

    def _plot_losses(self, losses):
        plt.figure(figsize=(10, 5))
        plt.plot(losses['train'], label='Train loss')
        plt.plot(losses['val'], label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{PATH}/{self.model_type}/training_loss.png', dpi=300)
        plt.show()


class Inference:
    def __init__(self, model, mean, std, device='cuda', model_type='hybrid'):
        self.model_type = model_type
        self.model = model
        self.mean = mean
        self.std = std
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_trajectory(self, data_set, steps=40, Nx=40):
        x_val = data_set[0][0]  # Get the first sample
        x_val = x_val.unsqueeze(0).to(self.device)  # Add batch dimension
        predicted_trajectory = np.zeros((steps, Nx))

        for t in range(steps):
            x_val = self.model.predict(x_val, steps=1)
            x_val_out = x_val.cpu().numpy() * self.std + self.mean  # Denormalise
            predicted_trajectory[t] = x_val_out

        self._plot_trajectory(predicted_trajectory,
                              title=f'{self.model_type.capitalize()} model predicted', vmin=-10, vmax=15)
        # plot validation trajectory
        val_trajectory = np.zeros((steps, Nx))
        for t in range(steps):
            x_val = data_set[t+1][0]
            val_trajectory[t] = x_val.numpy() * self.std + self.mean
        self._plot_trajectory(val_trajectory, title='True', vmin=-10, vmax=15)
        # plot the error
        error = predicted_trajectory - val_trajectory
        self._plot_trajectory(error, title=f'{self.model_type.capitalize()} model error',
                              vmin=-15, vmax=15)

    def _plot_trajectory(self, trajectory, title='Predicted', vmin=None, vmax=None):
        if vmin is not None and vmin == -vmax:
            cmap = cmocean.cm.balance
        else:
            cmap = cmocean.cm.curl
        plt.figure(figsize=(10, 5))
        plt.imshow(trajectory.T, aspect='auto', cmap=cmap,
                   origin='lower', interpolation='spline36', vmin=vmin, vmax=vmax)
        cbar_label = 'Error value' if 'Error' in title else 'State value'
        plt.colorbar(label=cbar_label)
        plt.xlabel('Time step')
        plt.ylabel('Lorenz96 variables')
        plt.title(f'{title} Lorenz-96 trajectory')
        plt.tight_layout()
        plt.savefig(
            f'{PATH}/{self.model_type}/{title.replace(" ", "_").lower()}_traj.png', dpi=300)
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Lorenz-96 Data Assimilation')
    parser.add_argument('--Nt', type=int, default=10)
    parser.add_argument('--Nt_train', type=int, default=10000)
    parser.add_argument('--inference', action='store_true',
                        help='Perform inference after training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--validation', action='store_true',
                        help='Use validation set for inference')
    args = parser.parse_args()
    # use double precision
    torch.set_default_dtype(torch.float64)
    # models_list = ['naive', 'conv', 'hybrid']
    models_list = ['hybrid']
    Nx = 40
    dt = 0.01
    F = 8.0
    true_model = Lorenz96(Nx=Nx, dt=dt, F=F, integrator='rk4')
    xt = true_model.generate_dataset(
        Nt_train=args.Nt_train, Nt_spinup=1000, seed=42, Nt_shift=10)
    # change xt to double precision
    xt = xt.astype(np.float64)

    # Plot heatmap of trajectory
    plot_lorenz96_trajectory(xt, Nt=50, mode='heatmap')

    # # Plot trajectories at grid points 0, 10, and 20
    # plot_lorenz96_trajectory(
    #     xt, Nt=50, mode='lines', grid_points=[0, 10, 20])

    train_set, val_set, mean, std = preprocess_dataset(xt)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    for model_type in models_list:
        # Initialize model
        if model_type == 'naive':
            model = NaiveNetwork(Nx=true_model.Nx)
        elif model_type == 'conv':
            model = ConvolutionalNetwork(
                Nx=true_model.Nx, num_filters=64, kernel_size=5)
        else:
            # Hybrid network
            # Use the same parameters as the true model
            # Nx = true_model.Nx, dt = true_model.dt, integrator = true_model.integrator
            model = HybridNetwork(Nx=true_model.Nx, num_filters=6,
                                  kernel_size=5, dt=true_model.dt, integrator='rk4')

        if not args.inference:
            # Train the model
            trainer = Trainer(model, train_loader, val_loader, mean, std,
                              device='cuda' if torch.cuda.is_available() else 'cpu',
                              model_type=model_type, dt=dt, F=F)
            trainer.train(epochs=args.epochs, lr=1e-3)

        # Load the model
        model.load_state_dict(torch.load(
            f'{PATH}/{model_type}/best_dt{dt}_Nx{Nx}_F{F}_network.pth', weights_only=True))
        print("Model loaded for inference.")

        # Perform inference
        inference = Inference(model, mean, std, model_type=model_type,
                              device='cuda' if torch.cuda.is_available() else 'cpu')
        inference.predict_trajectory(val_set if args.validation else train_set,
                                     steps=args.Nt, Nx=true_model.Nx)
    print("Inference completed.")
    print("Default precision: ", torch.get_default_dtype())
