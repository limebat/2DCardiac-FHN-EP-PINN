import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import pandas as pd

class PINN(nn.Module):
    def __init__(self, NeuronCount):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(NeuronCount) - 1):
            self.layers.append(nn.Linear(NeuronCount[i], NeuronCount[i + 1]))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.tanh(self.layers[i](x))  # TANH HIDDEN LAYERS
        x = self.layers[-1](x)  # FINAL LAYER, NO ACTIVATION
        return x[:, 0], x[:, 1]  # Return two outputs: u and v

def residual(model, input, constants):
    input_tensor = input.clone().detach().requires_grad_(True)  # Ensure gradient tracking
    x_tensor = input_tensor[:, 0]
    y_tensor = input_tensor[:, 1]
    t_tensor = input_tensor[:, 2]

    a, beta, gamma, eps, delta, dx, dt, D_u = constants

    u_pred, v_pred = model(input_tensor)

    # Time derivatives
    u_t_pred = torch.autograd.grad(u_pred, t_tensor, grad_outputs=torch.ones_like(u_pred), create_graph=True, allow_unused=True)[0]
    v_t_pred = torch.autograd.grad(v_pred, t_tensor, grad_outputs=torch.ones_like(v_pred), create_graph=True, allow_unused=True)[0]

    # Spatial derivatives for u
    u_x_pred = torch.autograd.grad(u_pred, x_tensor, grad_outputs=torch.ones_like(u_pred), create_graph=True, allow_unused=True)[0]
    u_xx_pred = torch.autograd.grad(u_x_pred, x_tensor, grad_outputs=torch.ones_like(u_x_pred), create_graph=True, allow_unused=True)[0] if u_x_pred is not None else None

    u_y_pred = torch.autograd.grad(u_pred, y_tensor, grad_outputs=torch.ones_like(u_pred), create_graph=True, allow_unused=True)[0]
    u_yy_pred = torch.autograd.grad(u_y_pred, y_tensor, grad_outputs=torch.ones_like(u_y_pred), create_graph=True, allow_unused=True)[0] if u_y_pred is not None else None

    # Handle cases where gradients might be None
    u_t_pred = torch.zeros_like(u_pred) if u_t_pred is None else u_t_pred
    v_t_pred = torch.zeros_like(v_pred) if v_t_pred is None else v_t_pred
    u_x_pred = torch.zeros_like(u_pred) if u_x_pred is None else u_x_pred
    u_y_pred = torch.zeros_like(u_pred) if u_y_pred is None else u_y_pred
    u_xx_pred = torch.zeros_like(u_pred) if u_xx_pred is None else u_xx_pred
    u_yy_pred = torch.zeros_like(u_pred) if u_yy_pred is None else u_yy_pred

    # Compute the Laplacian of u: u_xx + u_yy
    laplace_u = u_xx_pred + u_yy_pred

    # Residuals including the diffusion term D_u * laplace(u)
    residual_u = u_t_pred - (u_pred * (1 - u_pred) * (u_pred - a) - u_pred * v_pred + D_u * laplace_u)
    residual_v = v_t_pred - eps * (beta * u_pred - gamma * v_pred - delta)

    return residual_u, residual_v



def analytical_solution(input, constants, input_time):
    file_path = 'TimeVH.txt' 
    # Read the text file, skipping the header
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
      
    nx, ny = 250, 250
    total_points = nx * ny
    time_column = data[:, 0]
    
    print(time_column)
    print(input_time)
    
    # Find the index corresponding to input_time
    time_index = np.where(np.isclose(time_column, input_time, atol=1e-5))[0]
    
    print(time_index)
    
    index = time_index
    u_flattened = data[index, 1:total_points+1]
    v_flattened = data[index, total_points+1:]
    
    u_reshaped = u_flattened.reshape(nx, ny)
    v_reshaped = v_flattened.reshape(nx, ny)
    
    u_analytical = torch.tensor(u_reshaped, dtype=torch.float32)
    v_analytical = torch.tensor(v_reshaped, dtype=torch.float32)
    
    return u_analytical, v_analytical


def IC_loss(model, x_ic_tensor, x_ic_true):
    u_ic_pred, _ = model(x_ic_tensor)  # Get predictions
    x_ic_true_tensor = torch.tensor(x_ic_true[:, 0], dtype=torch.float32)
    loss_ic = torch.mean((u_ic_pred - x_ic_true_tensor) ** 2)  # Compute loss
    return loss_ic

def residual_loss(model, x_res_tensor, constants):
    residual_value_u, residual_value_v = residual(model, x_res_tensor, constants)
    loss_residual_u = torch.mean(residual_value_u ** 2)
    loss_residual_v = torch.mean(residual_value_v ** 2)
    return loss_residual_u + loss_residual_v

def PDE_loss(model, N_analytical, constants, times):
    total_loss_PDE = 0
    for input_time in times:
        x_spatial = torch.linspace(0, 1, N_analytical).reshape((-1, 1))
        y_spatial = torch.linspace(0, 1, N_analytical).reshape((-1, 1))
        x_mesh, y_mesh = torch.meshgrid(x_spatial.squeeze(), y_spatial.squeeze())
        xy = torch.stack([x_mesh.flatten(), y_mesh.flatten(), input_time * torch.ones_like(x_mesh.flatten())], dim=1)

        u_pred_analytical, v_pred_analytical = model(xy)
        u_analytical, v_analytical = analytical_solution(xy, constants, input_time)

        loss_PDE = torch.sqrt(torch.mean((u_pred_analytical - u_analytical) ** 2 + (v_pred_analytical - v_analytical) ** 2))    # TODO RMSE or MSE?
        total_loss_PDE += loss_PDE
    
    # Average loss across time steps
    return total_loss_PDE / len(times)


def loss(model, x_ic, x_res, N_analytical, epoch_max, constants, times):
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epoch_max):
        optimizer.zero_grad()
        x_ic_tensor = torch.tensor(x_ic, dtype=torch.float32)
        x_res_tensor = torch.tensor(x_res, dtype=torch.float32)

        loss_ic = IC_loss(model, x_ic_tensor, x_ic)
        loss_residual = residual_loss(model, x_res_tensor, constants)
        loss_PDE = PDE_loss(model, N_analytical, constants, times)
        loss_tot = loss_ic + loss_residual + loss_PDE

        loss_tot.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss IC: {loss_ic.item()}, Loss Residual: {loss_residual.item()}, Loss PDE: {loss_PDE.item()}")

    end_time = time.time()
    print(f"Total time is now: {end_time - start_time} seconds")
    return model

def plot_transient_2d(model, time_steps, constants, N_res=250):
    x = torch.linspace(0, 1, N_res)
    y = torch.linspace(0, 1, N_res)
    x_mesh, y_mesh = torch.meshgrid(x, y)

    for time in time_steps:
        # Prepare input grid with current time for the model prediction
        xy = torch.stack([x_mesh.flatten(), y_mesh.flatten(), time * torch.ones_like(x_mesh.flatten())], dim=1)
        
        # Get model-predicted u
        u_pred, _ = model(xy)
        u_pred = u_pred.detach().numpy().reshape(N_res, N_res)
        
        # Get analytical solution for u at the given time
        u_analytical, _ = analytical_solution(xy, constants, time)
        u_analytical = u_analytical.detach().numpy().reshape(N_res, N_res)
        
        # Create figure to compare u_pred and u_analytical
        fig, (ax_u_pred, ax_u_analytical) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot U_pred
        norm_u = plt.Normalize(vmin=u_pred.min(), vmax=u_pred.max())
        im_u_pred = ax_u_pred.imshow(u_pred, cmap='viridis', norm=norm_u, origin='lower', extent=[0, 1, 0, 1])
        ax_u_pred.set_title(f'Predicted U at t = {time:.2f}')
        ax_u_pred.set_xlabel('x')
        ax_u_pred.set_ylabel('y')
        fig.colorbar(im_u_pred, ax=ax_u_pred)

        # Plot U_analytical
        norm_u_analytical = plt.Normalize(vmin=u_analytical.min(), vmax=u_analytical.max())
        im_u_analytical = ax_u_analytical.imshow(u_analytical, cmap='plasma', norm=norm_u_analytical, origin='lower', extent=[0, 1, 0, 1])
        ax_u_analytical.set_title(f'Analytical U at t = {time:.2f}')
        ax_u_analytical.set_xlabel('x')
        ax_u_analytical.set_ylabel('y')
        fig.colorbar(im_u_analytical, ax=ax_u_analytical)

        # Show each comparison figure separately
        plt.tight_layout()
        plt.show()


# Constants and other initializations
a = 0.1
gamma = 1
beta = 0.5
eps = 0.01
delta = 0.0
dx = 0.5
dt = 0.2
D_u = 0.001
constants = [a, beta, gamma, eps, delta, dx, dt, D_u]
NeuronCount = [3, 20, 20, 2]  # Input dimension is 3 (x, y, t)
N_ic, N_res, N_analytical = 100, 10000, 50  # Increased number of points
epoch_max = 10000  # Increased number of epochs
model = PINN(NeuronCount)

# Generate initial and residual points including time
x_ic = np.random.rand(N_ic, 3).astype(np.float32)
x_ic[:, 2] = 0  # Set time to 0 for initial conditions
x_res = np.random.rand(N_res, 3).astype(np.float32)  # Random residual points in space and time

times = torch.arange(0, 1, dt)

time_steps = [0.2, 0.4, 0.6, 0.8, 1.0]  # Adjust as needed
plot_transient_2d(model, time_steps, constants, N_res=250)