import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import pandas as pd

nx, ny = 250, 250       # Number of grid points in x and y directions
diff = 0.001            # Diffusion coefficient
dt_o_dx2 = 0.2/(0.2**2) # dt/(dx*dx)

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
    
    # Find the index corresponding to input_time
    time_index = np.where(np.isclose(time_column, input_time, atol=1e-5))[0]
    
    index = time_index
    u_flattened = data[index, 1:total_points+1]
    v_flattened = data[index, total_points+1:]
    
    u_reshaped = u_flattened.reshape(nx, ny)
    v_reshaped = v_flattened.reshape(nx, ny)
    
    u_analytical = torch.tensor(u_reshaped, dtype=torch.float32)
    v_analytical = torch.tensor(v_reshaped, dtype=torch.float32)
    
    return u_analytical, v_analytical

# TODO Fix
def BC_loss(model, x_ic_tensor, x_ic_true):
    # u_ic_pred, v_ic_pred = model(x_ic_tensor)  # Get predictions from the PINN model
    # u_ic_true_tensor = torch.tensor(x_ic_true[:, 0], dtype=torch.float32)
    # v_ic_true_tensor = torch.tensor(x_ic_true[:, 1], dtype=torch.float32)
    # loss_bc = torch.sqrt(torch.mean((u_ic_pred - u_ic_true_tensor) ** 2 + (v_ic_pred - v_ic_true_tensor) ** 2))  # Compute loss; TODO RMSE or MSE?

    x_bc_tensor = np.zeros((nx, ny))
    u_analytical, v_analytical = analytical_solution(input, constants, 0)

    for j in range(ny):
        for i in range(nx):
            if i == 0:
                xlap1 = 2 * (u_analytical[1, j] - u_analytical[0, j])
            elif i == nx - 1:
                xlap1 = 2 * (u_analytical[nx - 2, j] - u_analytical[nx - 1, j])
            else:
                xlap1 = u_analytical[i - 1, j] - 2 * u_analytical[i, j] + u_analytical[i + 1, j]

            if j == 0:
                xlap2 = 2 * (u_analytical[i, 1] - u_analytical[i, 0])
            elif j == ny - 1:
                xlap2 = 2 * (u_analytical[i, ny - 2] - u_analytical[i, ny - 1])
            else:
                xlap2 = u_analytical[i, j - 1] - 2 * u_analytical[i, j] + u_analytical[i, j + 1]

            x_bc_tensor[i, j] = xlap1 + xlap2

    x_bc_tensor *= diff * dt_o_dx2
    
    # TODO Getting 250 x 250 times 3 x 20 matrix (i.e., x_bc_tensor * x_ic). Don't sample whole 250 x 250 grid; sample at 3 points along the walls to get RMSE.
    u_bc_pred, v_bc_pred = model(torch.tensor(x_bc_tensor, dtype=torch.float32))  # Get predictions from the PINN model
    # Extract analytical solutions at given boundary points
    u_analytical, v_analytical = analytical_solution(x_bc_tensor, constants, 0)  # t = 0 for boundary conditions
    # Compute the loss as Mean Squared Error (MSE) between predicted and analytical solutions
    loss_bc = torch.mean((u_bc_pred - u_analytical.flatten()) ** 2 + (v_bc_pred - v_analytical.flatten()) ** 2)  # MSE

    return loss_bc

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
        # print("x_spatial.shape: ", x_spatial.shape)
        # print("y_spatial.shape: ", y_spatial.shape)
        # print("x_spatial: ", x_spatial)
        # print("y_spatial: ", y_spatial)
        x_mesh, y_mesh = torch.meshgrid(x_spatial.squeeze(), y_spatial.squeeze())
        xy = torch.stack([x_mesh.flatten(), y_mesh.flatten(), input_time * torch.ones_like(x_mesh.flatten())], dim=1)

        # print("xy.shape: ", xy.shape)
        u_pred_analytical, v_pred_analytical = model(xy)
        u_analytical, v_analytical = analytical_solution(xy, constants, input_time)
        # TODO Need to fix these dimension mismatches properly
        u_pred_analytical = torch.vstack((u_pred_analytical.reshape(len(u_pred_analytical),-1), torch.zeros(25).reshape(25,-1)))
        v_pred_analytical = torch.vstack((v_pred_analytical.reshape(len(v_pred_analytical),-1), torch.zeros(25).reshape(25,-1)))

        # print(u_pred_analytical.shape, u_analytical.shape)
        # print(v_pred_analytical.shape, v_analytical.shape)

        loss_PDE = torch.sqrt(torch.mean((u_pred_analytical - u_analytical) ** 2 + (v_pred_analytical - v_analytical) ** 2))  # RMSE
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

        if times[0] == 0:
            loss_bc = BC_loss(model, x_ic_tensor, x_ic)
        else:
            loss_bc = 0  # No initial condition loss if t != 0

        loss_residual = residual_loss(model, x_res_tensor, constants)
        loss_PDE = PDE_loss(model, N_analytical, constants, times)

        print("loss_bc.shape: ", loss_bc.shape)
        print("loss_residual.shape: ", loss_residual.shape)
        print("loss_PDE.shape: ", loss_PDE.shape)

        loss_tot = loss_bc + loss_residual + loss_PDE

        loss_tot.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss IC: {loss_bc.item()}, Loss Residual: {loss_residual.item()}, Loss PDE: {loss_PDE.item()}")

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
        norm_a = plt.Normalize(vmin=u_analytical.min(), vmax=u_analytical.max())
        im_u_analytical = ax_u_analytical.imshow(u_analytical, cmap='viridis', norm=norm_a, origin='lower', extent=[0, 1, 0, 1])
        ax_u_analytical.set_title(f'Analytical U at t = {time:.2f}')
        ax_u_analytical.set_xlabel('x')
        ax_u_analytical.set_ylabel('y')
        fig.colorbar(im_u_analytical, ax=ax_u_analytical)
        
        plt.show()

# Constants and other initializations
nx, ny = 250, 250  # Grid dimensions
a = 0.1
gamma = 1
beta = 0.5
eps = 0.01
delta = 0.0
dx = 0.5
dt = 0.2
D_u = 0.001
N_analytical = 15
constants = [a, beta, gamma, eps, delta, dx, dt, D_u]
NeuronCount = [3, 20, 20, 2]  # Input dimension is 3 (x, y, t)
epoch_max = 10000  # Increased number of epochs
model = PINN(NeuronCount)

# Initialize initial conditions (x_ic)
u_initial = np.zeros((nx, ny), dtype=np.float32)  # u = 0
v_initial = np.full((nx, ny), 0.5, dtype=np.float32)  # v = 0.5
x_ic = np.stack([np.linspace(0, 1, nx).repeat(ny), 
                  np.tile(np.linspace(0, 1, ny), nx), 
                  np.zeros(nx * ny)], axis=1)  # Set time to 0 for initial conditions

# Set true initial conditions in x_ic for loss calculation
x_ic[:, 0] = u_initial.flatten()  # Flattened initial u
x_ic[:, 1] = v_initial.flatten()  # Flattened initial v

# Generate residual points (x_res)
N_res = 10000  # Adjust as needed
x_res = np.random.rand(N_res, 3).astype(np.float32)  # Random residual points in space and time

# Time steps for the PDE loss function
times = torch.arange(0, 1, dt)

# Proceed to model training
model = loss(model, x_ic, x_res, N_analytical, epoch_max, constants, times)

# Plotting transient results
time_steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Adjust as needed
plot_transient_2d(model, time_steps, constants, N_res=250)