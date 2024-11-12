'''
Final / consolidated code for our PINN will be implemented here.
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import pandas as pd

# Constants and other initializations
a = 0.1         # The model parameters for FHN
beta = 0.5      # -
gamma = 1       # -
delta = 0.0     # -
eps = 0.01      # -
dx = 0.05 * 5        # -
dt = 5        # -
end_time = 350
D_u = 1e-3      # Our diffusion coefficient for u
nx = ny = 250//5   # Number of spatial points in x and y directions
NeuronCount = [3, 20, 20, 2]  # Input dimension is 3 (x, y, t); output is 2 (u, v)
N_ic, N_res, N_analytical, N_bc = 3**2, 5**2, 5**2, 3**2  # Number of initial conditions, residual points, and analytical points
epoch_max = int(1e4)  # Number of epochs

times = torch.arange(250, end_time+dt, dt)  # List of discrete evaluation times starting at 0 with spacing dt
print(times)

x_end = y_end = 250

# Function to return initial x / y / t values separately for ICs and residuals so they don't have to be the same. Create grid over x, y with t=0.
def return_x_tensor(N, is_IC, input_time):
    # Values for residuals so they don't have to be the same as those for ICs: grid over x, y with t=0
    x_vals = torch.linspace(0, x_end, N)
    y_vals = torch.linspace(0, y_end, N)
    x_grid, y_grid = torch.meshgrid(x_vals, y_vals, indexing='ij')
    x_flat = x_grid.flatten().view(-1, 1)
    y_flat = y_grid.flatten().view(-1, 1)

    if is_IC == True:
        num_time_steps = 1
        t = torch.full_like(x_flat, input_time)  # Only apply IC at t=initial condition time
    else:
        num_time_steps = len(times)  
        t = times.repeat(N * N, 1)  # Repeated time steps for every x,y combination possible with all t values
        
    x = torch.cat([x_flat, y_flat, t], dim=1)

    # Total concatenation for x_res as well.
    x_tensor = torch.cat([x_flat.repeat(num_time_steps, 1), y_flat.repeat(num_time_steps, 1), t.view(-1, 1)], dim=1)  # Shape: [N_res * N_res * num_time_steps, 3]
    
    return x_tensor

# input_time should be 250 if wanting to observe the spiral reults
def load_initial_conditions(input_time):
    file_path = 'TimeVH.txt'
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    
    total_points = nx * ny
    time_column = data[:, 0]
    
    # Locate the row where time equals 250
    time_index = np.where(time_column == input_time)[0]
    
    if len(time_index) == 0:
        raise ValueError("Specified time not found in the data file.")
    
    index = time_index[0]
    u_flattened = data[index, 1:total_points+1]
    v_flattened = data[index, total_points+1:]
    
    u_initial = torch.tensor(u_flattened, dtype=torch.float32).flatten()
    v_initial = torch.tensor(v_flattened, dtype=torch.float32).flatten()
    
    
    u_ic_2D = u_initial.view(nx, ny)
    v_ic_2D = v_initial.view(nx, ny)
    
    
    downsampled_x_indices = np.linspace(0, nx - 1, N_ic, dtype=int)
    downsampled_y_indices = np.linspace(0, ny - 1, N_ic, dtype=int)

    u_ic_sampled = u_ic_2D[downsampled_x_indices][:, downsampled_y_indices].flatten()
    v_ic_sampled = v_ic_2D[downsampled_x_indices][:, downsampled_y_indices].flatten()

    print(f"Uniformly downsampled u_ic shape: {u_ic_sampled.shape}")
    print(f"Uniformly downsampled v_ic shape: {v_ic_sampled.shape}")
    
    return u_ic_sampled, v_ic_sampled

# Load initial conditions at t=250
u_ic, v_ic = load_initial_conditions(input_time=250)

#Make tensors for IC and residuals for later use in the code.
x_ic = return_x_tensor(N_ic, is_IC=True, input_time=250)
x_res = return_x_tensor(N_res, is_IC=False, input_time=250)


class PINN(nn.Module):
    # Initialize network with NeuronCount defining the number of neurons in each layer.
    # Params:
    #   NeuronCount - List of integers defining the number of neurons in each subsequent layer
    def __init__(self, NeuronCount):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(NeuronCount) - 1):
            self.layers.append(nn.Linear(NeuronCount[i], NeuronCount[i + 1]))

    # Neural network forward pass method. Use tanh activation function for hidden layers.
    # Params:
    #   x - Input tensor; iteratively passed through each network layer
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.tanh(self.layers[i](x))  # TANH HIDDEN LAYERS
        x = self.layers[-1](x)  # FINAL LAYER, NO ACTIVATION
        u = x[:, 0]
        v = x[:, 1]
        return u, v  # Return two outputs: u and v


# Defines the residual function for the FitzHugh-Nagumo model. a, beta, gamma, delta, and eps represent standard FHN model coefficients.
def residual(model, input):
    input_tensor = input.clone().detach().requires_grad_(True)  # Ensure gradient tracking
    x_tensor = input_tensor[:, 0]
    y_tensor = input_tensor[:, 1]
    t_tensor = input_tensor[:, 2]

    u_pred, v_pred = model(input_tensor)

    x_tensor.requires_grad_(True)
    y_tensor.requires_grad_(True)
    t_tensor.requires_grad_(True)
    
    u_pred.requires_grad_(True)
    v_pred.requires_grad_(True)
    
    # Time derivatives
    u_t_pred = torch.autograd.grad(u_pred, t_tensor, grad_outputs=torch.ones_like(u_pred), create_graph=True, allow_unused=True)[0]
    v_t_pred = torch.autograd.grad(v_pred, t_tensor, grad_outputs=torch.ones_like(v_pred), create_graph=True, allow_unused=True)[0]

    # Spatial derivatives for u
    u_x_pred = torch.autograd.grad(u_pred, x_tensor, grad_outputs=torch.ones_like(u_pred), create_graph=True, allow_unused=True)[0]
    #if u_x_pred == None:
    #    u_x_pred = u_pred
    u_xx_pred = torch.autograd.grad(u_x_pred, x_tensor, grad_outputs=torch.ones_like(u_x_pred), create_graph=True, allow_unused=True)[0] if u_x_pred is not None else None

    u_y_pred = torch.autograd.grad(u_pred, y_tensor, grad_outputs=torch.ones_like(u_pred), create_graph=True, allow_unused=True)[0]
    u_yy_pred = torch.autograd.grad(u_y_pred, y_tensor, grad_outputs=torch.ones_like(u_y_pred), create_graph=True, allow_unused=True)[0] if u_y_pred is not None else None

    # Apply gradient keys and assign their values for iteration -- if a key is None, apply 0.
    gradients = {
        'u_t_pred': u_t_pred,
        'v_t_pred': v_t_pred,
        'u_x_pred': u_x_pred,
        'u_y_pred': u_y_pred,
        'u_xx_pred': u_xx_pred,
        'u_yy_pred': u_yy_pred
    }
    zeros_vectors = {
        'u_pred': u_pred,
        'v_pred': v_pred
    }
    for key in gradients.keys():
        if gradients[key] is None:
            gradients[key] = torch.zeros_like(zeros_vectors['u_pred' if 'u_' in key else 'v_pred'])
        #print(f'{key} value:', gradients[key])

    # Compute the Laplacian of u: u_xx + u_yy
    laplace_u = gradients['u_xx_pred'] + gradients['u_yy_pred']

    residual_u = gradients['u_t_pred'] - (u_pred * (1 - u_pred) * (u_pred - a) - u_pred * v_pred + D_u * laplace_u)
    residual_v = gradients['v_t_pred'] - eps * (beta * u_pred - gamma * v_pred - delta)

    return residual_u, residual_v


# Defines the analytical solution. This function uses data generated from the MATLAB file, which generates our baseline using Mitchell-Schaeffer.
def analytical_solution(input, input_time):
    file_path = 'TimeVH.txt'
    # Read the text file, skipping the header
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
      
    total_points = nx * ny
    time_column = data[:, 0]
    
    # Find the index corresponding to input_time
    time_index = np.where(time_column == input_time)[0]
    
    if len(time_index) == 0:
        raise ValueError(f"Specified time {input_time} not found in the data file.")
    
    
    index = time_index
    u_flattened = data[index, 1:total_points+1] # The first half of data contains u values -- +1 to skip time-index at first element
    v_flattened = data[index, total_points+1:]  # And the second half contains v values
    
    u_reshaped = u_flattened.reshape(nx, ny)
    v_reshaped = v_flattened.reshape(nx, ny)
    
    u_analytical = torch.tensor(u_reshaped, dtype=torch.float32).flatten()
    v_analytical = torch.tensor(v_reshaped, dtype=torch.float32).flatten()
    
    sampled_u = u_analytical[:len(input)]
    sampled_v = v_analytical[:len(input)]
    
    return sampled_u, sampled_v


def BC_loss(model, N_bc, times):
    total_loss_bc = 0
    
    for input_time in times:
        # Sample top, bottom, left, right
        x_boundary = torch.linspace(0, nx - 1, N_bc, dtype=int)
        y_boundary = torch.linspace(0, ny - 1, N_bc, dtype=int)
        
        # Define the walls at the boundaries of the analytical solution, N_bc long
        top_wall = torch.stack([x_boundary, torch.zeros(N_bc, dtype=int)], dim=1)
        bottom_wall = torch.stack([x_boundary, (ny - 1) * torch.ones(N_bc, dtype=int)], dim=1)
        left_wall = torch.stack([torch.zeros(N_bc, dtype=int), y_boundary], dim=1)
        right_wall = torch.stack([(nx - 1) * torch.ones(N_bc, dtype=int), y_boundary], dim=1)
        
        # Combine all walls into a single tensor, which will be read for the combined time-tensor as well.
        sampled_xy = torch.cat([top_wall, bottom_wall, left_wall, right_wall], dim=0)
        time_tensor = input_time * torch.ones(sampled_xy.size(0), 1)
        sampled_xy = torch.cat([sampled_xy, time_tensor], dim=1)
        
        # Now see what the model will predict for the given samples in xy, at time t. 
        u_bc_pred, v_bc_pred = model(sampled_xy)
        u_analytical, v_analytical = analytical_solution(sampled_xy, input_time)

        # MSE error
        loss_bc = torch.mean((u_bc_pred - u_analytical) ** 2 + (v_bc_pred - v_analytical) ** 2)
        
        total_loss_bc += loss_bc
    
    # Average loss across time steps
    return total_loss_bc / len(times)


def IC_loss(model, x_ic_tensor):
    '''
    The first of our PINN's 3 loss functions, based on the difference between the true and predicted initial conditions.
    Params:
        model - The PINN model
        x_ic_tensor - the initial conditions; this should be a (size of x_vals) x (size of y_vals) x (size of t=0s) tensor; see top of file
    '''
    u_ic_pred, v_ic_pred = model(x_ic_tensor)  # Predicted output of model at this x_ic_tensor.

    # MSE Losses
    loss_ic = torch.sqrt(torch.mean((u_ic_pred - u_ic) ** 2 + (v_ic_pred - v_ic) ** 2))  # Compute loss
    return loss_ic

# The second of our PINN's 3 loss functions, based on the MSE from the residuals.
def residual_loss(model, x_res_tensor):
    residual_value_u, residual_value_v = residual(model, x_res_tensor)
    loss_residual_u = torch.mean(residual_value_u ** 2)
    loss_residual_v = torch.mean(residual_value_v ** 2)
    return loss_residual_u + loss_residual_v


# The third of our PINN's 3 loss functions, based on the difference between the true and predicted analytical u and v values.
def PDE_loss(model, N_analytical, times):
    total_loss_PDE = 0

    for input_time in times:
        '''
        A temporary spatial coordinates system, which will correspond to the analytical sampling points in 2D.
        Sqrt for x,y for total points needed for analytical, i.e. sqrt(N_analytical)**2 returns N_analytical total. 
        '''
        # Define spatial coordinates
        x_spatial = torch.linspace(0, nx - 1, nx).reshape((-1, 1))
        y_spatial = torch.linspace(0, ny - 1, ny).reshape((-1, 1))
        x_mesh, y_mesh = torch.meshgrid(x_spatial.squeeze(), y_spatial.squeeze())

        # Stack the flattened x, y, and time coordinates into a single tensor
        xy = torch.stack([x_mesh.flatten(), y_mesh.flatten(), input_time * torch.ones_like(x_mesh.flatten())], dim=1)

        # Sample a subset of the grid in both x and y directions
        step_size_x = nx // N_analytical
        step_size_y = ny // N_analytical
        sampled_indices_x = torch.arange(0, nx, step_size_x)
        sampled_indices_y = torch.arange(0, ny, step_size_y)
        sampled_x_mesh, sampled_y_mesh = torch.meshgrid(sampled_indices_x, sampled_indices_y)
        sampled_indices = (sampled_y_mesh * nx + sampled_x_mesh).flatten()  # Calculate flattened indices

        # Select sampled points from xy using these indices
        sampled_xy = xy[sampled_indices]

        # Use sampled_xy in the model and analytical solution
        u_pred_analytical, v_pred_analytical = model(sampled_xy)
        u_analytical, v_analytical = analytical_solution(sampled_xy, input_time)


        loss_PDE = torch.sqrt(torch.mean((u_pred_analytical - u_analytical) ** 2 + (v_pred_analytical - v_analytical) ** 2))
        total_loss_PDE += loss_PDE
    
    # Average loss across time steps
    return total_loss_PDE / len(times)



def loss(model, x_ic, x_res, N_analytical, epoch_max, times, tolerance=1e-1):
    '''
    Loss function combining the three described above.
    Params:
        model - An instance of PINN
        x_ic - Our initial conditions
        x_res - Our residuals
        N_analytical - The number of analytical points
        epoch_max - The number of epochs we will iterate our loss over
        times - A vector of times from 0 --> final time at spacing dt
    '''
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
    x_ic_tensor = x_ic.clone().detach()  # Reuse x_ic
    x_res_tensor = x_res.clone().detach()  # Reuse x_res
    
    for epoch in range(epoch_max):
        optimizer.zero_grad()
        #x_ic_tensor = torch.tensor(x_ic, dtype=torch.float32)
        #x_res_tensor = torch.tensor(x_res, dtype=torch.float32)
        

        #loss_ic = IC_loss(model, x_ic_tensor)
        loss_residual = residual_loss(model, x_res_tensor)
        loss_PDE = PDE_loss(model, N_analytical, times)
        loss_bc = BC_loss(model, N_bc, times)

        #loss_ic + 
        loss_tot = loss_residual + loss_PDE + loss_bc

        loss_tot.backward()
        optimizer.step()

        # Keep track of our losses at periodic intervals.
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss BC: {loss_bc.item()}, Loss Residual: {loss_residual.item()}, Loss PDE: {loss_PDE.item()}") #Loss IC: {loss_ic.item()}, 

        if loss_tot < tolerance:
            break

    end_time = time.time()
    print(f"Total time is now: {end_time - start_time} seconds")
    return model


# TODO Look at v_pred + v_analytical, too
# Create plots to compare our analytical and predicted solutions for u/v over x and y.
def plot_transient_2d(model, times, N_res, x_ic, x_res, N_analytical, epoch_max):
    x = torch.linspace(0, x_end, nx)
    y = torch.linspace(0, y_end, ny)
    x_mesh, y_mesh = torch.meshgrid(x, y)
    
    # Train a model based on every time step, given parameters of model and input residual / IC tensors
    model = loss(model, x_ic, x_res, N_analytical, epoch_max, times)

    for time in times:
        # Prepare input grid with current time for the model prediction
        xy = torch.stack([x_mesh.flatten(), y_mesh.flatten(), time * torch.ones_like(x_mesh.flatten())], dim=1)
       
        u_pred, v_pred = model(xy)
        u_pred = u_pred.detach().numpy().reshape(nx, ny)
        
        # Get analytical solution for u at the given time
        u_analytical, v_analytical = analytical_solution(xy, time)
        u_analytical = u_analytical.detach().numpy().reshape(nx, ny)
        
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


# TODO Unfinished
# Create plots to analyze the convergence of our residuals.
def plot_residuals(model, times, N_res, x_ic, x_res, N_analytical, epoch_max):
    x = torch.linspace(0, x_end, nx)
    y = torch.linspace(0, y_end, ny)
    x_mesh, y_mesh = torch.meshgrid(x, y)
    
    # Train a model based on every time step, given parameters of model and input residual / IC tensors
    model = loss(model, x_ic, x_res, N_analytical, epoch_max, times)

    for time in times:
        # Prepare input grid with current time for the model prediction
        xy = torch.stack([x_mesh.flatten(), y_mesh.flatten(), time * torch.ones_like(x_mesh.flatten())], dim=1)
        print('Input is of dimensions: ', np.size(xy), 'Value of : ', xy)
       
        u_pred, v_pred = model(xy)
        u_pred = u_pred.detach().numpy().reshape(nx, ny)
        v_pred = v_pred.detach().numpy().reshape(nx, ny)
        
        # Get analytical solution for u at the given time
        u_analytical, v_analytical = analytical_solution(xy, time)
        u_analytical = u_analytical.detach().numpy().reshape(nx, ny)
        v_analytical = v_analytical.detach().numpy().reshape(nx, ny)
        abs_u_residual = np.absolute(u_analytical - u_pred)
        abs_v_residual = np.absolute(v_analytical - v_pred)
        
        # Create figure to compare u_pred and u_analytical
        fig, (ax_abs_u_residual, ax_abs_v_residual) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot u residual
        norm_u = plt.Normalize(vmin=abs_u_residual.min(), vmax=abs_u_residual.max())
        im_u_pred = ax_abs_u_residual.imshow(abs_u_residual, cmap='viridis', norm=norm_u, origin='lower', extent=[0, 1, 0, 1])
        ax_abs_u_residual.set_title(f'Abs(u residual) at t = {time:.2f}')
        ax_abs_u_residual.set_xlabel('x')
        ax_abs_u_residual.set_ylabel('y')
        fig.colorbar(im_u_pred, ax=ax_abs_u_residual)

        # Plot v residual
        norm_v = plt.Normalize(vmin=abs_v_residual.min(), vmax=abs_v_residual.max())
        im_u_analytical = ax_abs_v_residual.imshow(abs_v_residual, cmap='plasma', norm=norm_v, origin='lower', extent=[0, 1, 0, 1])
        ax_abs_v_residual.set_title(f'Abs(v residual) at t = {time:.2f}')
        ax_abs_v_residual.set_xlabel('x')
        ax_abs_v_residual.set_ylabel('y')
        fig.colorbar(im_u_analytical, ax=ax_abs_v_residual)

        # Show each comparison figure separately
        plt.tight_layout()
        plt.show()

# Our main code block
model = PINN(NeuronCount)
plot_transient_2d(model, times, N_res, x_ic, x_res, N_analytical, epoch_max)
# plot_residuals(model, times, N_res, x_ic, x_res, N_analytical, epoch_max, times)