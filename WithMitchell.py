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
a = 0.1                 # The model parameters for FHN
beta = 0.5              # -
gamma = 1               # -
delta = 0.0             # -
eps = 0.01              # -
conv_factor = 5.0         # -
dx = 0.05 * conv_factor # -
dt = 100                 # -
begin_time = 700        # -
end_time = 800 #500          # -
D_u = 1e-3      # Our diffusion coefficient for u
nx = ny = int(250 // conv_factor)   # Number of spatial points in x and y directions
NeuronCount = [3, 32, 32, 32, 2]  # Input dimension is 3 (x, y, t); output is 2 (u, v)
#Note : The below points are wrt to each axis, so if choosing 5 points, then it's actually 25 pts being sampled
N_ic, N_res, N_analytical, N_bc =  18, 30, 18, 12  # Number of sqrt pts initial conditions, residual points, and analytical points
epoch_max = int(300)  # Number of epochs

times = torch.arange(begin_time, end_time+dt, dt)  # List of discrete evaluation times starting at 0 with spacing dt
print(times)

x_end = y_end = nx * dx

# TODO Cuda not working in this file, so disabled for the moment.
# Make the code run on cpu if cuda is not available and gpu if it is.
print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
device = torch.device("cpu")    #"cuda" if torch.cuda.is_available() else 

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
    
    index = int(time_index[0])
    
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

# Load initial conditions at t=begin_time
u_ic, v_ic = load_initial_conditions(input_time=begin_time)

#Make tensors for IC and residuals for later use in the code.
x_ic = return_x_tensor(N_ic, is_IC=True, input_time=begin_time)
x_res = return_x_tensor(N_res, is_IC=False, input_time=begin_time)


class PINN(nn.Module):
    # Initialize network with NeuronCount defining the number of neurons in each layer.
    # Params:
    #   NeuronCount - List of integers defining the number of neurons in each subsequent layer
    def __init__(self, NeuronCount):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(len(NeuronCount) - 1):
            layer = nn.Linear(NeuronCount[i], NeuronCount[i + 1])
            nn.init.xavier_normal_(layer.weight)
            self.layers.append(layer)
            
            # Add batch norm for all but the last layer
            if i < len(NeuronCount) - 2:
                self.batch_norms.append(nn.BatchNorm1d(NeuronCount[i + 1]))

    # Neural network forward pass method. Use tanh activation function for hidden layers.
    # Params:
    #   x - Input tensor; iteratively passed through each network layer
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.batch_norms[i](x)
            x = torch.tanh(x)  # Using tanh for better gradient flow
        x = self.layers[-1](x)  # FINAL LAYER, NO ACTIVATION
        u = x[:, 0]
        v = x[:, 1]
        return u, v  # Return two outputs: u and v

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
    
    index = time_index[0]
    u_flattened = data[index, 1:(total_points+1)]  # The first half of data contains u values -- +1 to skip time-index at first element
    v_flattened = data[index, (total_points+1):]   # And the second half contains v values
    
    u_reshaped = u_flattened.reshape(nx, ny)
    v_reshaped = v_flattened.reshape(nx, ny)
    
    
    u_analytical = torch.tensor(u_reshaped, dtype=torch.float32)
    v_analytical = torch.tensor(v_reshaped, dtype=torch.float32)
    
    return u_analytical, v_analytical


def residual(model, input):
    # Only detach the input once and enable gradients
    input_tensor = input.clone().detach().requires_grad_(True)
    
    # Extract variables
    x_tensor = input_tensor[:, 0].unsqueeze(1)
    y_tensor = input_tensor[:, 1].unsqueeze(1)
    t_tensor = input_tensor[:, 2].unsqueeze(1)
    
    input_combined = torch.cat([x_tensor, y_tensor, t_tensor], dim=1)
    
    # Get model predictions - DO NOT detach these
    u_pred, h_pred = model(input_combined.to(device))
    u_pred, h_pred = u_pred.cpu(), h_pred.cpu()
    
    # Create gradient outputs of appropriate size
    grad_outputs = torch.ones_like(u_pred)
    
    # Calculate time derivatives
    u_t_pred = torch.autograd.grad(
       outputs=u_pred,
       inputs=t_tensor,
       grad_outputs=grad_outputs,
       create_graph=True,
       retain_graph=True
    )[0]
    
    h_t_pred = torch.autograd.grad(
       outputs=h_pred,
       inputs=t_tensor,
       grad_outputs=grad_outputs,
       create_graph=True,
       retain_graph=True
    )[0]
    
    # Calculate spatial derivatives for u
    u_x_pred = torch.autograd.grad(
       outputs=u_pred,
       inputs=x_tensor,
       grad_outputs=grad_outputs,
       create_graph=True,
       retain_graph=True
    )[0]
    
    u_xx_pred = torch.autograd.grad(
       outputs=u_x_pred,
       inputs=x_tensor,
       grad_outputs=torch.ones_like(u_x_pred),
       create_graph=True,
       retain_graph=True
    )[0]
    
    u_y_pred = torch.autograd.grad(
       outputs=u_pred,
       inputs=y_tensor,
       grad_outputs=grad_outputs,
       create_graph=True,
       retain_graph=True
    )[0]
    
    u_yy_pred = torch.autograd.grad(
       outputs=u_y_pred,
       inputs=y_tensor,
       grad_outputs=torch.ones_like(u_y_pred),
       create_graph=True,
       retain_graph=True
    )[0]
    
    # Compute Laplacian
    laplace_u = u_xx_pred + u_yy_pred
    
    # Model parameters (from MATLAB)
    tau_in = 0.3
    tau_out = 6.0
    tau_open = 120.0
    tau_close = 150.0
    v_gate = 0.13
    diff = 0.001  # diffusion coefficient
    
    # Calculate currents (from MATLAB)
    j_in = h_pred * u_pred * u_pred * (1 - u_pred) / tau_in
    j_out = -u_pred / tau_out
    
    # Define residuals (Mitchell-Schaeffer model)
    residual_u = u_t_pred - (j_in + j_out + diff * laplace_u)
    residual_h = h_t_pred - (
        (u_pred < v_gate) * ((1 - h_pred) / tau_open) + 
        (u_pred >= v_gate) * (-h_pred / tau_close)
    )
    
    return residual_u, residual_h


def BC_loss(model, N_bc, times, analytical_solution, nx, ny):
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
    num_points = sampled_xy.size(0)
    
    # Pre-allocate tensors to store analytical values across time steps
    u_analytical_2d = torch.zeros((num_points, len(times), ny, nx))
    v_analytical_2d = torch.zeros((num_points, len(times), ny, nx))

    # Generate inputs for model and analytical solution for all time steps
    sampled_xy_time = []
    for i, input_time in enumerate(times):
        time_column = input_time * torch.ones((num_points, 1), dtype=torch.float32)
        sampled_xy_time.append(torch.cat([sampled_xy.float(), time_column], dim=1))
        
        # Get analytical solution at boundary for this time step
        u_slice, v_slice = analytical_solution(sampled_xy, input_time)
        u_analytical_2d[:, i] = u_slice.unsqueeze(0)
        v_analytical_2d[:, i] = v_slice.unsqueeze(0)

    # Stack the tensor
    sampled_xy_time = torch.cat(sampled_xy_time, dim=0)
    
    # Compute model predictions
    u_pred_bc, v_pred_bc = model(sampled_xy_time)
    
    # Reshape predictions to match analytical data dimensions
    u_pred_bc = u_pred_bc.view(num_points, len(times), 1, 1)
    v_pred_bc = v_pred_bc.view(num_points, len(times), 1, 1)

    # Define the threshold
    threshold = 0.05  # Example threshold

    # Compute the differences
    diff_u_bc = u_pred_bc - u_analytical_2d
    diff_v_bc = v_pred_bc - v_analytical_2d

    # Create masks for differences below the threshold
    mask_u_bc = (diff_u_bc.abs() < threshold).float()
    mask_v_bc = (diff_v_bc.abs() < threshold).float()

    # Apply masks to the loss computation
    loss_bc = torch.sqrt(torch.mean(
        mask_u_bc * (diff_u_bc ** 2) + mask_v_bc * (diff_v_bc ** 2)
    ))


    # Average loss across time steps
    return loss_bc

# Example usage:
# Define your model, analytical_solution, nx, ny, etc.



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
def residual_loss(model, res_tensor):
    residual_value_u, residual_value_v = residual(model, res_tensor)
    loss_residual_u = torch.mean(residual_value_u ** 2)
    loss_residual_v = torch.mean(residual_value_v ** 2)
    return torch.sqrt(loss_residual_u + loss_residual_v)


def PDE_loss(model, N_analytical, times, analytical_solution, nx, ny):
    threshold = 0.1
    total_loss_PDE = 0

    # Define spatial coordinates
    x_spatial = torch.linspace(0, x_end, nx)
    y_spatial = torch.linspace(0, y_end, ny)
    x_mesh, y_mesh = torch.meshgrid(x_spatial, y_spatial, indexing='ij')

    # Flatten the grids
    xy = torch.stack([x_mesh.flatten(), y_mesh.flatten()], dim=-1)

    # Randomly sample N_analytical points from the flattened grid
    indices = torch.randperm(xy.size(0))[:N_analytical**2]
    sampled_xy = xy[indices]
    
    # Print the shape of sampled_xy
    
    num_points = sampled_xy.size(0)
    u_analytical_2d = torch.zeros((num_points, len(times), nx, ny))
    v_analytical_2d = torch.zeros((num_points, len(times), nx, ny))

    sampled_xy_time = []
    for i, input_time in enumerate(times):
        time_column = input_time * torch.ones((num_points, 1))
        sampled_xy_time.append(torch.cat([sampled_xy, time_column], dim=1))
        
        # Calculate analytical solution for this time step
        u_slice, v_slice = analytical_solution(sampled_xy, input_time)
        u_analytical_2d[:, i, :, :] = u_slice
        v_analytical_2d[:, i, :, :] = v_slice

    # Stack sampled_xy_time into a single tensor
    sampled_xy_time = torch.cat(sampled_xy_time, dim=0)

    # Get predictions for all time steps at once
    u_pred_analytical, v_pred_analytical = model(sampled_xy_time)

    # Reshape predictions to match analytical data dimensions
    u_pred_analytical = u_pred_analytical.view(num_points, len(times), 1, 1)
    v_pred_analytical = v_pred_analytical.view(num_points, len(times), 1, 1)

    # Calculate the PDE loss without flattening
    # Compute the differences
    diff_u = u_pred_analytical - u_analytical_2d
    diff_v = v_pred_analytical - v_analytical_2d

    # Create masks for differences below the threshold
    mask_u = (diff_u.abs() > threshold).float()
    mask_v = (diff_v.abs() > threshold).float()

    # Apply masks to the loss computation
    loss_PDE = torch.sqrt(torch.mean(
        mask_u * (diff_u ** 2) + mask_v * (diff_v ** 2)
    ))
    # Average loss across time steps
    return loss_PDE

def loss(model, x_ic, x_res, N_analytical, epoch_max, times, tolerance=1e-2):
    """
    Updated loss function to save loss values and U, V predictions at specific times.
    """
    #import time
    #start_time = time.time()
    # Choose Adam optimizer w/ set learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # Open files for saving data
    with open("loss_epoch.txt", "w") as loss_file, open("output_uv.txt", "w") as uv_file:
        uv_file.write("Time, U_values, V_values\n")  # Header for UV file

        x_ic_tensor = x_ic.clone().detach()
        res_tensor = x_res.clone().detach()
        
        loss_PDE_weight = 1.0
        loss_ic_weight = 1.0
        
        for epoch in range(epoch_max):
            optimizer.zero_grad()

            loss_ic = IC_loss(model, x_ic_tensor)
            loss_residual = residual_loss(model, res_tensor)
            loss_PDE = PDE_loss(model, N_analytical, times, analytical_solution, nx, ny)
            loss_bc = BC_loss(model, N_bc, times, analytical_solution, nx, ny)
            
            if loss_PDE > loss_ic:
                loss_PDE_weight *= 0.9  # Decrease weight of PDE loss if it's dominating
                loss_ic_weight *= 1.1   # Increase weight of IC loss if PDE is too large
            else:
                loss_PDE_weight *= 1.1  # Decrease weight of PDE loss if it's dominating
                loss_ic_weight *= 0.9   # Increase weight of IC loss if PDE is too large
                
                        
            # Total loss with weighted contributions
            # 3 * loss_ic
            loss_tot =  1.9 * loss_residual + loss_PDE + loss_bc
            loss_tot.backward()

            # Prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(loss_tot)

            # Log progress every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss IC: {loss_ic.item()}, Loss BC: {loss_bc.item()}, "
                      f"Loss Residual: {loss_residual.item()}, Loss PDE: {loss_PDE.item()}")

                # Save loss data to file
                loss_file.write(
                    f"{epoch}, {loss_ic.item()}, {loss_bc.item()}, {loss_residual.item()}, {loss_PDE.item()}, {loss_tot.item()}\n"
                )
            
            # Stop if loss is below tolerance
            if loss_tot < tolerance:
                break

        # Open separate files for Time, U, and V
        with open("time.txt", "w") as time_file, \
            open("u_pred.txt", "w") as u_file, \
            open("v_pred.txt", "w") as v_file:

            # Iterate over times and save predictions
            for time in times:
                x = torch.linspace(0, x_end, nx)
                y = torch.linspace(0, y_end, ny)
                x_mesh, y_mesh = torch.meshgrid(x, y)
                xy = torch.stack([x_mesh.flatten(), y_mesh.flatten(), time * torch.ones_like(x_mesh.flatten())], dim=1)
                u_pred, v_pred = model(xy.to(device))
                u_pred = u_pred.detach().cpu().numpy().flatten()
                v_pred = v_pred.detach().cpu().numpy().flatten()

                # Save time, U, and V to their respective files
                time_file.write(f"{time}\n")
                u_file.write(",".join(map(str, u_pred)) + "\n")
                v_file.write(",".join(map(str, v_pred)) + "\n")

    
    #end_time = time.time()
    #print(f"Training completed in {end_time - start_time:.2f} seconds.")
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
       
        u_pred, v_pred = model(xy.to(device))
        u_pred = u_pred.detach().cpu().numpy().reshape(nx, ny)
        v_pred = v_pred.detach().cpu().numpy().reshape(nx, ny)
        
        # Get analytical solution for u at the given time
        u_analytical, v_analytical = analytical_solution(xy, time)
        u_analytical = u_analytical.detach().numpy().reshape(nx, ny)
        
        # Create figure to compare u_pred and u_analytical
        fig, (ax_u_pred, ax_u_analytical) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot U_pred
        norm_u = plt.Normalize(vmin=u_pred.min(), vmax=u_pred.max())
        im_u_pred = ax_u_pred.imshow(u_pred, cmap='plasma', norm=norm_u, origin='lower', extent=[0, 1, 0, 1])
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

        
# Our main code block
model = PINN(NeuronCount)
model = model.to(device)
plot_transient_2d(model, times, N_res, x_ic, x_res, N_analytical, epoch_max)