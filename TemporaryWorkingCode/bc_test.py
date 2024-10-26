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
def sample_analytical_solution(boundary_tensor):
    """
    Sample the analytical solution at the given boundary points.
    
    Args:
        boundary_tensor: A tensor containing boundary points with shape (N, 3) for [x, y, t]
    
    Returns:
        Tuple of tensors (u_analytical, v_analytical) containing the solutions at boundary points
    """
    # Convert input tensor to numpy for processing
    x_points = boundary_tensor[:, 0].numpy()
    y_points = boundary_tensor[:, 1].numpy()
    time_points = boundary_tensor[:, 2].numpy()
    
    # Read the full analytical solution data
    file_path = 'TimeVH.txt'
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    
    nx, ny = 250, 250
    total_points = nx * ny
    time_column = data[:, 0]
    
    # Initialize arrays for results
    u_analytical = np.zeros(len(x_points))
    v_analytical = np.zeros(len(x_points))
    
    # Process each boundary point
    for i in range(len(x_points)):
        # Find the closest time in our data
        time_idx = np.abs(time_column - time_points[i]).argmin()
        
        # Get the u and v solutions for this timestep
        u_field = data[time_idx, 1:total_points+1].reshape(nx, ny)
        v_field = data[time_idx, total_points+1:].reshape(nx, ny)
        
        # Convert x, y coordinates to grid indices
        x_idx = int(np.clip(x_points[i] * (nx - 1), 0, nx - 1))
        y_idx = int(np.clip(y_points[i] * (ny - 1), 0, ny - 1))
        
        # Sample the solution at this point
        u_analytical[i] = u_field[x_idx, y_idx]
        v_analytical[i] = v_field[x_idx, y_idx]
    
    return torch.tensor(u_analytical, dtype=torch.float32), torch.tensor(v_analytical, dtype=torch.float32)

def BC_loss(model, boundary_tensor, times, constants):
    """
    Calculate the boundary condition loss using the analytical solution.
    
    Args:
        model: The PINN model
        boundary_tensor: Tensor of boundary points (N, 2) for [x, y]
        times: Tensor of time points to evaluate
        constants: List of physical constants for the problem
    
    Returns:
        Total boundary condition loss
    """
    total_bc_loss = 0.0
    
    # Get model predictions and analytical solutions for each time point
    for t in times:
        # Create full input tensor including time
        time_tensor = torch.full((boundary_tensor.shape[0], 1), t)
        input_tensor = torch.cat((boundary_tensor, time_tensor), dim=1)
        
        # Get model predictions
        u_pred, v_pred = model(input_tensor)
        
        # Get analytical solutions
        u_true, v_true = sample_analytical_solution(input_tensor)
        
        # Calculate MSE loss for this time point
        loss_t = torch.mean((u_pred - u_true)**2 + (v_pred - v_true)**2)
        total_bc_loss += loss_t
    
    # Average loss across all time points
    return total_bc_loss / len(times)
    
def generate_boundary_conditions(num_points_per_wall):
    """
    Generate boundary points for a unit square domain.
    """
    # Generate points along each wall
    x = torch.linspace(0., 1., num_points_per_wall)
    y = torch.linspace(0., 1., num_points_per_wall)
    
    boundary_points = []
    
    # Bottom wall (y=0)
    boundary_points.extend([[x_i.item(), 0.] for x_i in x])
    
    # Top wall (y=1)
    boundary_points.extend([[x_i.item(), 1.] for x_i in x])
    
    # Left wall (x=0)
    boundary_points.extend([[0., y_i.item()] for y_i in y])
    
    # Right wall (x=1)
    boundary_points.extend([[1., y_i.item()] for y_i in y])
    
    return torch.tensor(boundary_points, dtype=torch.float32)

def loss(model, x_ic, x_res, N_analytical, epoch_max, constants, times, num_boundary_points):
    """
    Modified loss function with boundary conditions
    """
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Generate boundary points once at the start
    boundary_points = generate_boundary_conditions(num_boundary_points)
    
    for epoch in range(epoch_max):
        optimizer.zero_grad()
        x_res_tensor = torch.tensor(x_res, dtype=torch.float32)
        
        # Calculate all loss components
        loss_bc = BC_loss(model, boundary_points, times, constants)
        loss_residual = residual_loss(model, x_res_tensor, constants)
        loss_PDE = PDE_loss(model, N_analytical, constants, times)
        
        # Weighted sum of losses
        loss_tot = loss_bc + loss_residual + loss_PDE
        
        loss_tot.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss BC: {loss_bc.item():.6f}, Loss Residual: {loss_residual.item():.6f}, Loss PDE: {loss_PDE.item():.6f}")
    
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")
    return model

def residual_loss(model, x_res_tensor, constants):
    residual_value_u, residual_value_v = residual(model, x_res_tensor, constants)
    loss_residual_u = torch.mean(residual_value_u ** 2)
    loss_residual_v = torch.mean(residual_value_v ** 2)
    return torch.sqrt(loss_residual_u + loss_residual_v)  # TODO RMSE or MSE?

def PDE_loss(model, N_analytical, constants, times):
    total_loss_PDE = 0
    for input_time in times:
        x_spatial = torch.linspace(0, 1, N_analytical).reshape((-1, 1))
        y_spatial = torch.linspace(0, 1, N_analytical).reshape((-1, 1))
        x_mesh, y_mesh = torch.meshgrid(x_spatial.squeeze(), y_spatial.squeeze())
        xy = torch.stack([x_mesh.flatten(), y_mesh.flatten(), input_time * torch.ones_like(x_mesh.flatten())], dim=1)

        u_pred_analytical, v_pred_analytical = model(xy)
        u_analytical, v_analytical = analytical_solution(xy, constants, input_time)

        loss_PDE = torch.sqrt(torch.mean((u_pred_analytical - u_analytical) ** 2 + (v_pred_analytical - v_analytical) ** 2))  # RMSE
        total_loss_PDE += loss_PDE
    
    # Average loss across time steps
    return total_loss_PDE / len(times)

def plot_transient_2d(model, time_steps, constants, N_res=250):
    x = torch.linspace(0, 1, N_res)
    y = torch.linspace(0, 1, N_res)
    x_mesh, y_mesh = torch.meshgrid(x, y)

    for time in time_steps:
        # Prepare input grid with current time for the model prediction
        xy = torch.stack([x_mesh.flatten(), y_mesh.flatten(), time * torch.ones_like(x_mesh.flatten())], dim=1)
        u_pred, v_pred = model(xy)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f'u at t = {time:.2f}')
        plt.contourf(x_mesh.numpy(), y_mesh.numpy(), u_pred.detach().numpy().reshape(N_res, N_res), levels=20)
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title(f'v at t = {time:.2f}')
        plt.contourf(x_mesh.numpy(), y_mesh.numpy(), v_pred.detach().numpy().reshape(N_res, N_res), levels=20)
        plt.colorbar()

        plt.show()
# Main execution code
if __name__ == "__main__":
    # Define constants for the model
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

    times = torch.arange(0, 1, dt)

    # Initialize x_ic and x_res uniformly
    x_ic = np.zeros((N_ic, 3), dtype=np.float32)
    x_ic[:, 2] = 0  # Set time to 0 for initial conditions

    x_res = np.linspace(0, 1, N_res).reshape(-1, 1)
    x_res = np.tile(x_res, (1, 2))  # Duplicate the x column for y
    x_res = np.hstack((x_res, np.random.rand(x_res.shape[0], 1)))  # Add random time values

    # Define number of boundary points and train model
    num_boundary_points = 50
    model = loss(model, x_ic, x_res, N_analytical, epoch_max, constants, times, num_boundary_points)

    # Plot results
    plot_transient_2d(model, times.numpy(), constants, N_res=250)
