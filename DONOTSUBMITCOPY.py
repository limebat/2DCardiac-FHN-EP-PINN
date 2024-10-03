import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import time

class PINN(nn.Module):
    def __init__(self, NeuronCount):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(NeuronCount) - 1):
            self.layers.append(nn.Linear(NeuronCount[i], NeuronCount[i+1]))
        
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.tanh(self.layers[i](x))  # TANH HIDDEN LAYERS
        x = self.layers[-1](x)  # FINAL LAYER, NO ACTIVATION
        return x[:, 0], x[:, 1]  # Return two outputs: u and v
        
def residual(model, input, constants):
    input_tensor = input.clone().detach().requires_grad_(True)
    x_tensor = input_tensor[:, 0]
    y_tensor = input_tensor[:, 1]
    t_tensor = input_tensor[:, 2]
    
    a, beta, gamma, eps, delta, dx, dt = constants
    
    u_pred, v_pred = model(input_tensor)
    
    u_t_pred = torch.autograd.grad(u_pred.sum(), t_tensor, create_graph=True, allow_unused=True)[0]
    v_t_pred = torch.autograd.grad(v_pred.sum(), t_tensor, create_graph=True, allow_unused=True)[0]
    
    # Handle cases where gradients might be None
    u_t_pred = torch.zeros_like(u_pred) if u_t_pred is None else u_t_pred
    v_t_pred = torch.zeros_like(v_pred) if v_t_pred is None else v_t_pred
    
    residual_u = u_t_pred - (u_pred * (1 - u_pred) * (u_pred - a) - u_pred * v_pred)
    residual_v = v_t_pred - eps * (beta * u_pred - gamma * v_pred - delta)
    
    return residual_u, residual_v

def analytical_solution(input, constants):
    # Assuming this function fetches analytical solutions
    # For now, return a dummy solution
    return torch.zeros_like(input[:, 0])

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

def PDE_loss(model, N_analytical, constants):
    x_spatial = torch.linspace(0, 1, N_analytical).reshape((-1, 1))
    t_analytical = torch.linspace(0, 1, N_analytical).reshape((-1, 1))
    zeros = torch.zeros_like(x_spatial)
    xt = torch.cat((x_spatial, zeros, t_analytical), dim=1)
    
    u_pred_analytical, _ = model(xt)
    u_analytical = analytical_solution(xt, constants)
    
    loss_PDE = torch.mean((u_pred_analytical - u_analytical) ** 2)
    return loss_PDE

def loss(model, x_ic, x_res, N_analytical, epoch_max, constants):
    start_time = time.time()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epoch_max):
        optimizer.zero_grad()
        x_ic_tensor = torch.tensor(x_ic, dtype=torch.float32)
        x_res_tensor = torch.tensor(x_res, dtype=torch.float32)
        
        loss_ic = IC_loss(model, x_ic_tensor, x_ic)
        loss_residual = residual_loss(model, x_res_tensor, constants)
        loss_PDE = PDE_loss(model, N_analytical, constants)
        loss_tot = loss_ic + loss_residual + loss_PDE        
        loss_tot.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss IC: {loss_ic.item()}, Loss Residual: {loss_residual.item()}, Loss PDE: {loss_PDE.item()}")
    
    end_time = time.time()
    print(f"Total time is now: {end_time - start_time} seconds")

    return model

def plot(model, N_res, N_analytical):
    x = torch.linspace(0, 1, 100).reshape((-1, 1))
    t = torch.linspace(0, 1, 100).reshape((-1, 1))
    zeros = torch.zeros_like(x)
    xt = torch.cat((x, zeros, t), dim=1)
    u_plot, _ = model(xt)
    u_plot = u_plot.detach().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.plot(x.numpy(), u_plot, label='PINN Prediction', color='blue')
    plt.xlabel('x [m]')
    plt.ylabel('U [m/s]')
    plt.title(f'PINN Prediction of U(x,t) at various t, N={N_res}, M={N_analytical}')
    plt.grid(True)
    plt.legend()
    plt.show()
    
# Constants and other initializations
a = 0.1
gamma = 1
beta = 0.5
eps = 0.01
delta = 0.
dx = 0.5
dt = 0.1

constants = [a, beta, gamma, eps, delta, dx, dt]

NeuronCount = [3, 20, 20, 2]  # Input dimension is 3 (x, y, t)
N_ic, N_res, N_analytical = 10, 1000, 100  # Increased number of points
epoch_max = 10000  # Increased number of epochs
model = PINN(NeuronCount)

# Generate initial and residual points including time
x_ic = np.array([[x, 0.0, 0.0] for x in np.linspace(0, 1, N_ic)], dtype=np.float32)  # ICs at t=0
x_res = np.random.rand(N_res, 3).astype(np.float32)  # Random residual points in space and time

model = loss(model, x_ic, x_res, N_analytical, epoch_max, constants)

plot(model, N_res, N_analytical)