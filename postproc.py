import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Loss plot per epoch
'''


# Load data
data = np.loadtxt("loss_epoch.txt", delimiter=",")
epochs = data[:, 0]
loss_ic = data[:, 1]
loss_bc = data[:, 2]
loss_residual = data[:, 3]
loss_pde = data[:, 4]
loss_tot = data[:, 5]

# Plot the loss per epoch with ylim set to 1
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_ic, label="Loss IC")
plt.plot(epochs, loss_bc, label="Loss BC")
plt.plot(epochs, loss_residual, label="Loss Residual")
plt.plot(epochs, loss_pde, label="Loss PDE")
plt.plot(epochs, loss_tot, label="Total Loss", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.legend()
plt.title("Loss vs. Epoch")
plt.ylim(0, 1)  # Set y-axis limit to 1
plt.grid()
plt.show()

# Log plot of the loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, np.log(loss_ic), label="Log Loss IC")
plt.plot(epochs, np.log(loss_bc), label="Log Loss BC")
plt.plot(epochs, np.log(loss_residual), label="Log Loss Residual")
plt.plot(epochs, np.log(loss_pde), label="Log Loss PDE")
plt.plot(epochs, np.log(loss_tot), label="Log Total Loss", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Log Loss Value")
plt.legend()
plt.title("Log Loss vs. Epoch")
plt.grid()
plt.show()


'''
Below is the U-plots
'''


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
    
    u_analytical = torch.tensor(u_reshaped, dtype=torch.float16).flatten()
    v_analytical = torch.tensor(v_reshaped, dtype=torch.float16).flatten()
    
    sampled_u = u_analytical[:len(input)]
    sampled_v = v_analytical[:len(input)]
    
    
    
    return sampled_u, sampled_v




conv_factor = 5.0         # -
nx = ny = int(250 / conv_factor)  # Grid dimensions
dx = 0.05 * conv_factor # -
dt = 100
time = 700.0

x_end = y_end = nx * dx

x = torch.linspace(0, x_end, nx)
y = torch.linspace(0, y_end, ny)
x_mesh, y_mesh = torch.meshgrid(x, y)

times = np.loadtxt("time.txt")
u_data = np.loadtxt("u_pred.txt", delimiter=",")
v_data = np.loadtxt("v_pred.txt", delimiter=",")

print(times)

print(np.where(times == time))

specific_time_index = int((times[np.where(times == time)] - time) / dt)  # Time index norammlized by dt and subtracted by IC of 250

# Calculate indices for slicing
start_idx = specific_time_index * (nx * ny)
end_idx = start_idx + (nx * ny)

u_snapshot = u_data[specific_time_index].reshape((nx, ny))  # Reshape into a 250x250 grid


# Prepare input grid with current time for the model prediction
xy = torch.stack([x_mesh.flatten(), y_mesh.flatten(), time * torch.ones_like(x_mesh.flatten())], dim=1)

u_analytical, v_analytical = analytical_solution(xy, time)
u_analytical = u_analytical.detach().numpy().reshape(nx, ny)


# Create a figure with two subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot the first contour plot (u_snapshot)
contour1 = ax1.contourf(u_snapshot, cmap="plasma")
fig.colorbar(contour1, ax=ax1, label="U Value")
ax1.set_title(f"Predicted U at Time {specific_time_index*dt+time}")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

# Plot the second contour plot (u_analytical)
contour2 = ax2.contourf(u_analytical, cmap="plasma")
fig.colorbar(contour2, ax=ax2, label="U Value")
ax2.set_title(f"Analytical U at Time {specific_time_index*dt+time}")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")

# Display the plots
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# Extract the U values along the centerline (at y = ny/2)
u_centerline = u_snapshot[ny // 2, :]
u_centerline_analytical = u_analytical[ny // 2, :]

# Plot the centerline U values with respect to x on the same plot
plt.figure(figsize=(8, 6))
plt.plot(np.linspace(0, 1, nx), u_centerline, color='b', label=f"Predicted U at centerline\n(y = {ny//2})")
plt.plot(np.linspace(0, 1, nx), u_centerline_analytical, color='r', label=f"Analytical U at centerline\n(y = {ny//2})")
plt.title(f"Centerline U at Time {specific_time_index*dt+time}")
plt.xlabel("X")
plt.ylabel("U Value")
plt.legend(loc='upper right')
plt.ylim(0, 1)  # Set y-axis limit to 1
plt.grid(True)
plt.show()
