import numpy as np
import matplotlib.pyplot as plt

# Mitchell-Schaeffer model in 2D
# from 2003 Bulletin of Mathematical Biology

# parameter values
tau_in = 0.3
tau_out = 6
tau_open = 120
tau_close = 150
v_gate = 0.13
v_stim = 0.056  # twice diastolic threshold for 2ms duration
convolution_factor = 2

# numerical and stimulation parameters
dt = 0.2
endtime = 350
dt_python = 25  # The time increment between lines printed to TimeVH.txt
nsteps = int(np.ceil(endtime / dt))
stimdur = 2
nstimdur = int(np.ceil(stimdur / dt))
spiraltime = 250
nspiraltime = int(np.ceil(spiraltime / dt))
dx = 0.05 * convolution_factor  # *5
diff = 0.001  # diffusion coefficient
nx = spiraltime // convolution_factor
ny = nx
dt_o_dx2 = dt / (dx * dx)

# initial values for state variables
v = np.zeros((nx, ny))
h = np.ones((nx, ny)) * 0.5

# Open files for saving the data
fileID = open('TimeVH.txt', 'w')
fileIDMESH = open('XY.txt', 'w')

# Write headers to the files
fileID.write('sec, V, H\n')
fileIDMESH.write('X, Y\n')

# Begin recording time and their iterations
record_iterations_begin = spiraltime - dt_python
t = np.arange(0, endtime + dt, dt)
xx = np.arange(1, nx + 1) * dx

# Time loop
for ntime in range(nsteps):
    # Apply stimulus if it's time
    if ntime == 1:
        v[:, :10] = 0.5
    if ntime == nspiraltime:
        v[:nx//2, :] = 0

    # Calculate currents
    jin = h * v * v * (1 - v) / tau_in
    jout = -v / tau_out
    
    # Update derivatives for state variables
    dv = jin + jout
    dh = (v < v_gate) * ((1 - h) / tau_open) + (v >= v_gate) * (-h / tau_close)

    xlap = np.zeros((nx, ny))
    for j in range(ny):
        for i in range(nx):
            if i == 0:
                xlap1 = 2 * (v[1, j] - v[0, j])
            elif i == nx - 1:
                xlap1 = 2 * (v[nx-1, j] - v[nx-2, j])
            else:
                xlap1 = v[i-1, j] - 2 * v[i, j] + v[i+1, j]
            
            if j == 0:
                xlap2 = 2 * (v[i, 1] - v[i, 0])
            elif j == ny - 1:
                xlap2 = 2 * (v[i, ny-1] - v[i, ny-2])
            else:
                xlap2 = v[i, j-1] - 2 * v[i, j] + v[i, j+1]

            xlap[i, j] = xlap1 + xlap2

    xlap *= diff * dt_o_dx2

    # Integrate using forward Euler method
    v = v + dt * dv + xlap
    h = h + dt * dh

    # Save data at specified intervals
    if np.mod(ntime * dt, dt_python) == 0 and ntime * dt > record_iterations_begin:
        plt.pcolor(v)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar()
        plt.title(f"time = {ntime * dt}")
        plt.draw()
        plt.pause(0.01)

        # Flatten and save to file in row-major order
        v_flatten = v.T.flatten()  # Transpose v to ensure row-major order
        h_flatten = h.T.flatten()  # Transpose h to ensure row-major order
        
        data_row = np.concatenate(([ntime * dt], v_flatten, h_flatten))

        # Print the data row without an extra comma at the end
        fileID.write(f'{data_row[0]}')
        for j in range(1, len(data_row)):
            fileID.write(f', {data_row[j]}')
        fileID.write('\n')  # Move to the next line

# Create meshgrid for X, Y and save it
x, y = np.meshgrid(xx, xx)
mesh = np.column_stack((x.flatten(), y.flatten()))

# Write mesh data to file
np.savetxt(fileIDMESH, mesh, delimiter=', ', fmt='%g')

# Close the files
fileID.close()
fileIDMESH.close()

print("Data saved to TimeVH.txt")

