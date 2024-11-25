import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class SimplePINN(nn.Module):
    def __init__(self, input_dim=3, hidden_layers=[32, 32], output_dim=2):
        super(SimplePINN, self).__init__()
        layers = []
        layer_sizes = [input_dim] + hidden_layers + [output_dim]
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def load_data(filename='TimeVH.txt'):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    return data

def compute_physics_loss(model, input_tensor):
    # Simplified physics constraint calculation
    x = input_tensor[:, 0]
    t = input_tensor[:, 2]
    
    # Compute model predictions
    predictions = model(input_tensor)
    
    # Simple physics constraint (example placeholder)
    physics_constraint = torch.mean((predictions[:, 0] - torch.sin(x * t)) ** 2)
    
    return physics_constraint

def train_pinn(model, epochs=4000, learning_rate=0.01):
    # Generate synthetic training data
    x = torch.linspace(0, 10, 100).unsqueeze(1)
    t = torch.linspace(0, 5, 100).unsqueeze(1)
    x_grid, t_grid = torch.meshgrid(x.squeeze(), t.squeeze())
    
    input_tensor = torch.stack([
        x_grid.flatten(), 
        t_grid.flatten(), 
        torch.zeros_like(x_grid.flatten())  # Placeholder for third dimension
    ], dim=1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute physics-informed loss
        physics_loss = compute_physics_loss(model, input_tensor)
        physics_loss.backward()
        
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Physics Loss: {physics_loss.item()}')
    
    return model

def visualize_results(model):
    # Create grid for visualization
    x = torch.linspace(0, 10, 100)
    t = torch.linspace(0, 5, 100)
    x_grid, t_grid = torch.meshgrid(x, t)
    
    input_tensor = torch.stack([
        x_grid.flatten(), 
        t_grid.flatten(), 
        torch.zeros_like(x_grid.flatten())
    ], dim=1)
    
    with torch.no_grad():
        predictions = model(input_tensor)
    
    # Reshape predictions
    u_pred = predictions[:, 0].numpy().reshape(x_grid.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(x_grid.numpy(), t_grid.numpy(), u_pred, cmap='viridis')
    plt.colorbar(label='Predicted Value')
    plt.title('PINN Predictions')
    plt.xlabel('Spatial Coordinate')
    plt.ylabel('Time')
    plt.show()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Initialize model
    model = SimplePINN()
    
    # Train model
    trained_model = train_pinn(model)
    
    # Visualize results
    visualize_results(trained_model)

if __name__ == '__main__':
    main()