from torch import nn
import torch

class ReLU_Scaler(nn.Module):
    def __init__(self, input_size, energy_init=1.0):
        """
        Initialize the EnergyScaledLayer.
        
        Parameters:
        - input_size (int): Number of input neurons (size of the input tensor).
        - energy_init (float): Initial value for the energy scaling factor.
        """
        super(ReLU_Scaler, self).__init__()
        
        # Initialize energy scaling factors, one for each input neuron
        self.energy = nn.Parameter(torch.full((input_size,), energy_init))

    def forward(self, x):
        """
        Forward pass through the EnergyScaledLayer.
        
        Parameters:
        - x (torch.Tensor): Input tensor with activations from the previous layer.
        
        Returns:
        - torch.Tensor: Output tensor with activations scaled by the energy factors.
        """
        # Element-wise multiplication of input activations with energy scaling factors
        return x * self.energy
    def update_energy(self, new_energy):
        with torch.no_grad():
            clipped_energy = torch.clamp(new_energy, min=0.7, max=1.5)
            self.energy.copy_(clipped_energy)
