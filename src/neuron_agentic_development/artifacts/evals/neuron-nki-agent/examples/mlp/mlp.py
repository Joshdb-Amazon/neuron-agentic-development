import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    SwiGLU MLP as used in LLaMA architecture.
    Performs: down_proj(silu(gate_proj(x)) * up_proj(x))
    """
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, input_size, bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, input_size).
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate) * up  # SwiGLU activation
        output = self.down_proj(hidden)
        return output


batch_size = 1024
seq_len = 2048
input_size = 4096
hidden_size = 8192  # Typical LLaMA expansion ratio (~2.7x), make it a power of 2 number for tiling friendly

def get_inputs():
    return [torch.rand(batch_size, seq_len, input_size)]

def get_init_inputs():
    return [input_size, hidden_size]
