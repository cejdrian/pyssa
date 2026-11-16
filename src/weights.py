import torch
import torch.nn as nn
from .config import PssaConfig

class PssaPlasticUpdateEngine(nn.Module):
    #v1 plastic weights
    def __init__(self, config: PssaConfig):
        super().__init__()
        self.d_model = config.d_model
        self.eta = config.plastic_eta
    def computePlasticStep(self, xt: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        #batch_size, d_model, return output
        
        #correlate memory state (current) with incoming token vector
        delta_w = torch.bmm(hidden_state.unsqueeze(-1), xt.unsqueeze(1))

        #scale by the plasticity learn rate
        return self.eta * delta_w