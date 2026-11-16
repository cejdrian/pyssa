# script to replace structs
from dataclasses import dataclass

@dataclass
class PssaConfig:
    vocab_size: int = 50257
    d_model: int = 512
    context_len: int = 1024
    learning_rate: float = 3e-4 #this value is graphed, so no int
    plastic_eta: float = 0.01
    n_layers: int = 12
    #more params soon maybe