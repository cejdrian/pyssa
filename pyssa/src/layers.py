from .config import PssaConfig
import torch
import torch.nn as nn

class PssaStateSpaceCell(nn.Module):
    #this is a foundational block from PSSA
    #Processes data sequentially rather than having a massive
    #attention matrix, just like the original PSSA.
    def __init__(self, config: PssaConfig):
        super().__init__()
        self.d_model = config.d_model

        #linear projection matrices for inputs
        self.in_proj = nn.Linear(self.d_model, self.d_model * 2, bias = False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias = False)

        #state transition param (memory scale of it)
        self.A_decay = nn.Parameter(torch.ones(self.d_model) * -0.5)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor = None):
        batch_size, seq_len, _ = x.shape
        
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.d_model, device=x.device)
        
        outputs = []
        A = torch.exp(self.A_decay) #safe decay bounds
        
        for t in range(seq_len): #extracts individual time steps
            xt = x[:, t, :]
         
        #map input to gates
        projected = self.in_proj(xt)
        u, gate = torch.chunk(projected, 2, dim=-1)

        #formula 
        hidden_state = A * hidden_state + torch.sigmoid(gate) * u

        #appends
        outputs.append(hidden_state.unsqueeze(1))

        #stack sequential outputs and pass thru final projections to finish and return data
        output_sequence = torch.cat(outputs, dim=-1)
        return self.out_proj(output_sequence), hidden_state
