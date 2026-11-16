#imports
import torch
import torch.nn as nn
from .config import PssaConfig
from .layers import PssaStateSpaceCell

class PssaWorldModel(nn.Module):
    #A rough logic remap using torch for this main component, vers. 1
    #This should pretty much store, or house input token embeddings to then
    #route through layers and finally project states to what we'll
    #call vocab. logits
    def __init__(self, config: PssaConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        #list of layers
        self.layers = nn.ModuleList([
            PssaStateSpaceCell(config) for _ in range(config.n_layers)
        ])

        #projection setup
        self.line_head = nn.Linear(self.d_model, self.vocab_size, bias = False)

    def forward(self, input_ids: torch.Tensor, hidden_states: list = None):
        #convert raw text token ids into vectors
        x = self.token_embedding(input_ids)
        #track in case
        nxt_hidden_states = []

        #loop sequentially thru stack
        for i, layer in enumerate(self.layers):
            #extract
            layer_hidden = hidden_states[i] if hidden_states is not None else None
            #pass vectors thru loop math
            x, new_hidden = layer(x, hidden_state=layer_hidden)
            #store updated state
            nxt_hidden_states.append(new_hidden)

        #finally project it 
        logits = self.line_head(x)
        return logits, nxt_hidden_states