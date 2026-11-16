from src import PssaConfig, PssaWorldModel
import torch

#init structures
config = PssaConfig(vocab_size=1000, d_model=512, n_layers=2) #quick test
model = PssaWorldModel(config)

#here this might get complicated, but basically I'll mock
#a synthetic tensor batch of inputs. That'll be the main idea 
#for any kind of test/experiment I run here.
#(batch size of 2 lines of text, 5 token-long context, and 256 for Vector Dimension)

dummy_txt_input = torch.randint(0, config.vocab_size, (2, 8))
logits, final_states = model(dummy_txt_input)

print("pyssa archit. assembled")
print(f"token input shape: {dummy_txt_input.shape} (matches vocab bound: {logits.shape[-1]})")
print(f"logits output shape: {logits.shape}")
print(f"layer state cache: {len(final_states)} active layers tracked")
