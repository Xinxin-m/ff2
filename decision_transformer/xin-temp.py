import torch
from accelerate import Accelerator
'''
model_name = 'checkpoint_ff_time_90_random'
state_path = '/home/xinmeng/freeflyer2/ff_control/transformer_controller/decision_transformer/saved_files/checkpoints/checkpoint_ff_time_90_random/pytorch_model.bin' #/home/xinmeng/freeflyer2/ff_control/transformer_controller/saved_files/checkpoints/' + model_name
state_dict = torch.load(state_path)
print(state_dict.keys(), "\n")

# Load the optimizer state dictionary
optimizer_state = torch.load('/home/xinmeng/freeflyer2/ff_control/transformer_controller/decision_transformer/saved_files/checkpoints/checkpoint_ff_time_90_random/optimizer.bin')
print(optimizer_state.keys()) # return> dict_keys(['state', 'param_groups'])
'''
accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=8)
accelerator.load_state('/home/xinmeng/freeflyer2/ff_control/transformer_controller/decision_transformer/saved_files/checkpoints/checkpoint_ff_ctgrtg')
# accelerator.load_state('/home/xinmeng/freeflyer2/ff_control/transformer_controller/decision_transformer/saved_files/checkpoints/checkpoint_ff_time_90_random')