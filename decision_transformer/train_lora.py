# copied from main_train_time.py
import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_folder)
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from transformers import DecisionTransformerConfig
from decision_transformer.art_lora import ff_Transformer_pred_time_lora
import torch
import decision_transformer.manage_time as TTO_manager
from decision_transformer.manage_time import device
from optimization.ff_scenario_time import chunksize

import loralib as lora
# rsync -avz -e ssh xinmeng@iris-asl.stanford.edu:/home/xinmeng/freeflyer2/ff_control/transformer_controller/dataset/time_const_90 /Users/xin/marco_pavone/freeflyer2-davide/dataset

# check `decision_transformer.manage_time.transformer_import_config` to see how to name the model
# check optimization.ff_scenario_time for environment setup
model_name_4_saving = 'checkpoint_ff_time_lora_const90'
model_config = TTO_manager.transformer_import_config(model_name_4_saving)

# load datasets
datasets, dataloaders = TTO_manager.get_train_val_test_data(mdp_constr=model_config['mdp_constr'], dataset_scenario=model_config['dataset_scenario'],
                                                            timestep_norm=model_config['timestep_norm'], chunksize=model_config['chunksize'])
train_loader, eval_loader, test_loader = dataloaders
n_state = train_loader.dataset.n_state
n_data = train_loader.dataset.n_data
n_action = train_loader.dataset.n_action
n_time = train_loader.dataset.max_len

# Transformer parameters
config = DecisionTransformerConfig(
    state_dim=n_state, 
    act_dim=n_action,
    hidden_size=384,
    max_ep_len=n_time,
    vocab_size=1,
    action_tanh=False,
    n_positions=2048,
    n_layer=6,
    n_head=6,
    n_inner=None,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    )

model = ff_Transformer_pred_time_lora(config)
lora.mark_only_lora_as_trainable(model)

model_size = sum(t.numel() for t in model.parameters())
print(f"GPT size: {model_size/1000**2:.1f}M parameters")

# model = model.to(torch.float32) # for mac running on cpu
model.to(device)

# float64 is not supported by the MPS (Metal Performance Shaders) framework, which is used for GPU acceleration on Apple devices.
# model.to(device)

from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
optimizer = AdamW(model.parameters(), lr=3e-5)
accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=8)
# prepare the model, optimizer, and data loaders for distributed training
# train_loader and eval_loader are assumed to be previously defined data loaders
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_loader, eval_loader)

num_train_epochs = 1
num_update_steps_per_epoch = len(train_dataloader) # equal to number of batches
num_training_steps = 10000000000

# Learning Rate Scheduler Setup
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=10, # Number of steps for the learning rate warm-up phase
    num_training_steps=num_training_steps, # Total number of training steps for the scheduler to plan the learning rate decay
)

# To activate only when starting from a pretrained model
# accelerator.load_state(root_folder + '/decision_transformer/saved_files/checkpoints/' + model_name_4_saving)

'''
 Evaluation: Assesses model performance on a validation set without updating model parameters.
    - Uses @torch.no_grad() to disable gradient computation.
    - Uses eval_dataloader.
    - Accumulates losses over multiple batches for averaging.
    - No optimizer steps.

    
Training: Computes gradients for parameter updates.
    - Updates model parameters to improve performance on the training set.
    - Uses train_dataloader
    - Computes loss for each batch and immediately uses it for backpropagation.
    - Performs optimizer steps to update model parameters.
'''

# Eval function
eval_iters = 100
@torch.no_grad() # This decorator ensures that no gradients are calculated during evaluation, which saves memory and computation.

def evaluate():
    model.eval() # Sets the model to evaluation mode, which affects layers like dropout and batch normalization.
    # store different types of losses
    losses = []
    losses_state = []
    losses_action = []
    losses_ttgs = []
    for step in range(eval_iters):
        # Loads a batch of data from the evaluation data loader.
        data_iter = iter(eval_dataloader)
        states_i, actions_i, rtgs_i, ctgs_i, ttgs_i, goal_i, timesteps_i, attention_mask_i, _, _, _ = next(data_iter)
        mask = rtgs_i < 0
        # Creates masks to filter out certain elements based on conditions.
        mask_act = torch.repeat_interleave(mask, 3, 2)
        mask_st = torch.repeat_interleave(mask[:,1:,:], 6, 2)
        # Model Prediction
        with torch.no_grad():
            state_preds, action_preds, ttgs_pred = model(
                states=states_i,
                actions=actions_i,
                goal=goal_i,
                returns_to_go=rtgs_i,
                constraints_to_go=ctgs_i,
                times_to_go=ttgs_i,
                timesteps=timesteps_i,
                attention_mask=attention_mask_i,
                return_dict=False,
            )
        loss_i = torch.mean((action_preds[mask_act] - actions_i[mask_act]) ** 2)#torch.mean((action_preds - actions_i) ** 2)#
        loss_i_state = torch.mean((state_preds[:,:-1,:][mask_st] - states_i[:,1:,:][mask_st]) ** 2)#torch.mean((state_preds[:,:-1,:] - states_i[:,1:,:]) ** 2)#
        loss_i_ttgs = torch.mean((ttgs_pred[mask] - ttgs_i[mask]) ** 2)#torch.mean((ttgs_pred - ttgs_i) ** 2)#
        losses.append(accelerator.gather(loss_i + loss_i_state + loss_i_ttgs))
        losses_state.append(accelerator.gather(loss_i_state))
        losses_action.append(accelerator.gather(loss_i))
        losses_ttgs.append(accelerator.gather(loss_i_ttgs))
    # Average the losses over all evaluation iterations
    loss = torch.mean(torch.tensor(losses))
    loss_state = torch.mean(torch.tensor(losses_state))
    loss_action = torch.mean(torch.tensor(losses_action))
    loss_ttgs = torch.mean(torch.tensor(losses_ttgs))
    model.train()
    return loss.item(), loss_state.item(), loss_action.item(), loss_ttgs.item()

eval_loss, loss_state, loss_action, loss_ttg = evaluate()
accelerator.print({"loss/eval": eval_loss, "loss/state": loss_state, "loss/action": loss_action, "loss/ttg": loss_ttg})


# Training
eval_steps = 500
samples_per_step = accelerator.state.num_processes * train_loader.batch_size
torch.manual_seed(4)

model.train() # set model to training mode
completed_steps = 0
log = {
    'loss':[],
    'loss_state':[],
    'loss_action':[],
    'loss_ttg' : []
}
'''log = np.load(root_folder + '/decision_transformer/saved_files/checkpoints/' + model_name_4_saving + '/log.npz', allow_pickle=True)['log'].item()'''
# Training loop
for epoch in range(num_train_epochs):
    for step, batch in enumerate(train_dataloader, start=0):
        with accelerator.accumulate(model):
            # Unpack batch data
            states_i, actions_i, rtgs_i, ctgs_i, ttgs_i, goal_i, timesteps_i, attention_mask_i, _, _, _ = batch
            mask = rtgs_i < 0
            mask_act = torch.repeat_interleave(mask, 3, 2)
            mask_st = torch.repeat_interleave(mask[:,1:,:], 6, 2)
            state_preds, action_preds, ttgs_pred = model(
                states=states_i,
                actions=actions_i,
                goal=goal_i,
                returns_to_go=rtgs_i,
                constraints_to_go=ctgs_i,
                times_to_go=ttgs_i,
                timesteps=timesteps_i,
                attention_mask=attention_mask_i,
                return_dict=False,
            )
            loss_i_action = torch.mean((action_preds[mask_act] - actions_i[mask_act]) ** 2)#torch.mean((action_preds - actions_i) ** 2)#
            loss_i_state = torch.mean((state_preds[:,:-1,:][mask_st] - states_i[:,1:,:][mask_st]) ** 2)#torch.mean((state_preds[:,:-1,:] - states_i[:,1:,:]) ** 2)#
            loss_i_ttg = torch.mean((ttgs_pred[mask] - ttgs_i[mask]) ** 2)#torch.mean((ttgs_pred - ttgs_i) ** 2)#
            loss = loss_i_action + loss_i_state + loss_i_ttg
            if step % 100 == 0:
                accelerator.print(
                    {
                        "lr": lr_scheduler.get_lr(),
                        "samples": step * samples_per_step,
                        "steps": completed_steps,
                        "loss/train": loss.item(),
                    }
                )
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
            if (step % (eval_steps)) == 0:
                eval_loss, loss_state, loss_action, loss_ttg = evaluate()
                accelerator.print({"loss/eval": eval_loss, "loss/state": loss_state, "loss/action": loss_action, "loss/ttg": loss_ttg})
                log['loss'].append(eval_loss)
                log['loss_state'].append(loss_state)
                log['loss_action'].append(loss_action)
                log['loss_ttg'].append(loss_ttg)
                model.train()
                accelerator.wait_for_everyone()
            if (step % (eval_steps*10)) == 0:
                print('Saving model..')
                accelerator.save_state(root_folder+'/decision_transformer/saved_files/checkpoints/'+model_name_4_saving)
                np.savez_compressed(root_folder + '/decision_transformer/saved_files/checkpoints/' +model_name_4_saving+ '/log',
                            log = log
                            )
                
                # ===== Before =====
# torch.save(model.state_dict(), checkpoint_path)
# ===== After (only save lora weight updates) =====
checkpoint_path = root_folder + '/decision_transformer/saved_files/checkpoints/' +model_name_4_saving+ '/lora_params'
torch.save(lora.lora_state_dict(model), checkpoint_path)