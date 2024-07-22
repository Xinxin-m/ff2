
# import the same files as for art.py
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.decision_transformer.configuration_decision_transformer import DecisionTransformerConfig
from transformers.models.decision_transformer.modeling_decision_transformer import DecisionTransformerPreTrainedModel, DecisionTransformerGPT2Model, DecisionTransformerOutput

''' To-Do
    - should lora be applied to all linear layers, or only layers within the encoders and not the final prediction?

'''
root_folder = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_folder)
from loralib import Linear as loraLinear

class ff_Transformer_lora(DecisionTransformerPreTrainedModel):
    # without time-to-go 
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.encoder = DecisionTransformerGPT2Model(config)

        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size) # positional encoding
        self.embed_goal = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        self.embed_constraint = torch.nn.Linear(1, config.hidden_size)
        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_action = torch.nn.Linear(config.act_dim, config.hidden_size)

        self.embed_ln = nn.LayerNorm(config.hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(config.hidden_size, config.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(config.hidden_size, config.act_dim)] + ([nn.Tanh()] if config.action_tanh else []))
        )

        # Initialize weights and apply final processing
        self.post_init()

    def __init__(self, config, r=16, alpha=1, p=0.1):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        # lora parameters
        self.lora_r = r
        self.lora_alpha = alpha
        self.lora_dropout = p

        # Use the original GPT2 model
        self.encoder = DecisionTransformerGPT2Model(config)

        # Replace linear layers with lora layers
        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_goal = loraLinear(config.state_dim, config.hidden_size, r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout)
        self.embed_return = loraLinear(1, config.hidden_size, r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout)
        self.embed_constraint = loraLinear(1, config.hidden_size, r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout)
        self.embed_state = loraLinear(config.state_dim, config.hidden_size, r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout)
        self.embed_action = loraLinear(config.act_dim, config.hidden_size, r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout)

        self.embed_ln = nn.LayerNorm(config.hidden_size)

        self.predict_state = loraLinear(config.hidden_size, config.state_dim, r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout)
        
        # For the action prediction, we'll apply lora to the first linear layer
        self.predict_action = nn.Sequential(
            loraLinear(config.hidden_size, config.act_dim, r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout),
            *([] if not config.action_tanh else [nn.Tanh()])
        )

        # Initialize weights and apply final processing
        self.post_init()

    # The forward method remains the same as in the original class

    # @add_start_docstrings_to_model_forward(DECISION_TRANSFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @replace_return_docstrings(output_type=DecisionTransformerOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        states: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        goal: Optional[torch.FloatTensor] = None,
        returns_to_go: Optional[torch.FloatTensor] = None,
        constraints_to_go: Optional[torch.FloatTensor] = None,
        timesteps: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], DecisionTransformerOutput]:
        

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        goal_embeddings = self.embed_goal(goal)
        returns_embeddings = self.embed_return(returns_to_go)
        constraints_embeddings = self.embed_constraint(constraints_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        goal_embeddings = goal_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        constraints_embeddings = constraints_embeddings + time_embeddings

        # this makes the sequence look like (T_1, R_1, C_1 s_1, a_1, T_2, R_2, C_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((goal_embeddings, returns_embeddings, constraints_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 5 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 5 * seq_length)
        )
        device = stacked_inputs.device
        # we feed in the input embeddings (not word indices as in NLP) to the model
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(stacked_attention_mask.shape, device=device, dtype=torch.long),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = encoder_outputs[0]

        # reshape x so that the second dimension corresponds to the original
        # goal (0), return_to_go (1), constraint_to_go (2), states (3), or actions (4); i.e. x[:,3,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 5, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        state_preds = self.predict_state(x[:, 4])  # predict next state given (T+R+C)+state and action
        action_preds = self.predict_action(x[:, 3])  # predict next action given (T+R+C)+state
        if not return_dict:
            return (state_preds, action_preds)

        return DecisionTransformerOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            state_preds=state_preds,
            action_preds=action_preds,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )



class ff_Transformer_pred_time_lora(DecisionTransformerPreTrainedModel):
    # Add time-to-go & time prediction at each step 
    def __init__(self, config, r=16, alpha=1, p=0.1):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        # lora parameters
        self.lora_r = r
        self.lora_alpha = alpha
        self.lora_dropout = p

        # Use the original GPT2 model
        self.encoder = DecisionTransformerGPT2Model(config)

        # Replace linear layers with lora layers
        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_goal = loraLinear(config.state_dim, config.hidden_size, r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout)
        self.embed_return = loraLinear(1, config.hidden_size, r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout)
        self.embed_constraint = loraLinear(1, config.hidden_size, r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout)
        self.embed_time = loraLinear(1, config.hidden_size, r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout)
        self.embed_state = loraLinear(config.state_dim, config.hidden_size, r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout)
        self.embed_action = loraLinear(config.act_dim, config.hidden_size, r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout)

        self.embed_ln = nn.LayerNorm(config.hidden_size)

        self.predict_state = loraLinear(config.hidden_size, config.state_dim, r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout)
        self.predict_time = loraLinear(config.hidden_size, 1, r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout)
        
        # For the action prediction, we'll apply lora to the first linear layer
        self.predict_action = nn.Sequential(
            loraLinear(config.hidden_size, config.act_dim, r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout),
            *([] if not config.action_tanh else [nn.Tanh()])
        )

        # Initialize weights and apply final processing
        self.post_init()

    # The forward method remains the same as in the original class
    def forward(
        self,
        states: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        goal: Optional[torch.FloatTensor] = None,
        returns_to_go: Optional[torch.FloatTensor] = None,
        constraints_to_go: Optional[torch.FloatTensor] = None,
        times_to_go: Optional[torch.FloatTensor] = None,
        timesteps: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], DecisionTransformerOutput]:
        

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        goal_embeddings = self.embed_goal(goal)
        returns_embeddings = self.embed_return(returns_to_go)
        constraints_embeddings = self.embed_constraint(constraints_to_go)
        timetogo_embeddings = self.embed_time(times_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        goal_embeddings = goal_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        constraints_embeddings = constraints_embeddings + time_embeddings
        timetogo_embeddings = timetogo_embeddings + time_embeddings

        # this makes the sequence look like (T_1, R_1, C_1 s_1, t_1, a_1, T_2, R_2, C_2, s_2, t_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((goal_embeddings, returns_embeddings, constraints_embeddings, state_embeddings, timetogo_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 6 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask, attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 6 * seq_length)
        )
        device = stacked_inputs.device
        # we feed in the input embeddings (not word indices as in NLP) to the model
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(stacked_attention_mask.shape, device=device, dtype=torch.long),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = encoder_outputs[0]

        # reshape x so that the second dimension corresponds to the original
        # goal (0), return_to_go (1), constraint_to_go (2), states (3), time_to_go(4) or actions (5); i.e. x[:,3,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 6, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        state_preds = self.predict_state(x[:, 5])  # predict next state given (T+R+C)+state+(t) and action
        action_preds = self.predict_action(x[:, 4])  # predict next action given (T+R+C)+state+(t)
        timetogo_preds = self.predict_time(x[:, 3])  # predict next timetogo given (T+R+C)+state
        if not return_dict:
            return (state_preds, action_preds, timetogo_preds)

        return DecisionTransformerOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            state_preds=state_preds,
            action_preds=action_preds,
            timetogo_preds=timetogo_preds,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
