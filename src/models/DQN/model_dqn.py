

"""
Models
"""
from typing import Tuple

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    

    def __init__(self, embedded_state_size: int, action_weight_size: int, hidden_sizes: Tuple[int]):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embedded_state_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_weight_size),
        )

    def forward(self, embedded_state):
        """
        Forward

        :param embedded_state: embedded state
        :return: action weight
        """
        return self.net(embedded_state)


class TargetNetwork(nn.Module):
   
    def __init__(self, embedded_state_size: int, embedded_action_size: int, hidden_sizes: Tuple[int]):
        super(TargetNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embedded_state_size + embedded_action_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )

    def forward(self, embedded_state, embedded_action):
        return self.net(torch.cat([embedded_state, embedded_action], dim=-1))


class Embedding(nn.Module):
   
    def __init__(self, embedding_size: int, user_num: int, item_num: int):
       
        super(Embedding, self).__init__()
        self.user_embedding = nn.Embedding(user_num + 1, embedding_size)
        self.item_embedding = nn.Embedding(item_num + 1, embedding_size)
        self.user_attention = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 1)
        )
        self.user_softmax = nn.Softmax(dim=-1)

    def forward(self, group_members, history):
        
        embedded_group_members = self.user_embedding(group_members)
        group_member_attentions = self.user_softmax(self.user_attention(embedded_group_members))
        embedded_group = torch.squeeze(torch.inner(group_member_attentions.T, embedded_group_members.T))
        embedded_history = torch.flatten(self.item_embedding(history), start_dim=-2)
        embedded_state = torch.cat([embedded_group, embedded_history], dim=-1)
        return embedded_state


def Qloss(batch, net, gamma=0.99, device="cpu"):
    states, actions, next_states, rewards, _ = batch
    lbatch = len(states)
    state_action_values = net(states.view(lbatch,-1))
    state_action_values = state_action_values.gather(1, actions.unsqueeze(-1))
    state_action_values = state_action_values.squeeze(-1)
    
    next_state_values = net(next_states.view(lbatch, -1))
    next_state_values = next_state_values.max(1)[0]
    
    next_state_values = next_state_values.detach()
    target = next_state_values*gamma + rewards
    
    return nn.MSELoss()(state_action_values, target)#finding loss between x and y

