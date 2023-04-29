"""
DDPG Agent
"""

from typing import List

import numpy as np
import torch
import torch.nn.functional as functional
from torch import optim, nn

import model_dqn as model
import utils as utils
from config import Config
import collections

Transition = collections.namedtuple('Experience',
                                    field_names=['state', 'action',
                                                 'next_state', 'reward',
                                                 'is_game_on'])

class DQNAgent(object):

    def __init__(self, config: Config, noise: utils.OUNoise, group2members_dict: dict, verbose=False):
       
        self.config = config
        self.noise = noise
        self.group2members_dict = group2members_dict
        self.tau = config.tau
        self.gamma = config.gamma
        self.device = config.device

        self.embedding = model.Embedding(embedding_size=config.embedding_size,
                                         user_num=config.user_num,
                                         item_num=config.item_num).to(config.device)
        self.qnet = model.QNetwork(embedded_state_size=config.embedded_state_size,
                                 action_weight_size=config.embedded_action_size,
                                 hidden_sizes=config.actor_hidden_sizes).to(config.device)
       
        self.tnet = model.TargetNetwork(embedded_state_size=config.embedded_state_size,
                                   embedded_action_size=config.embedded_action_size,
                                   hidden_sizes=config.critic_hidden_sizes).to(config.device)

        self.replay_memory = utils.ReplayMemory(buffer_size=config.buffer_size)

        self.optimizer = optim.Adam([
                {'params': self.embedding.parameters()},
                {'params': self.qnet.parameters(), 'lr': 1e-3}
            ], lr=1e-3)
        self.tnet_optimizer = optim.Adam(self.tnet.parameters(), lr=config.critic_learning_rate,
                                           weight_decay=config.critic_weight_decay)

    def get_action(self, state: list, item_candidates: list = None, top_K: int = 1):
        with torch.no_grad():
            states = [state]
            
            embedded_states = self.embed_states(states)
            # print(len(state), embedded_states.detach().numpy().shape)
            action_weights = self.qnet(embedded_states)
            action_weight = torch.squeeze(action_weights)

            if item_candidates is None:
                item_embedding_weight = self.embedding.item_embedding.weight.clone()
            else:
                item_candidates = np.array(item_candidates)
                item_candidates_tensor = torch.tensor(item_candidates, dtype=torch.int).to(self.device)
                item_embedding_weight = self.embedding.item_embedding(item_candidates_tensor)

            scores = torch.inner(action_weight, item_embedding_weight).detach().cpu().numpy()
            sorted_score_indices = np.argsort(scores)[:top_K]
            if item_candidates is None:
                action = sorted_score_indices
            else:
                action = item_candidates[sorted_score_indices]
            action = np.squeeze(action)
            if top_K == 1:
                action = action.item()
        return action

    def get_embedded_actions(self, embedded_states: torch.Tensor):
        action_weights = self.qnet(embedded_states)
        item_embedding_weight = self.embedding.item_embedding.weight.clone()
        scores = torch.inner(action_weights, item_embedding_weight)
        embedded_actions = torch.inner(functional.gumbel_softmax(scores, hard=True), item_embedding_weight.t())
        return embedded_actions

    def embed_state(self, state: list):
        group_id = state[0]
        group_members = torch.tensor(self.group2members_dict[group_id], dtype=torch.int).to(self.device)
        history = torch.tensor(state[1:], dtype=torch.int).to(self.device)
        embedded_state = self.embedding(group_members, history)
        return embedded_state

    def embed_states(self, states: List[list]):
        embedded_states = torch.stack([self.embed_state(state) for state in states], dim=0)
        return embedded_states

    def embed_actions(self, actions: list):
        actions = torch.tensor(actions, dtype=torch.int).to(self.device)
        embedded_actions = self.embedding.item_embedding(actions)
        return embedded_actions

    def update(self, episode):
        batch = self.replay_memory.sample(self.config.batch_size)
        states, actions, rewards, next_states = list(zip(*batch))
        if episode%15 == 0:
            self.tnet_optimizer.zero_grad()

        self.optimizer.zero_grad()
       
        
        embedded_states = self.embed_states(states)
        embedded_actions = self.embed_actions(actions)
        rewards = torch.unsqueeze(torch.tensor(rewards, dtype=torch.int).to(self.device), dim=-1)
        embedded_next_states = self.embed_states(next_states)
        q_values = self.tnet(embedded_states, embedded_actions)

        with torch.no_grad():
            embedded_next_actions = self.get_embedded_actions(embedded_next_states)
            next_q_values = self.tnet(embedded_next_states, embedded_next_actions)
            q_values_target = rewards + self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, q_values_target)
        loss.backward()

        if episode%15 == 0:
            self.tnet_optimizer.step()

        self.optimizer.step()

        return loss.detach().cpu().numpy()