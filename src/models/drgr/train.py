"""
Main
"""
import pandas as pd

from .agent import DDPGAgent
from config import DRGR_Config
# from .data import DataLoader
from .env import Env
from .eval import Evaluator
from .utils import OUNoise
import torch
import os
import time
import logging

logger = logging.getLogger(__name__)

class Trainer():

    def __init__(self, config: DRGR_Config, env: Env, agent: DDPGAgent, evaluator: Evaluator, df_eval_user: pd.DataFrame(), df_eval_group: pd.DataFrame()):
        """
        Initialize Trainer 

        :param config: configurations
        :param env: environment
        :param agent: agent
        :param evaluator: evaluator
        :param df_eval_user: user evaluation data
        :param df_eval_group: group evaluation data 
        
        """
        self.config = config
        self.env = env
        self.agent = agent
        self.evaluator = evaluator
        self.df_eval_user = df_eval_user
        self.df_eval_group = df_eval_group
        self.history = {}

    def train(self):
        """
        Train the agent with the environment

        :return:
        """
        try:
            rewards = []
            for episode in range(self.config.num_episodes):
                hist = {}
                state = self.env.reset()
                self.agent.noise.reset()
                episode_reward = 0

                for step in range(self.config.num_steps):
                    action = self.agent.get_action(state)
                    new_state, reward, _, _ = self.env.step(action)
                    self.agent.replay_memory.push((state, action, reward, new_state))
                    state = new_state
                    episode_reward += reward
                    if len(self.agent.replay_memory) >= self.config.batch_size:
                        self.agent.update()
                reward_per_episode = episode_reward / self.config.num_steps
                rewards.append(reward_per_episode)
                print('Episode = %d, average reward = %.4f' % (episode, reward_per_episode))
                if (episode + 1) % self.config.eval_per_iter == 0:
                    hist['reward'] = reward_per_episode
                    for top_K in self.config.top_K_list:
                        avg_user_recall, avg_user_ndcg = self.evaluator.evaluate(agent=self.agent, df_eval=self.df_eval_user, mode='user', top_K=top_K)
                        hist[f'user_recall@{top_K}'] = avg_user_recall
                        hist[f'user_ndcg@{top_K}'] = avg_user_ndcg
                    for top_K in self.config.top_K_list:
                        avg_group_recall, avg_group_ndcg = self.evaluator.evaluate(agent=self.agent, df_eval=self.df_eval_group, mode='group', top_K=top_K)
                        hist[f'group_recall@{top_K}'] = avg_group_recall
                        hist[f'group_ndcg@{top_K}'] = avg_group_ndcg
                    self.history[episode] = hist

            logger.info('Training finished')
            logger.info('Saving model')
            #self.save_agent()
            return self.history
        except:
            logger.error('Training failed')
            raise Exception('Training failed')

    def save_agent(self):
        """
        Save the agent

        :return:

        """
        try:
            os.makedirs(self.config.save_model_path, exist_ok=True)
            torch.save(self.agent, os.path.join(self.config.save_model_path, 'ddpg_model_{0}.pth'.format(time.time())))
            logger.info('Model saved at %s' % self.config.save_model_path)
        except:
            logger.error('Failed to save model.')
# if __name__ == '__main__':
#     config = Config()
#     dataloader = DataLoader(config)
#     rating_matrix_train = dataloader.load_rating_matrix(dataset_name='val')
#     df_eval_user_test = dataloader.load_eval_data(mode='user', dataset_name='test')
#     df_eval_group_test = dataloader.load_eval_data(mode='group', dataset_name='test')
#     env = Env(config=config, rating_matrix=rating_matrix_train, dataset_name='val')
#     noise = OUNoise(config=config)
#     agent = DDPGAgent(config=config, noise=noise, group2members_dict=dataloader.group2members_dict, verbose=True)
#     evaluator = Evaluator(config=config)
#     train(config=config, env=env, agent=agent, evaluator=evaluator,
#           df_eval_user=df_eval_user_test, df_eval_group=df_eval_group_test)
