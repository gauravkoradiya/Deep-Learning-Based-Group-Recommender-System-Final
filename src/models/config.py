"""
Configurations
"""
import os
import torch

class DRGR_Config(object):
    """
    Configurations
    """
    def __init__(self,data_set):
        # Data
        self.data_set = data_set
        self.data_folder_path = os.path.join('data','processed',self.data_set)
        self.item_path = os.path.join(self.data_folder_path,'item_data.csv')
        self.user_path = os.path.join(self.data_folder_path,'users_data.csv')
        self.group_path = os.path.join(self.data_folder_path,'groupMember.dat')
        self.saves_folder_path = os.path.join('saves')
        self.save_model_path= os.path.join('models','DRGR')
        
        # Recommendation system
        self.history_length = 5
        self.top_K_list = [5, 10, 20]
        self.rewards = [0, 5]

        # Reinforcement learning
        self.embedding_size = 32
        self.state_size = self.history_length + 1
        self.action_size = 1
        self.embedded_state_size = self.state_size * self.embedding_size
        self.embedded_action_size = self.action_size * self.embedding_size

        # Numbers
        self.item_num = None
        self.user_num = None
        self.group_num = None
        self.total_group_num = None

        # Environment
        self.env_n_components = self.embedding_size
        self.env_tol = 1e-4
        self.env_max_iter = 1000
        self.env_alpha = 0.001

        # Actor-Critic network
        self.actor_hidden_sizes = (128, 64)
        self.critic_hidden_sizes = (32, 16)

        # DDPG algorithm
        self.tau = 1e-3
        self.gamma = 0.9

        # Optimizer
        self.batch_size = 64
        self.buffer_size = 100000
        self.num_episodes = 2
        self.num_steps = 200
        self.embedding_weight_decay = 1e-4
        self.actor_weight_decay = 1e-4
        self.critic_weight_decay = 1e-4
        self.embedding_learning_rate = 1e-2
        self.actor_learning_rate = 1e-2
        self.critic_learning_rate = 1e-2
        self.eval_per_iter = 10

        # OU noise
        self.ou_mu = 0.1
        self.ou_theta = 0.15
        self.ou_sigma = 0.2
        self.ou_epsilon = 1.0

        # GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

class AGREE_Config(object):
    def __init__(self,data_set):
        self.data_set = data_set
        self.path = os.path.join('data','processed',self.data_set)
        self.user_dataset = os.path.join(self.path,'userRating')
        self.group_dataset = os.path.join(self.path, 'groupRating')
        self.user_in_group_path = os.path.join(self.path, 'groupMember.dat')
        self.embedding_size = 32
        self.epoch = 1
        self.num_negatives = 6
        self.batch_size = 256
        self.lr = [0.000005, 0.000001, 0.0000005]
        self.drop_ratio = 0.2
        self.topK = 5
        self.save_model_path= os.path.join('models','AGREE')