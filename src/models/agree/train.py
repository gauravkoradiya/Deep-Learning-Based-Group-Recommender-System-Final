from .agree import AGREE
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from time import time
from config import AGREE_Config
from .util import Helper
from .dataset import GDataset
import os
from tqdm import tqdm
import time


import logging

logger = logging.getLogger(__name__)

class Trainer():

    def __init__(self, config:AGREE_Config):
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

        # initial helper
        self.helper = Helper()

        # get the dict of users in group
        self.g_m_d = self.helper.gen_group_member_dict(self.config.user_in_group_path)

        # initial dataSet class
        self.dataset = GDataset(self.config.user_dataset, self.config.group_dataset, self.config.num_negatives)

        # get group number
        num_group = len(self.g_m_d)
        num_users, num_items = self.dataset.num_users, self.dataset.num_items

        # build AGREE model
        self.agree = AGREE(num_users, num_items, num_group, self.config.embedding_size, self.g_m_d, self.config.drop_ratio)

        self.history = {}

    def train_step(self, dataloader, epoch_id, type_m):

        # user trainning
        learning_rates = self.config.lr
        # learning rate decay
        lr = learning_rates[0]
        if epoch_id >= 15 and epoch_id < 25:
            lr = learning_rates[1]
        elif epoch_id >=20:
            lr = learning_rates[2]
        # lr decay
        if epoch_id % 5 == 0:
            lr /= 2

        # optimizer
        optimizer = optim.RMSprop(self.agree.parameters(), lr)

        losses = []
        print('%s train_loader length: %d' % (type_m, len(dataloader)))
        for batch_id, (u, pi_ni) in tqdm(enumerate(dataloader)):
            # Data Load
            user_input = u
            pos_item_input = pi_ni[:, 0]
            neg_item_input = pi_ni[:, 1]
            # Forward
            if type_m == 'user':
                pos_prediction = self.agree(None, user_input, pos_item_input)
                neg_prediction = self.agree(None, user_input, neg_item_input)
            elif type_m == 'group':
                pos_prediction = self.agree(user_input, None, pos_item_input)
                neg_prediction = self.agree(user_input, None, neg_item_input)
            # Zero_grad
            self.agree.zero_grad()
            # Loss
            loss = torch.mean((pos_prediction - neg_prediction -1) **2)
            # record loss history
            losses.append(loss)  
            # Backward
            loss.backward()
            optimizer.step()
        avg_loss = torch.mean(torch.stack(losses))
        print('Iteration %d, loss is [%.4f ]' % (epoch_id, avg_loss))
        return avg_loss
            
    def training(self):
        # config information
        print("AGREE at embedding size %d, run Iteration:%d, NDCG and HR at %d" %(self.config.embedding_size, self.config.epoch, self.config.topK))
        # train the model
        for epoch in range(self.config.epoch):
            self.agree.train()
            t1 = time.time()
            avg_user_loss = self.train_step(self.dataset.get_user_dataloader(self.config.batch_size), epoch,  'user')
            avg_group_loss = self.train_step(self.dataset.get_group_dataloader(self.config.batch_size), epoch, 'group')
            print("user and group training time is: [%.1f s]" % (time.time()-t1))
            
            t2 = time.time()
            u_hr, u_ndcg = self.evaluation(self.dataset.user_testRatings, self.dataset.user_testNegatives, self.config.topK, 'user')
            print('User Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, [%.1f s]' % (epoch, time.time() - t1, u_hr, u_ndcg, time.time() - t2))

            g_hr, g_ndcg = self.evaluation(self.dataset.group_testRatings, self.dataset.group_testNegatives, self.config.topK, 'group')
            print('Group Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, [%.1f s]' % (epoch, time.time() - t1, g_hr, g_ndcg, time.time() - t2))

            self.history[epoch] = {"User_Average_Loss" : avg_user_loss.item(), "Group_Average_Loss": avg_group_loss.item(), "User_HR": u_hr.item(), "User_NDCG": u_ndcg.item(), "Group_HR": g_hr.item(), "Group_NDCG": g_ndcg.item()}
        #self.save_agent()
        return self.history

    def evaluation(self, testRatings, testNegatives, K, type_m):
        self.agree.eval()
        (hits, ndcgs) = self.helper.evaluate_model(self.agree, testRatings, testNegatives, K, type_m)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        return hr, ndcg      

    def save_agent(self):
        """
            Save the agent
        """
        try:
            os.makedirs(self.config.save_model_path, exist_ok=True)
            torch.save(self.agree, os.path.join(self.config.save_model_path, 'AGREE_model_{0}.pth'.format(time())))
            logger.info('Model saved at %s' % self.config.save_model_path)
        except:
            logger.error('Failed to save model.')


    