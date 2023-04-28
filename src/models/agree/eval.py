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
import heapq
import pandas as pd
from tqdm import tqdm


class Evaluator(object):
    """
    Evaluator
    """

    def __init__(self, model: AGREE, helper: Helper):
        """
        Initialize Evaluator

        :param config: configurations
        """
        self.model = model
        self.helper = helper

    def predict(self, testRatings, testNegatives, K, type_m):
        """
        Predict the recommendation list
        """
        df_pred = pd.DataFrame()
        for idx in range(len(testRatings)):
            rating = testRatings[idx]
            items = testNegatives[idx]
            u = rating[0]
            gtItem = rating[1]
            items.append(gtItem)
            # Get prediction scores
            map_item_score = {}
            users = np.full(len(items), u)

            users_var = torch.from_numpy(users)
            users_var = users_var.long()
            items_var = torch.LongTensor(items)
            if type_m == 'group':
                predictions = self.model(users_var, None, items_var)
            elif type_m == 'user':
                predictions = self.model(None, users_var, items_var)
            for i in range(len(items)):
                item = items[i]
                map_item_score[item] = predictions.data.numpy()[i]
            items.pop()

            # Evaluate top rank list
            ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)

            # Appending Result
            if type_m == 'group':
                df_pred = df_pred.append({'Group': u, 'Ground_Truth': gtItem, 'Prediction_Item': ranklist}, ignore_index=True)
            else:
                df_pred = df_pred.append({'User': u, 'Ground_Truth': gtItem, 'Prediction_Item': ranklist}, ignore_index=True)

        return df_pred

    def evaluate(self, testRatings, testNegatives, K, type_m):
        """
        Evaluate the agent
        """
        self.model.eval()
        (hits, ndcgs) = self.helper.evaluate_model(
            self.model, testRatings, testNegatives, K, type_m)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        return hr, ndcg
