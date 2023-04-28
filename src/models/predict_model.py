"""
    Inference for the recommendation system.
"""

import pandas as pd
import mlflow
import traceback
import torch
from drgr.agent import DDPGAgent
from config import AGREE_Config, DRGR_Config
from drgr.data import DataLoader
from drgr.env import Env
from drgr.eval import Evaluator
from drgr.utils import OUNoise

from agree.agree import AGREE
from agree.util import Helper
from agree.dataset import GDataset
from agree.eval import Evaluator as AGREE_Evaluator

import click
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

@click.command()
@click.option('--algorithm', type = str, default = 'DRGR', show_default = True, help ='Algorithm to be used for Inference. Possible values are ["DRGR","AGREE"].')
@click.option('--mode', type=str, default = 'user', show_default = True, help ='Nature of Recommender System. Possible values are ["user","group"].' )
@click.option('--run_id', type=str, help ='Get model from run_id in mlflow.' )
def main(algorithm, mode,run_id):
    with mlflow.start_run(experiment_id=0) as run:
        try:
            if algorithm == 'DRGR':
                config = DRGR_Config()
                dataloader = DataLoader(config)
                mlflow.log_params({"Dataset": "Book Review", "Algorithm": "DRGR",
                                "Model": "DDPG", "Operation": "Evaluation"})
                mlflow.log_param("mode", mode)

                rating_matrix_train = dataloader.load_rating_matrix(
                    dataset_name='test')
                df_eval_user_val = dataloader.load_eval_data(
                    dataset_name='val', mode='user')
                df_eval_group_val = dataloader.load_eval_data(
                    dataset_name='val', mode='group')
                env = Env(config=config, rating_matrix=rating_matrix_train,
                        dataset_name='train')
                noise = OUNoise(config=config)

                # agent = DDPGAgent(config=config, noise=noise, group2members_dict=dataloader.group2members_dict, verbose=True)
                # agent = torch.load('models/DRGR/ddpg_model_1682529405.3536892.pth')
                # agent.eval()
                # Inference after loading the logged model
                model_uri = "runs:/{}/model".format(run_id)
                agent = mlflow.pytorch.load_model(model_uri)
                agent.eval()

                evaluator = Evaluator(config=config)
                evaluator.predict(agent, df_eval_group_val, mode='group', top_K=5)
                if mode == 'group':
                    avg_recall_score, avg_ndcg_score = evaluator.evaluate(agent, df_eval_group_val, mode='group', top_K=5)
                else:
                    avg_recall_score, avg_ndcg_score = evaluator.evaluate(agent, df_eval_user_val, mode='user', top_K=5)
                mlflow.log_dict(config.__dict__, "inference/config.yaml")
                mlflow.log_metric("Recall",avg_recall_score)
                mlflow.log_metric("NDCG", avg_ndcg_score)
                logger.info(" DONE !!")
                
            elif algorithm == 'AGREE':
                mlflow.log_params({"Dataset": "Book Review", "Algorithm": "AGREE", "Model": "DDPG", "Operation": "Evaluation"})
                mlflow.log_param("mode", mode)
                config = AGREE_Config()
                helper = Helper()

                # initial dataset class
                dataset = GDataset(config.user_dataset, config.group_dataset, config.num_negatives)

                # Inference after loading the logged model
                model_uri = "runs:/{}/model".format(run_id)
                model = mlflow.pytorch.load_model(model_uri)

                evaluator = AGREE_Evaluator(model = model, helper=helper)
                #evaluator.predict(dataset.group_testRatings, dataset.group_testNegatives, config.topK, 'group')
                if mode == 'group':
                    hr, ndcg =  evaluator.evaluate(dataset.group_testRatings, dataset.group_testNegatives, config.topK, 'group')
                else:
                    hr, ndcg = evaluator.evaluate(dataset.group_testRatings, dataset.group_testNegatives, config.topK, 'user')
                
                mlflow.log_dict(config.__dict__, "inference/config.json")
                mlflow.log_metric("Recall", hr)
                mlflow.log_metric("NDCG", ndcg)
                logger.info(" DONE !!")
            else:
                pass

        except Exception as e:
            logger.error(str(traceback.format_exc()))
            # log the stack trace as a parameter
            mlflow.set_tag("Error", str(traceback.format_exc()))
            # end the run with an error status
            mlflow.end_run(status="FAILED")


if __name__ == '__main__':
    main()
