"""
Training the model
"""
import pandas as pd
import mlflow
import traceback
from drgr.agent import DDPGAgent
from config import AGREE_Config, DRGR_Config

from agree.util import Helper
from agree.agree import AGREE
from agree.train import Trainer as AGREE_Trainer
from agree.eval import Evaluator


from drgr.data import DataLoader
from drgr.env import Env
from drgr.eval import Evaluator
from drgr.utils import OUNoise
from drgr.train import Trainer 
import click
import logging
import os

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

@click.command()
@click.option('--algorithm', type = str, default = 'DRGR', show_default = True, help ='Algorithm to be used for training. Possible values are ["DRGR","AGREE"]')
@click.option('--dataset', type = str, default = 'Book_Review', show_default = True, help ='Dataset to be used for training. Possible values are ["Book_Review, "Market_Bias"]')
def main(algorithm, dataset):
    with mlflow.start_run(experiment_id=0):
        try:
            if algorithm == 'DRGR':
                config = DRGR_Config(data_set=dataset)
                # config.data_folder_path = os.path.join(config.data_folder_path, dataset)
                dataloader = DataLoader(config)
                # mlflow.log_param("config", config)
                mlflow.log_params({"Dataset": dataset , "Algorithm" : "DRGR", "Model": "DDPG" , "Operation": "Training"})
                rating_matrix_train = dataloader.load_rating_matrix(dataset_name='train')
                df_eval_user_test = dataloader.load_eval_data(mode='user', dataset_name='test')
                df_eval_group_test = dataloader.load_eval_data(mode='group', dataset_name='test')
                env = Env(config=config, rating_matrix=rating_matrix_train, dataset_name='train')
                noise = OUNoise(config=config)
                agent = DDPGAgent(config=config, noise=noise, group2members_dict=dataloader.group2members_dict, verbose=True)
                evaluator = Evaluator(config=config)
                trainer = Trainer(config=config, env=env, agent=agent, evaluator=evaluator, df_eval_user=df_eval_user_test, df_eval_group=df_eval_group_test)
                history = trainer.train()
                mlflow.log_dict(history, 'training/history.json')
                mlflow.log_dict(config.__dict__,'training/config.yaml')

                # Log the model with MLflow
                # Log the model file as an artifact
                mlflow.pytorch.log_model(pytorch_model=trainer.agent, artifact_path="model")

                # Register the model
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                registered_model = mlflow.register_model(model_uri=model_uri, name="DRGR")
            
            elif algorithm == 'AGREE':
                mlflow.log_params({"Dataset": dataset, "Algorithm" : "AGREE", "Model": "DDPG" , "Operation": "Training"})

                 # initial parameter class
                config = AGREE_Config(data_set=dataset)
                # config.data_folder_path = os.path.join(config.data_folder_path, dataset)
             
                # initial trainer
                trainer = AGREE_Trainer(config=config)
                
                # train the model
                history = trainer.training()
                mlflow.log_dict(history, 'training/history.json')
                mlflow.log_dict(config.__dict__,'training/config.yaml')

                 # Log the model with MLflow
                # Log the model file as an artifact
                # mlflow.log_artifact("path/to/model.pth", artifact_path="models")
                mlflow.pytorch.log_model(pytorch_model=trainer.agree, artifact_path="model")

                # Register the model
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                registered_model = mlflow.register_model(model_uri=model_uri, name="AGREE")

            else:
                raise Exception("Invalid algorithm")

        except Exception as e:
            # log the stack trace as a parameter
            logger.error("Failed to run script", exc_info=True)
            mlflow.log_text(str(traceback.format_exc()), 'Error.txt')
            # end the run with an error status
            mlflow.end_run(status="FAILED")


if __name__ == '__main__':
    main()        
