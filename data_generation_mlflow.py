import os
import csv
import mlflow
import pandas as pd
from os.path import join, dirname
from dotenv import load_dotenv
from mlflow.data.pandas_dataset import PandasDataset

# local imports
import celery_queue
import data_generation_rag

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

PORT = os.environ.get('PORT')
mlflow.set_tracking_uri(f'http://127.0.0.1:{PORT}')
experiment_name = 'data_generation'
input_folder_name = os.environ.get('DATAGEN_UPLOAD_FOLDER')

def add_file(file_to_upload,run_name,run_id=''):
    '''Adds uploaded file to the mlflow UI via artifacts.'''
    # reset it each time to avoid clash with data_generation.py
    mlflow.set_experiment(input_folder_name)
    
    with mlflow.start_run(run_name=run_name,run_id=run_id):
        # in artifacts, doesn't display as csv if only 1 column
        # not sure why it just shows everything on 1 row instead
        mlflow.log_artifact(file_to_upload)

def generate_data(filepath, window_size, window_step):

    mlflow.set_experiment(experiment_name)

    df = pd.read_csv(filepath)
    # Construct an MLflow PandasDataset from the Pandas DataFrame, and specify
    # filepath as the source
    dataset: PandasDataset = mlflow.data.from_pandas(df, source=filepath)

    with mlflow.start_run(run_name=filepath):
        mlflow.log_input(dataset, context="generation")

        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
            
        json_output_path = data_generation_rag.get_file(filepath, experiment_name, window_size, window_step)
        
        csv_output_path = data_generation_rag.reformat(json_output_path)

        mlflow.log_artifact(csv_output_path)