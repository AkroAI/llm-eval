import os
import mlflow
from os.path import join, dirname
from dotenv import load_dotenv
import pandas as pd
from mlflow.data.pandas_dataset import PandasDataset

#local imports
import evaluation

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

PORT = os.environ.get('PORT')

mlflow.set_tracking_uri(f'http://127.0.0.1:{PORT}')
experiment_name = 'evaluation'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

def evaluate(model_name,benchmark_uri):
    mlflow.set_experiment(experiment_name)

    local_artifact_filepath = mlflow.artifacts.download_artifacts(benchmark_uri,dst_path='./eval_downloads')

    with mlflow.start_run(run_name=benchmark_uri.rsplit('/', 1)[1]):

        df = pd.read_csv(local_artifact_filepath)

        dataset: PandasDataset = mlflow.data.from_pandas(df, source=local_artifact_filepath)

        mlflow.log_input(dataset, context="evaluation")

        result_filepath = evaluation.evaluate_model(model_name, local_artifact_filepath)

        mlflow.log_artifact(result_filepath)