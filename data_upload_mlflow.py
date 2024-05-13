import os
import mlflow
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

PORT = os.environ.get('PORT')
input_folder_name = os.environ.get('UPLOAD_FOLDER')

mlflow.set_tracking_uri(f'http://127.0.0.1:{PORT}')

def add_file(file_to_upload,run_name):
    '''Adds uploaded file to the mlflow UI via artifacts.'''
    # reset it each time to avoid clash with data_generation.py
    mlflow.set_experiment(input_folder_name)
    
    with mlflow.start_run(run_name=run_name):
        # in artifacts, doesn't display as csv if only 1 column
        # not sure why it just shows everything on 1 row instead
        mlflow.log_artifact(file_to_upload)