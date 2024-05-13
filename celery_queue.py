from flask import Flask
from celery import Celery, Task, shared_task

# local file imports
import data_generation_mlflow
import evaluation_mlflow

# config for celery job queue
def celery_init_app(app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask)
    celery_app.config_from_object(app.config['CELERY'])
    celery_app.set_default()
    app.extensions['celery'] = celery_app
    return celery_app

app = Flask(__name__)
app.config.from_mapping(
    CELERY=dict(
        broker_url='pyamqp://guest@localhost//',
        result_backend='redis://localhost:6379/0',
        task_ignore_result=True
    ),
)
celery_app = celery_init_app(app)

# config for job queue
# separated logic and queue to make it clearer.

# queue within data_upload_api
@shared_task()
def generate_data(filepath, window_size, window_step):
    data_generation_mlflow.generate_data(filepath, window_size, window_step)

@shared_task()
def evaluate_model(model_name, benchmark_uri):
    evaluation_mlflow.evaluate(model_name, benchmark_uri)