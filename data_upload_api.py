from flask import Flask, flash, request, redirect, url_for
import datetime
import pandas as pd
import os
from os.path import join, dirname
from dotenv import load_dotenv
import json
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

# from local .py files
import data_upload_mlflow
import data_generation_mlflow
import celery_queue

ALLOWED_EXTENSIONS = {'csv'}

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER')
app.config['DATAGEN_UPLOAD_FOLDER'] = os.environ.get('DATAGEN_UPLOAD_FOLDER')
app.config['SESSION_TYPE'] = os.environ.get('SESSION_TYPE')
app.config['SECRET_KEY'] = os.environ.get('FLASK_KEY')
app.config['UPLOAD_PORT'] = os.environ.get('UPLOAD_PORT')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1000 * 1000 # 100 MB
app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def type_text():
    return '''
        <!doctype html>
        <title>POST file</title>
        <form method="POST">
         <table>
            <tr>
                <td><label for="filename">File name:</label></td>
                <td><input type="text" name="filename"></td>
            </tr>
            <tr>
                <td><label for="window_size">Window size:</label></td>
                <td><input type="number" name="window_size"></td>
            </tr>
            <tr>
                <td><label for="window_step">Window step:</label></td>
                <td><input type="number" name="window_step"></td>
            </tr>
            <tr>
                <td><label for="input">JSON:</label></td>
                <td><textarea id="input" name="input" rows="4" cols="50">[
    {
        "filename": "Llama 2 - Open Foundation and Fine-Tuned Chat Models.pdf",
        "page": 1,
        "text": "Llama 2 : Open Foundation and Fine-Tuned Chat Models\nHugo Touvron\u2217Louis Martin\u2020Kevin Stone\u2020\nPeter Albert Amjad Almahairi Yasmine Babaei Nikolay Bashlykov Soumya Batra\nPrajjwal Bhargava Shruti Bhosale Dan Bikel Lukas Blecher Cristian Canton Ferrer Moya Chen\nGuillem Cucurull David Esiobu Jude Fernandes Jeremy Fu Wenyin Fu Brian Fuller\nCynthia Gao Vedanuj Goswami Naman Goyal Anthony Hartshorn Saghar Hosseini Rui Hou\nHakan Inan Marcin Kardas Viktor Kerkez Madian Khabsa Isabel Kloumann Artem Korenev\nPunit Singh Koura Marie-Anne Lachaux Thibaut Lavril Jenya Lee Diana Liskovich\nYinghai Lu Yuning Mao Xavier Martinet Todor Mihaylov Pushkar Mishra\nIgor Molybog Yixin Nie Andrew Poulton Jeremy Reizenstein Rashi Rungta Kalyan Saladi\nAlan Schelten Ruan Silva Eric Michael Smith Ranjan Subramanian Xiaoqing Ellen Tan Binh Tang\nRoss Taylor Adina Williams Jian Xiang Kuan Puxin Xu Zheng Yan Iliyan Zarov Yuchen Zhang\nAngela Fan Melanie Kambadur Sharan Narang Aurelien Rodriguez Robert Stojnic\nSergey Edunov Thomas Scialom\u2217\nGenAI, Meta\nAbstract\nIn this work, we develop and release Llama 2,"
    },
    {
        "filename": "Llama 2 - Open Foundation and Fine-Tuned Chat Models.pdf",
        "page": 1,
        "text": "Meta\nAbstract\nIn this work, we develop and release Llama 2, a collection of pretrained and fine-tuned\nlarge language models (LLMs) ranging in scale from 7 billion to 70 billion parameters.\nOur fine-tuned LLMs, called Llama 2-Chat , are optimized for dialogue use cases. Our\nmodels outperform open-source chat models on most benchmarks we tested, and based on\nourhumanevaluationsforhelpfulnessandsafety,maybeasuitablesubstituteforclosed-\nsource models. We provide a detailed description of our approach to fine-tuning and safety\nimprovements of Llama 2-Chat in order to enable the community to build on our work and\ncontribute to the responsible development of LLMs.\n\u2217Equal contribution, corresponding authors: {tscialom, htouvron}@meta.com\n\u2020Second author\nContributions for all the authors can be found in Section A.1.arXiv:2307.09288v2  [cs.CL]  19 Jul 2023"
    }
]</textarea><br><br></td>
            </tr>
        </table>
            <input type="submit">
        </form>
        '''

@app.route('/',methods=['POST'])
def handle_form_post(): 
    # todo: could not figure out how to sanitize user inputs      
    user_input = request.form['input']
    output_filename = request.form['filename']
    window_size = request.form['window_size']
    window_step = request.form['window_step']

    if len(output_filename) == 0:
        return 'Filename required'
    
    if window_size <= 0:
        return 'positive window_size required'
    
    if window_step <= 0:
        return 'positive window_step required'
    
    if len(user_input) == 0:
        return 'No input received'
    
    input_filepath = f'{app.config["UPLOAD_FOLDER"]}/{output_filename}_{datetime.datetime.now()}.json'
    output_path = json_to_csv(input_filepath, json.loads(user_input,strict=False))
    add_file_to_mlflow(output_path)
    celery_queue.generate_data.delay(output_path, window_size, window_step)
    
    return f'Success, your file is at {input_filepath}'
            
@app.route('/post_json',methods=['POST'])
def handle_json_post():
    content_type = request.headers.get('Content-Type')
    output_file_name = request.headers.get('File-Name')
    window_size = request.headers.get('window-size',type=int)
    window_step = request.headers.get('window-step',type=int)

    if output_file_name == None or len(output_file_name) == 0:
        # unsure if we want to allow file name to be a default
        return 'File-Name required in header'
    
    if window_size == None or window_size <= 0:
        return 'positive window_size required'
    
    if window_step == None or window_step <= 0:
        return 'positive window_step required'

    if (content_type == 'application/json'):
        json_input = request.json
        input_filepath = f'{app.config["UPLOAD_FOLDER"]}/{output_file_name}_{datetime.datetime.now()}.json'

        output_path = json_to_csv(input_filepath, json_input)
        add_file_to_mlflow(output_path)
        celery_queue.generate_data.delay(output_path, window_size, window_step)

        return f'Success, your file is at {input_filepath} and csv at {output_path}'
    else:
        return 'Content-Type not supported!'
    
@app.route('/upload_data_generation_edits',methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        run_id = request.form.get('run_id')

        if not run_id:
            flash('No run_id provided')
            run_id = ''

        file = request.files['file']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            duplicate_file_counter = 0

            while os.path.isfile(os.path.join(app.config['DATAGEN_UPLOAD_FOLDER'], filename)):
                if duplicate_file_counter > 0:
                    filename = filename.rsplit('_', 1)[0]
                filename = filename.rsplit('.',1)[0] + '_' + str(duplicate_file_counter) + '.csv'
                duplicate_file_counter +=1
            
            duplicate_file_counter = 0
            
            output_path = os.path.join(app.config['DATAGEN_UPLOAD_FOLDER'], filename)
            file.save(output_path)
            add_file_to_mlflow_datagen(output_path,run_id)

            if run_id == '':
                return f'No run_id provided, assume it is new run. Success, your csv is at {output_path}'

            return f'Success, your csv is at {output_path}, run_id: {run_id}'
    return '''
    <!doctype html>
    <title>Upload File</title>
    <h1>Upload File</h1>
    <form method=post enctype=multipart/form-data>
    <table>
            <tr>
                <td><label for="run_id">Run_id:</label></td>
                <td><input type="text" name="run_id"></td>
            </tr>
            <tr>
                <td><input type=file name=file></td>
                <td><input type=submit value=Upload></td>
            </tr>
    </table>
    </form>
    '''

@app.route('/evaluate_model',methods=['POST'])
def evaluate_model_input():
    model_name = request.headers.get('Model-Name')
    benchmark_uri = request.headers.get('Benchmark-Uri')

    celery_queue.evaluate_model(model_name, benchmark_uri)

    return 'Success, your evaluation is processing'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def json_to_csv(input_filepath, json_user_input):
        # create a backup of the json uploaded
        with open(input_filepath, 'w', encoding='utf-8') as f:
            json.dump(json_user_input, f, ensure_ascii=False, indent=4)
        
        # in case the json is not a flat file
        df = pd.json_normalize(json_user_input)

        output_filepath = f'{input_filepath.replace(".json","")}.csv'

        df.to_csv(output_filepath, encoding='utf-8', index=False)

        return output_filepath

def add_file_to_mlflow(output_path):
    data_upload_mlflow.add_file(output_path, output_path.split('/')[-1])

def add_file_to_mlflow_datagen(output_path, run_id):
    data_generation_mlflow.add_file(output_path, output_path.split('/')[-1], run_id)

if __name__ == '__main__':
    app.run(debug=True, port=app.config['UPLOAD_PORT'])