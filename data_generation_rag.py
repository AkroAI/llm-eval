import logging
import sys
import os
from os.path import join, dirname
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.schema import TextNode
from pathlib import Path
import datetime
import pandas as pd

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# set context for llm provider
llm_context=OpenAI(model="gpt-3.5-turbo", temperature=0.3)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# api_key is added automatically, no need to re-specify
ai_model = OpenAI(model='gpt-3.5-turbo')

def get_file(input_filepath, output_folder, window_size, window_step):

    filename = input_filepath.split('/')[-1].replace('.csv','')

    # we want to customise the node ourselves
    # so we have to manually define it
    list_of_nodes = get_node_from_csv(input_filepath, window_size, window_step)

    dataset_generator = RagDatasetGenerator(
        nodes=list_of_nodes,
        llm=llm_context,
        # num_questions_per_chunk=1, # for testing only
        show_progress=True
    )

    print(f'Nodes: {len(dataset_generator.nodes)}')

    rag_dataset = dataset_generator.generate_dataset_from_nodes()

    save_location = f'{output_folder}/{filename}_rag_{datetime.datetime.now()}.json'

    rag_dataset.save_json(save_location)

    return save_location

def reformat(input_filepath):
    """Reformats raw rag output into format""" 
    # grab input 
    df = pd.read_json(input_filepath)
    df_output = pd.json_normalize(df['examples'])

    # generate required format
    df_final_output = pd.DataFrame()
    df_final_output['query'] = df_output['query']
    df_final_output['relevant_chunks'] = df_output['reference_contexts']
    df_final_output['response'] = df_output['reference_answer']

    print(df_final_output.head())

    output_filepath = f'{input_filepath.replace(".json","")}.csv'

    df_final_output.to_csv(output_filepath, encoding='utf-8', index=False)

    return output_filepath

def get_node_from_csv(input_filepath, window_size, window_step):
    list_of_nodes = []
    df = pd.read_csv(Path(input_filepath))

    # df.rolling doesn't work for non-numeric types
    # implement manual one
    # deque doesn't catch edge case of window_step > window_size (or it would be more complicated)

    grouped_chunks = pd.DataFrame(columns=['text','filename','page'])
    window_row = pd.Series()
    window_row['text'] = ''
    window_row['filename'] = ['']
    window_row['page'] = []

    total_csv_rows = df.shape[0]
    current_row = 0

    left = 0
    right = window_size

    while True:
        text_list = []
        filename_list = []
        page_list = []

        for idx in range(left, right):
            current_row = df.iloc[idx]
            text_list.append(current_row['text'])
            filename_list.append(current_row['filename'])
            page_list.append(current_row['page'])

        window_row['text'] = ' '.join(text_list)
        window_row['filename'] = list(filename_list)
        window_row['page'] = list(page_list)
        grouped_chunks.loc[grouped_chunks.shape[0]] = window_row

        if right == total_csv_rows: break

        left += window_step
        right = min(right + window_step, total_csv_rows)

    for i in range(grouped_chunks.shape[0]):
        row = grouped_chunks.loc[i]
        print(row['text'])
        list_of_nodes.append(TextNode(
            text=row['text'],
            extra_info={
                'filename':row['filename'],
                'page':row['page']
                }
            ))

    return list_of_nodes