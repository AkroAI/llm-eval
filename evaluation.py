import json

def evaluate_model(model_to_evaluate, benchmark_filepath):
    # {'context_precision': 0.817,
    # 'faithfulness': 0.892, 
    # 'answer_relevancy': 0.874}

    result_dict = {
    'context_precision': 0.1,
    'faithfulness': 0.1,
    'answer_relevancy': 0.1
    }

    output_filename = 'evaluation_outputs/evaluation_' + model_to_evaluate + '_' + benchmark_filepath.rsplit('/',1)[1].rsplit('.',1)[0] + '.json'

    with open(output_filename, 'w') as f:
        f.write(json.dumps(result_dict))

    return output_filename