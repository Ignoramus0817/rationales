import json
import os
import argparse
import random
import multiprocessing
import time
import re
from functools import partial
from tqdm import tqdm

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    return data


def remove_special_characters(in_str):
    out_str = re.sub(r'[a-zA-Z]', '', in_str)
    out_str = re.sub(r'[!@#$%^&*()=+;\[\]\{\},，。]', '', in_str)
    out_str = out_str.strip('.')
    out_str = out_str.replace(' ', '')
    return out_str.strip()


# TODO: Implement the model call yourself
def call_model(messages):
    pass


prompt_template = "{instruction}\nGive your answer ending with \"The answer is x\" where x is your choice.\n\nHere is the question:\n{question}\n\nAnswer: Let's think step by step.\n"

prompt_template_math = "{instruction}\nAdd a line \"The answer is n\" at the end where n is the answer value.\n\nHere is the question:\n{question}\n\nAnswer: Let's think step by step.\n"

# synthesize rationales
def run(item, instruction, dataset_name, max_retries=10):
    
    label = item['label'].strip()
    question = item['question'].strip()

    if dataset_name in ['multiarith', 'asdiv', 'svamp', 'gsm8k']:
        _template = prompt_template_math
    else:
        _template = prompt_template

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': _template.format(instruction=instruction, question=question)}
    ]

    retries = 0
    while retries < max_retries:
        solution_generated = call_model(messages)
        if solution_generated is not None and "The answer is" in solution_generated:
            if dataset_name in ['multiarith', 'asdiv', 'svamp', 'gsm8k']:
                answer_generated = remove_special_characters(solution_generated.split('\n')[-1].split('The answer is')[1].strip())
                if abs(float(answer_generated) - float(label)) < 1e-4:
                    item['model_solution'] = solution_generated
                    return item
            elif dataset_name not in ['multiarith', 'asdiv', 'svamp', 'gsm8k']:
                answer_generated = solution_generated.split('\n')[-1].split('The answer is')[1].strip()
                if answer_generated == label:
                    item['model_solution'] = solution_generated
                    return item
        retries += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--num_threads', type=int, required=False, default=1)
    parser.add_argument('--max_retries', type=int, required=False, default=10)
    args = parser.parse_args()

    raw_data = load_jsonl(args.raw_data_path)
    generated_ids = []

    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding="utf-8") as f:
            generated_ids = [json.loads(line)['id'] for line in f]
    data_to_process = [item for item in raw_data if item['id'] not in generated_ids]
    print(f'{len(data_to_process)} data to process')

    with open('task_instructions.json', 'r', encoding="utf-8") as f:
        instructions = json.load(f)
    
    run_item = partial(run, instruction=instructions[args.dataset_name], dataset_name=args.dataset_name, max_retries=args.max_retries)
    
    with multiprocessing.Pool(processes=args.num_threads) as pool:
        results = pool.map(run_item, data_to_process)

    with open(args.output_path, 'a+', encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result)+'\n')