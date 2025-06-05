import json
import argparse
import numpy as np
from collections import Counter
from tqdm import tqdm
from copy import deepcopy

# TODO: Implement the model call yourself
def call_model(messages):
    pass


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    return data
    

def get_ece_results(test_data, output_path, rationale, task):
    
    with open('data/task_instructions.json', 'r', encoding='utf-8') as f:
        instruction = json.load(f)[task]

    if task in ['multiarith', 'asdiv', 'svamp', 'gsm8k']:
        _template = instruction + "\nAdd a line \"The answer is n\" at the end where n is the answer value.\n\nHere is the question:\n{question}\n\nAnswer:"
    else:
        _template = instruction + "\nGive your answer ending with \"The answer is x\" where x is your choice.\n\nHere is the question:\n{question}\n\nAnswer:"
    
    if rationale:
        _template = _template + "\nLet's think step by step."

    res_dic = {}

    for i in range(10):
        res_dic[i+1] = {
            'total': 0,
            'correct': 0,
            'incorrect': 0,
            'confidence': []
        }
    
    counter = 0
    with tqdm(total=len(test_data)) as pbar:

        for dic in test_data:
            label = dic['label']
            
            prompt = 'Human:' + _template.format(question=dic['question']) + '\n\nAssistant: '
            message = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}
            ]
              
            answers = []
            _generated_cnt = 0

            while _generated_cnt < 10:
                res = call_model(message)
                _answer = res.split('\n')[-1].split("The answer is")[1].strip()
                answers.append(_answer)
                _generated_cnt += 1

            answers_counted = Counter(answers)
            answer_generated = answers_counted.most_common(1)[0][0]
            confidence = answers_counted.most_common(1)[0][1] / 10.0

            idx = int(confidence * 10.0)
            is_correct = False
            if task in ['multiarith', 'asdiv', 'svamp', 'gsm8k']:
                if abs(float(answer_generated) - float(label)) <= 1e-4:
                    is_correct = True
            elif answer_generated == label:
                    is_correct = True

            if is_correct:
                res_dic[idx]['correct'] += 1
            else:
                res_dic[idx]['incorrect'] += 1

            res_dic[idx]['total'] += 1
            res_dic[idx]['confidence'].append(confidence)
            
            counter += 1
            pbar.update(1)

            # save every 10 samples
            if counter % 10 == 0:
                new_dic = deepcopy(res_dic)
                for j in range(10):
                    avg_conf = np.mean(new_dic[j+1]['confidence']) if new_dic[j+1]['confidence'] != [] else 0
                    new_dic[j+1]['confidence'] = avg_conf

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(new_dic, f, indent=4)
    
    for j in range(10):
        avg_conf = np.mean(res_dic[j+1]['confidence']) if res_dic[j+1]['confidence'] != [] else 0
        res_dic[j+1]['confidence'] = avg_conf

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(res_dic, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--rationale', action='store_true')
    args = parser.parse_args()

    test_data = load_jsonl(args.test_data)
    train_data = load_jsonl(args.train_data)

    get_ece_results(
        test_data=test_data,
        output_file=args.output_file,
        rationale=args.rationale,
        task=args.task,
    )