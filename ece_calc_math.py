import torch
import json
import os
import re
import argparse
import random
import numpy as np
from collections import Counter
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria

class EosListStoppingCriteria(StoppingCriteria):
    # def __init__(self, eos_sequence = [29871, 13, 13, 29984, 29901, 29871]):
    def __init__(self, eos_sequence = [13, 13]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids


def remove_special(in_str):
    out_str = re.sub(r'[a-zA-Z]', '', in_str)
    out_str = re.sub(r'[!@#$%^&*()=+;\[\]\{\},，。]', '', in_str)
    out_str = out_str.strip('.')
    out_str = out_str.replace(' ', '')
    return out_str.strip()


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    return data


def format_train_sample(question, train_data, rationale, demo_num):
    prefix = 'Here are some math problems and their solutions:'
    if rationale is True:
        question += "\nLet's think step by step"
    demos = random.sample(train_data, demo_num)
    demo_body = ''
    for demo in demos:
        if rationale is True:
            demo_body += f'\n\nQ: {demo["question"]}\nA: {demo["answer"]}'
        else:
            demo_body += f'\n\nQ: {demo["question_raw"]}\nA: {demo["answer_raw"]}'
    
    full_prompt = prefix + demo_body + f'\n\nQ: {question}\nA: '
    return full_prompt
    

def get_ece_results(model_path, test_data, train_data, output_file, rationale, demo_num=3):

    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.cuda()
    model.eval()

    res_dic = {}

    for i in range(10):
        res_dic[i+1] = {
            'total': 0,
            'correct': 0,
            'incorrect': 0,
            'confidence': []
        }
    
    counter = 0
    fail_counter = 0
    for dic in tqdm(test_data):
        question = dic['question'].strip()
        answer = dic['answer']
        answer =  remove_special(answer) if isinstance(answer, str) else answer
        try:
            answer = float(answer)
        except:
            continue
        
        if demo_num == 0:
            if rationale is False:
                prompt = 'Human: Solve the following math problem. \n' + question + '\n\nAssistant: '
            else:
                # prompt = 'Human: Solve the following math problem. \n' + question + "\nLet's think step by step" + '\n\nAssistant: '
                prompt = 'Human: Solve the following math problem. \n' + question + " Let's think step by step" + '\n\nAssistant: '
        else:
            prompt = format_train_sample(question, train_data, rationale, demo_num)

        inputs = tokenizer(prompt, max_length=2048, truncation=True, return_tensors="pt", padding=False, return_token_type_ids=False)
        if 'qwen' in model_path:
            for _l in inputs.input_ids:
                _l = torch.cat((torch.LongTensor([tokenizer.bos_token_id]), _l))
        inputs = inputs.to('cuda')

        answers = []
        _generate_success = 0
        _tried_count = 0
        failed_flag = None
        while _generate_success < 10:
            _tried_count += 1
            if demo_num == 0:
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    max_new_tokens=512,
                    temperature=0.8,
                    do_sample=True,
                )
            else:
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    max_new_tokens=512,
                    temperature=0.8,
                    do_sample=True,
                    stopping_criteria = [EosListStoppingCriteria()]
                )

            res = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            try:
                _answer_line = res.split('\n')[-1]
                if 'The answer is' in _answer_line:
                    _answer = _answer_line.split("The answer is")[-1].strip()
                else:
                    _answer = _answer_line.split("A:")[-1].strip()
                _answer = remove_special(_answer)
                answers.append(float(_answer))
                _generate_success += 1
            except:
                if _tried_count > 20:
                    failed_flag = True
                    _generate_success += 1
        
        if failed_flag is True:
            fail_counter += 1
            print(fail_counter)
            continue
        
        answers_counted = Counter(answers)
        answer_generated = answers_counted.most_common(1)[0][0]
        confidence = answers_counted.most_common(1)[0][1] / 10.0

        idx = int(confidence * 10.0)

        if abs(float(answer_generated) - float(answer)) <= 1e-4:
            res_dic[idx]['correct'] += 1
        else:
            res_dic[idx]['incorrect'] += 1

        res_dic[idx]['total'] += 1
        res_dic[idx]['confidence'].append(confidence)
        
        counter += 1

        if counter % 10 == 0:
            new_dic = deepcopy(res_dic)
            for j in range(10):
                avg_conf = np.mean(new_dic[j+1]['confidence']) if new_dic[j+1]['confidence'] != [] else 0
                new_dic[j+1]['confidence'] = avg_conf

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(new_dic, f, indent=4)
    
    for j in range(10):
        avg_conf = np.mean(res_dic[j+1]['confidence']) if res_dic[j+1]['confidence'] != [] else 0
        res_dic[j+1]['confidence'] = avg_conf

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(res_dic, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--demo_num', type=int, required=True)
    parser.add_argument('--rationale', action='store_true')
    args = parser.parse_args()

    test_data = load_jsonl(args.test_data)
    train_data = load_jsonl(args.train_data)

    get_ece_results(
        model_path=args.model_path,
        test_data=test_data,
        train_data=train_data,
        output_file=args.output_file,
        rationale=args.rationale,
        demo_num=args.demo_num,
    )