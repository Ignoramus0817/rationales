import torch
import json
import os
import argparse
import random
import sys
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


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    return data


def format_train_sample(test_tuple, train_data, rationale, icl_prompt, no_rationale_icl_prompt, task):
    demos = random.sample(train_data, 3)
    if task == 'cb' or task == 'anli':
        p_1, h_1, a_1, l_1 = demos[0]['premise'], demos[0]['hypothesis'], demos[0]['answer'], demos[0]['label']
        p_2, h_2, a_2, l_2 = demos[1]['premise'], demos[1]['hypothesis'], demos[1]['answer'], demos[1]['label']
        p_3, h_3, a_3, l_3 = demos[2]['premise'], demos[2]['hypothesis'], demos[2]['answer'], demos[2]['label']
    else:
        t_1, a_1, l_1 = demos[0]['text'], demos[0]['answer'], demos[0]['label']
        t_2, a_2, l_2 = demos[1]['text'], demos[1]['answer'], demos[1]['label']
        t_3, a_3, l_3 = demos[2]['text'], demos[2]['answer'], demos[2]['label']
    
    if rationale is True:
        if task == 'cb' or task == 'anli':
            full_prompt = icl_prompt.format(p_1=p_1, h_1=h_1, a_1=a_1,
                                            p_2=p_2, h_2=h_2, a_2=a_2,
                                            p_3=p_3, h_3=h_3, a_3=a_3,
                                            premise=test_tuple[0], hypothesis=test_tuple[1])
        else:
            full_prompt = icl_prompt.format(t_1=t_1, a_1=a_1,
                                            t_2=t_2, a_2=a_2,
                                            t_3=t_3, a_3=a_3,
                                            text=test_tuple[0])
    else:
        if task == 'cb' or task == 'anli':
            full_prompt = no_rationale_icl_prompt.format(p_1=p_1, h_1=h_1, a_1='The answer is '+ l_1,
                                            p_2=p_2, h_2=h_2, a_2='The answer is '+ l_2,
                                            p_3=p_3, h_3=h_3, a_3='The answer is '+ l_3,
                                            premise=test_tuple[0], hypothesis=test_tuple[1])
        else:
            full_prompt = no_rationale_icl_prompt.format(t_1=t_1, a_1='The answer is '+ l_1,
                                            t_2=t_2, a_2='The answer is '+ l_2,
                                            t_3=t_3, a_3='The answer is '+ l_3,
                                            text=test_tuple[0])
    return full_prompt
    

def get_ece_results(model_path, test_data, train_data, output_file, rationale, task, info_file, demo_num=3):
    
    with open(info_file, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    rationale_prompt = dataset_info[task]['train_prompt']
    no_rationale_prompt = dataset_info[task]['NR_train_prompt']
    icl_prompt = dataset_info[task]['ICL_prompt']
    no_rationale_icl_prompt = dataset_info[task]['NR_ICL_prompt']
    options = list(dataset_info[task]['labels'].keys())

    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    # model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    # model.bfloat16()
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
    with tqdm(total=len(test_data)) as pbar:

        for dic in test_data:
            label = dic['label']
            
            if demo_num == 0:
                if rationale is False:
                    if task == 'cb' or task == 'anli':
                        prompt = 'Human:' + no_rationale_prompt.format(premise=dic['premise'], hypothesis=dic['hypothesis']) + '\n\nAssistant: '
                    else:
                        prompt = 'Human:' + no_rationale_prompt.format(text=dic['text']) + '\n\nAssistant: '
                else:
                    if task == 'cb' or task == 'anli':
                        prompt = 'Human:' + rationale_prompt.format(premise=dic['premise'], hypothesis=dic['hypothesis']) + '\n\nAssistant: '
                    else:
                        prompt = 'Human:' + rationale_prompt.format(text=dic['text']) + '\n\nAssistant: '
            else:
                if task == 'cb' or task == 'anli':
                    prompt = format_train_sample([dic['premise'], dic['hypothesis']], train_data, rationale, icl_prompt, no_rationale_icl_prompt, task)
                else:
                    prompt = format_train_sample([dic['text']], train_data, rationale, icl_prompt, no_rationale_icl_prompt, task)

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
                        max_new_tokens=1024,
                        temperature=0.8,
                        do_sample=True,
                    )
                else:
                    outputs = model.generate(
                        input_ids=inputs.input_ids,
                        max_new_tokens=1024,
                        temperature=0.8,
                        do_sample=True,
                        stopping_criteria = [EosListStoppingCriteria()]
                    )

                res = tokenizer.decode(outputs[0], skip_special_tokens=True).strip() 
                try:
                    _answer_line = res.split('\n')[-1]
                    if "The answer is" in _answer_line:
                        _answer = _answer_line.split("The answer is")[1].strip()
                    else:
                        _answer = _answer_line.split("The answer")[1].strip()
                    answers.append(_answer[0])
                    _generate_success += 1
                except:
                    if _tried_count > 20:
                        failed_flag = True
                        _generate_success += 1
                        break
            
            if failed_flag is True:
                fail_counter += 1
                print(fail_counter)
                continue

            answers_counted = Counter(answers)
            answer_generated = answers_counted.most_common(1)[0][0]
            confidence = answers_counted.most_common(1)[0][1] / 10.0

            idx = int(confidence * 10.0)

            if answer_generated == label:
                res_dic[idx]['correct'] += 1
            else:
                res_dic[idx]['incorrect'] += 1

            res_dic[idx]['total'] += 1
            res_dic[idx]['confidence'].append(confidence)
            
            counter += 1
            pbar.update(1)

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
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--info_file', type=str, required=True)
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
        task=args.task,
        info_file=args.info_file,
        demo_num=args.demo_num,
    )