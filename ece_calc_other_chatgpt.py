import json
import argparse
import random
import requests
import time
import numpy as np
import multiprocessing
from collections import Counter
from functools import wraps
from tqdm import tqdm
from copy import deepcopy

url = "https://api.openai.one/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-rCog8I4naTdG94Ib9a7e4fAd34Eb4586A573A7C879F4D897"
}


def retry(max_attempts=3, delay=5):
    """
    A decorator for retrying a function call with a specified delay in case of exception.

    :param max_attempts: The maximum number of attempts. Default is 3.
    :param delay: The delay (in seconds) between attempts. Default is 1.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    print(f"Attempt {attempts}/{max_attempts} failed: {e}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


@retry(max_attempts=3, delay=10)
def ask_gpt(messages):
    data = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.8,
        "top_p": 1,
        "presence_penalty": 1,
        "max_tokens": 2048,
        "messages": messages
    }

    response = requests.post(url, headers=headers, data=json.dumps(data).encode('utf-8'))
    result = json.loads(response.content.decode("utf-8"))
    result = result['choices'][0]["message"]["content"].strip()

    return result


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
    

def get_ece_results(test_data, train_data, output_file, rationale, task, demo_num=3, num_threads=5):
    
    with open('data/dataset_info.json', 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    rationale_prompt = dataset_info[task]['train_prompt']
    no_rationale_prompt = dataset_info[task]['NR_train_prompt']
    icl_prompt = dataset_info[task]['ICL_prompt']
    no_rationale_icl_prompt = dataset_info[task]['NR_ICL_prompt']
    options = list(dataset_info[task]['labels'].keys())

    res_dic = {}

    for i in range(10):
        res_dic[i+1] = {
            'total': 0,
            'correct': 0,
            'incorrect': 0,
            'confidence': []
        }
    
    counter = 0
    for dic in tqdm(test_data):
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
        
        input_messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ]

        answers = []
        _generate_success = 0
        _tried_count = 0
        failed_flag = None
        while _generate_success < 10:
            _num_threads = num_threads if num_threads <= 10-_generate_success else 10-_generate_success
            with multiprocessing.Pool(processes=_num_threads) as pool:
                generated = pool.map(ask_gpt, [input_messages]*_num_threads)

            for _res in generated:
                _tried_count += 1
                try:
                    _answer_line = _res.split('\n')[-1]
                    _answer = _answer_line.split("The answer is")[1].strip()
                    answers.append(_answer[0])
                    _generate_success += 1
                    if _generate_success >= 10:
                        break
                except:
                    if _tried_count > 20:
                        failed_flag = True
                        _generate_success += 1
                        break
        
        if failed_flag is True:
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
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--demo_num', type=int, required=True)
    parser.add_argument('--rationale', action='store_true')
    parser.add_argument('--num_threads', type=int, default=5)
    args = parser.parse_args()

    test_data = load_jsonl(args.test_data)
    train_data = load_jsonl(args.train_data)

    get_ece_results(
        test_data=test_data,
        train_data=train_data,
        output_file=args.output_file,
        rationale=args.rationale,
        task=args.task,
        demo_num=args.demo_num,
        num_threads=args.num_threads
    )