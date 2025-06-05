#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import random
import logging
from dataclasses import dataclass, field
from datasets import load_dataset
from typing import Optional, Dict, Sequence

import json
import os
os.environ["WANDB_DISABLED"] = "true"
import torch
import transformers
from transformers import Trainer
from transformers.utils import is_datasets_available
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import seed_worker
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm import tqdm

if is_datasets_available():
    import datasets

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    task_name: str = field(default=None, metadata={"help": "Task name."})
    rationale: bool = field(default=True, metadata={"help": "Whether use rationale to train."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    return data


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:

    data_files = {'train': data_args.data_path}
    extension = data_args.data_path.split(".")[-1]
    extension = 'json' if extension == 'jsonl' else extension
    raw_dataset = load_dataset(extension, data_files=data_files)
    column_names = raw_dataset['train'].column_names

    instruction = json.load(open('task_instructions.json', 'r', encoding='utf-8'))[data_args.task_name]

    def _preprocess(examples):
        if data_args.task_name not in ['multiarith', 'asdiv', 'svamp', 'gsm8k']:
            if data_args.rationale:
                sources = [
                    'Human: ' + f"{instruction}\nGive your answer ending with \"The answer is x\" where x is your choice.\n\nHere is the question:\n{question}\n\nAnswer: Let's think step by step." + '\n\nAssistant: '
                    for question in examples['question']
                ]
                targets = [f"{solution}{tokenizer.eos_token}" for solution in examples['model_solution']]
            else:
                sources = [
                    'Human: ' + f"{instruction}\nGive your answer ending with \"The answer is x\" where x is your choice.\n\nHere is the question:\n{question}\n\nAnswer:" + '\n\nAssistant: '
                    for question in examples['question']
                ]
                targets = [f"The answer is {answer}{tokenizer.eos_token}" for answer in examples['label']]
        else:
            if data_args.rationale:
                sources = [
                    'Human: ' + f"{instruction}\nAdd a line \"The answer is n\" at the end where n is the answer value.\n\nHere is the question:\n{question}\n\nAnswer: Let's think step by step." + '\n\nAssistant: '
                    for question in examples['question']
                ]
                targets = [f"{solution}{tokenizer.eos_token}" for solution in examples['model_solution']]
            else:
                sources = [
                    'Human: ' + f"{instruction}\nAdd a line \"The answer is n\" at the end where n is the answer value.\n\nHere is the question:\n{question}\n\nAnswer:" + '\n\nAssistant: '
                    for question in examples['question']
                ]
                targets = [f"The answer is {answer}{tokenizer.eos_token}" for answer in examples['label']]

        src_tokenized = tokenizer(
            sources,
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        src_len = [torch.LongTensor(_src_tokenized).ne(tokenizer.pad_token_id).sum().item() for _src_tokenized in src_tokenized["input_ids"]]
        
        tgt_tokenized = tokenizer(
            targets,
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        input_ids = [torch.cat((torch.LongTensor(_src_tokenized), torch.LongTensor(_tgt_tokenized[1:]))) 
                        for _src_tokenized, _tgt_tokenized in zip(src_tokenized["input_ids"], tgt_tokenized["input_ids"])]
        labels = copy.deepcopy(input_ids)

        for label, source_len in zip(labels, src_len):
            label[:source_len] = IGNORE_INDEX

        input_id_truncated, label_truncated = [], []
        for _ids, _labels in zip(input_ids, labels):
            if len(_ids) >= tokenizer.model_max_length:
                _id_truncated = _ids[:tokenizer.model_max_length]
                _id_truncated[-1] = tokenizer.eos_token_id
                input_id_truncated.append(_id_truncated)
            else:
                input_id_truncated.append(_ids)
            if len(_labels) >= tokenizer.model_max_length:
                _label_truncated = _labels[:tokenizer.model_max_length]
                _label_truncated[-1] = tokenizer.eos_token_id
                label_truncated.append(_label_truncated)
            else:
                label_truncated.append(_labels)

        return dict(input_ids=input_id_truncated, labels=label_truncated)   

    # preprocess
    train_dataset = raw_dataset['train']
    train_dataset = train_dataset.map(
        _preprocess,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc='Running tokenizer on train dataset'
    )
    train_dataset.set_format(type='torch')

    if torch.distributed.get_rank() == 0:
        print("=" * 20 + "Sampled inputs:" + "=" * 20)
        _sample = train_dataset.__getitem__(random.choice(range(train_dataset.__len__())))
        print(tokenizer.decode(_sample["input_ids"], skip_special_tokens=True))
        print(tokenizer.decode(torch.where(_sample["labels"]== IGNORE_INDEX, tokenizer.pad_token_id, _sample["labels"])))
        print("=" * 50)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

@record
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parser.add_argument("--local-rank", type=int, default=0)
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True
    )
    config.use_cache = False
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        legacy=False,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
