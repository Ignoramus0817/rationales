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


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    task_name: str = field(default=None, metadata={"help": "Task name."})
    rationale: bool = field(default=True, metadata={"help": "Whether use rationale to train."})
    rationale_sub: Optional[str] = field(default=None, metadata={"help": "Whether substitute rationales with padding or random tokens"})
    annotator: Optional[str] = field(default=None, metadata={"help": "Whether substitute rationales with padding or random tokens"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    shuffle: Optional[bool] = field(default=True)


class NonShuffleTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self._train_batch_size,
                shuffle=False,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            shuffle=False
        )

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            # shuffle=False,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
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

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")

        data_dict = torch.load(data_path)
        logging.warning("Total data length: {}".format(len(data_dict["input_ids"])))

        # data_dict = torch.load(data_path)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


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
    """Make dataset and collator for supervised fine-tuning."""

    if data_args.annotator is not None:
        with open(f'data/dataset_info_{data_args.annotator}.json', 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
    else:
        with open('data/dataset_info.json', 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)

    # load data
    # train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_files = {'train': data_args.data_path}
    extension = data_args.data_path.split(".")[-1]
    extension = 'json' if extension == 'jsonl' else extension
    raw_dataset = load_dataset(extension, data_files=data_files)
    column_names = raw_dataset['train'].column_names
    if 'gsm8k' not in data_args.task_name:
        if data_args.rationale:
            if data_args.task_name == 'sent':
                prompt_template = dataset_info['cr']['train_prompt']
            elif data_args.task_name == 'nli':
                prompt_template = dataset_info['cb']['train_prompt']
            elif data_args.task_name == 'paraphrase':
                prompt_template = dataset_info['paws']['train_prompt']
            elif data_args.task_name == 'arc_ez':
                prompt_template = dataset_info['arc']['train_prompt']
            else:
                prompt_template = dataset_info[data_args.task_name]['train_prompt'] 
        else:
            if data_args.task_name == 'sent':
                prompt_template = dataset_info['cr']['NR_train_prompt']
            elif data_args.task_name == 'nli':
                prompt_template = dataset_info['cb']['NR_train_prompt']
            elif data_args.task_name == 'paraphrase':
                prompt_template = dataset_info['paws']['NR_train_prompt']
            elif data_args.task_name == 'arc_ez':
                prompt_template = dataset_info['arc']['NR_train_prompt']
            else:
                prompt_template = dataset_info[data_args.task_name]['NR_train_prompt'] 

    def _preprocess(examples):
        if 'gsm8k' not in data_args.task_name:
            if data_args.task_name != 'nli':
                sources = [
                    'Human: ' + prompt_template.format(text=_t) + '\n\nAssistant: '
                    for _t in examples['text']
                ]
            else:
                sources = [
                    'Human: ' + prompt_template.format(premise=_p, hypothesis=_h) + '\n\nAssistant: '
                    for _p, _h in zip(examples['premise'], examples['hypothesis'])
                ]
        
            if data_args.rationale:
                targets = [f"{answer}" for answer in examples['answer']]
            else:
                targets = [f"The answer is {answer}" for answer in examples['label']]
        else:
            if data_args.rationale:
                sources = [
                    'Human: Solve the following math problem. \n' + question.strip() + '\n\nAssistant: '
                    for question in examples['question']
                ]
                targets = [f"{answer}" for answer in examples['answer']]
            else:
                sources = [
                    'Human: Solve the following math problem. \n' + question.strip() + '\n\nAssistant: '
                    for question in examples['question_raw']
                ]
                targets = [f"{answer}" for answer in examples['answer_raw']]

        src_tokenized = tokenizer(
            sources,
            # return_tensors="pt",
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        src_len = [torch.LongTensor(_src_tokenized).ne(tokenizer.pad_token_id).sum().item()+1 for _src_tokenized in src_tokenized["input_ids"]]
        
        if data_args.rationale_sub is None:
            tgt_tokenized = tokenizer(
                targets,
                # return_tensors="pt",
                padding=False,
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
        else:
            _pad_str = ''
            if data_args.task_name == 'winogrande':
                _pad_num = 32
            elif data_args.task_name == 'nli':
                _pad_num = 45
            elif data_args.task_name == 'paraphrase':
                _pad_num = 38
            elif data_args.task_name == 'wic':
                _pad_num = 55
            elif data_args.task_name == 'creak':
                _pad_num = 54
            for counter in range(_pad_num):
                _pad_str += f'<Think_{counter}>'
                counter += 1
            tgt_answer = [_s.split('\n')[-1] for _s in targets]
            target_full = [_pad_str + _a for _a in tgt_answer]

            tgt_tokenized = tokenizer(
                target_full,
                padding=False,
                max_length=tokenizer.model_max_length,
                truncation=True
            )

        # input_ids = torch.cat((src_tokenized["input_ids"], tgt_tokenized["input_ids"]))
        input_ids = [torch.cat((torch.LongTensor([tokenizer.bos_token_id]), torch.LongTensor(_src_tokenized), torch.LongTensor(_tgt_tokenized), torch.LongTensor([tokenizer.eos_token_id]))) 
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
        # print(input_id_truncated[0])
        # import sys
        # sys.exit(0)
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

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        trust_remote_code=True
    )
    tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.bos_token_id = tokenizer.im_start_id
    tokenizer.eos_token_id = tokenizer.im_end_id

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
