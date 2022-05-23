
import json
import collections
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler

from transformers import is_datasets_available
import datasets
import poptorch
from poptorch import Options, OutputMode

_is_torch_generator_available = False

class NERDataset(Dataset):
    def __init__(self, data_path):
        data = pd.read_csv(data_path, sep='\t', names = ['text', 'label'])
        
        self.texts = []
        self.labels = []
        for i in range(len(data)):
            word_list = data.iloc[i].text.split()
            self.texts.append(word_list)            
            self.labels = self.labels2id(data.iloc[i].label.split())

    def labels2id(self, labels):
        labels_lst = ["O",
                "PER-B", "PER-I", "FLD-B", "FLD-I", "AFW-B", "AFW-I", "ORG-B", "ORG-I",
                "LOC-B", "LOC-I", "CVL-B", "CVL-I", "DAT-B", "DAT-I", "TIM-B", "TIM-I",
                "NUM-B", "NUM-I", "EVT-B", "EVT-I", "ANM-B", "ANM-I", "PLT-B", "PLT-I",
                "MAT-B", "MAT-I", "TRM-B", "TRM-I"]
        labels_dict = {label:i for i, label in enumerate(labels_lst)}
        id_value=[]
        try:
            for label in labels:
                id_value.append(labels_dict[label])
                print(id_value)
        except:
            raise Exception(f'Not in NER labels : {label}')
        return id_value

    def __getitem__(self, index):
        return {'text':self.texts[index], 'label':self.labels[index]}
        
    def __len__(self):
        return len(self.contexts)

@dataclass
class NERCollator:
    def __init__(self, tokenizer, mapping=None):
        self.tokenizer = tokenizer
        self.text = 'text'
        self.label = 'label'
        self.pad_token_label_id = -100

        labels_lst = ["O",
                "PER-B", "PER-I", "FLD-B", "FLD-I", "AFW-B", "AFW-I", "ORG-B", "ORG-I",
                "LOC-B", "LOC-I", "CVL-B", "CVL-I", "DAT-B", "DAT-I", "TIM-B", "TIM-I",
                "NUM-B", "NUM-I", "EVT-B", "EVT-I", "ANM-B", "ANM-I", "PLT-B", "PLT-I",
                "MAT-B", "MAT-I", "TRM-B", "TRM-I"]
        self.labels_dict = {label:i for i, label in enumerate(labels_lst)}


    def __call__(self, batch):
        if self.text not in batch[0] or self.label not in batch[0]:
            raise Exception("Error: Undefined data keys")

        sentence = [item[self.text] for item in batch]

        source_batch = self.tokenizer.batch_encode_plus(sentence,
                    padding='longest',
                    is_split_into_words=True,
                    max_length=512,
                    truncation=True, 
                    return_offsets_mapping=True,
                    return_tensors='pt')

        labels = [[self.labels_dict[la] for la in item[self.label].split()] for item in batch]
        sequence_length = source_batch.input_ids.shape[1]
        labels = [label + [self.pad_token_label_id] * (sequence_length-len(label)) for label in labels]
        

        return {'input_ids':source_batch.input_ids,
                 'attention_mask':source_batch.attention_mask,
                 'token_type_ids':source_batch.token_type_ids,
                 'labels': labels,
                 'offset_mapping': source_batch.offset_mapping,
                 'sentence': sentence
                 }

def get_dataloader(dataset, collator, batch_size=4, shuffle=False):
    data_loader = DataLoader(dataset, 
                              batch_size=batch_size, 
                              shuffle=shuffle, 
                              collate_fn=collator,
                              num_workers=2)
    return data_loader


class _WorkerInit:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, worker_id):
        np.random.seed((self.seed + worker_id) % np.iinfo(np.uint32).max)

def get_train_sampler(dataset, tokenizer, ipu_config, args) -> Optional[torch.utils.data.Sampler]:
    if not isinstance(dataset, collections.abc.Sized):
        return None
    generator = None
    if _is_torch_generator_available:
        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

    combined_batch_size = args.per_device_train_batch_size * ipu_config.batch_size_factor()

    # Build the sampler.
    if args.group_by_length:
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            lengths = (
                dataset[args.length_column_name]
                if args.length_column_name in train_dataset.column_names
                else None
            )
        else:
            lengths = None
        model_input_name = tokenizer.model_input_names[0] if tokenizer is not None else None

        return LengthGroupedSampler(
            combined_batch_size,
            dataset=dataset,
            lengths=lengths,
            model_input_name=model_input_name,
            generator=generator,
        )

    else:
        if args.complete_last_batch:
            num_examples = len(dataset)
            num_missing_examples = num_examples % combined_batch_size
            if num_missing_examples > 0:
                indices = torch.cat(
                    [torch.arange(num_examples), torch.randint(0, num_examples, size=(num_missing_examples,))]
                )
            return SubsetRandomSampler(indices, generator)
        return RandomSampler(dataset)


def ipu_dataloader(dataset, tokenizer, ipu_config, args ,collator, shuffle=True) -> poptorch.DataLoader:

    opts = ipu_config.to_options()

    if isinstance(dataset, torch.utils.data.IterableDataset):
        return poptorch.DataLoader(
            opts,
            dataset,
            batch_size=args.train_batch_size,
            collate_fn=collator,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
            **poptorch_specific_kwargs,
        )


    should_drop_last = not hasattr(collator, "__wrapped__") and not args.complete_last_batch
    poptorch_specific_kwargs = {
        "drop_last": should_drop_last,  # Not dropping last will end up causing NaN during training if the combined batch size does not divide the number of steps
        "auto_distributed_partitioning": not isinstance(dataset, torch.utils.data.IterableDataset),
        "mode": args.dataloader_mode,
        "worker_init_fn": _WorkerInit(123),
    }



    train_sampler = get_train_sampler(dataset, tokenizer, ipu_config, args)


    combined_batch_size = args.per_device_train_batch_size * ipu_config.batch_size_factor()
    rebatched_worker_size = (
        2 * (combined_batch_size // args.dataloader_num_workers)
        if args.dataloader_num_workers
        else combined_batch_size
    )

    return poptorch.DataLoader(
        opts,
        dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler if shuffle else None,
        collate_fn=collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
        rebatched_worker_size=rebatched_worker_size,
        **poptorch_specific_kwargs,
    )