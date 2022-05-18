
import json
import collections
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
from logging import logger

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler

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

    def __call__(self, batch):
        if self.text not in batch[0] or self.label not in batch[0]:
            raise Exception("Error: Undefined data keys")

        sentence = [item[self.text]+self.tokenizer.eos_token for item in batch]

        source_batch = self.tokenizer.batch_encode_plus(sentence,
                    padding='longest',
                    max_length=512,
                    truncation=True, 
                    return_tensors='pt')
       

            
        return {'input_ids':source_batch.input_ids,
                 'attention_mask':source_batch.attention_mask,
                 'token_type_ids':source_batch.token_type_ids,
                 'labels': labels,
                 }
    
    def custom_encode_plus(self, sentence, tokenizer, return_tensors=None):

        words = sentence.split()

        tokens = []
        tokens_mask = []

        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            tokens_mask.extend([1] + [0] * (len(word_tokens) - 1))

        ids = tokenizer.convert_tokens_to_ids(tokens)
        len_ids = len(ids)
        total_len = len_ids + tokenizer.num_special_tokens_to_add()
        if tokenizer.model_max_length and total_len > tokenizer.model_max_length:
            ids, _, _ = tokenizer.truncate_sequences(
                ids,
                pair_ids=None,
                num_tokens_to_remove=total_len - tokenizer.model_max_length,
                truncation_strategy="longest_first",
                stride=0,
            )

        sequence = tokenizer.build_inputs_with_special_tokens(ids)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(ids)
        # HARD-CODED: As I know, most of the transformers architecture will be `[CLS] + text + [SEP]``
        #             Only way to safely cover all the cases is to integrate `token mask builder` in internal library.
        tokens_mask = [1] + tokens_mask + [1]
        words = [tokenizer.cls_token] + words + [tokenizer.sep_token]

        encoded_inputs = {}
        encoded_inputs["input_ids"] = sequence
        encoded_inputs["token_type_ids"] = token_type_ids


        encoded_inputs["input_ids"] = torch.tensor([encoded_inputs["input_ids"]])

        if "token_type_ids" in encoded_inputs:
            encoded_inputs["token_type_ids"] = torch.tensor([encoded_inputs["token_type_ids"]])

        if "attention_mask" in encoded_inputs:
            encoded_inputs["attention_mask"] = torch.tensor([encoded_inputs["attention_mask"]])

        elif return_tensors is not None:
            logger.warning(
                "Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(
                    return_tensors
                )
            )

        return encoded_inputs, words, tokens_mask
        
        

        
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
