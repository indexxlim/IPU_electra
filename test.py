from transformers import BertTokenizer, BertForMaskedLM, BertForSequenceClassification
import torch



import torch
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", problem_type="multi_label_classification", num_labels=3)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits


import yaml
from easydict import EasyDict
from optimum.graphcore import IPUSeq2SeqTrainingArguments as Seq2SeqTrainingArguments


argus = {}
with open('configurations/electra.yaml') as f:
    hparams = yaml.load_all(f, Loader=yaml.FullLoader)
    for argu in hparams:
        argus[list(argu.keys())[0]]=list(argu.values())[0]



model_args, data_args, training_args = argus['ModelArguments'], argus['DataTrainingArguments'], argus['IPUSeq2SeqTrainingArguments']
model_args, data_args = EasyDict(model_args), EasyDict(data_args)
training_args = Seq2SeqTrainingArguments(**training_args)



from optimum.graphcore.models.bert.modeling_bert import PipelinedBertForMaskedLM, PipelinedBertForSequenceClassification
from optimum.graphcore import IPUConfig


ipu_config = IPUConfig.from_pretrained('configurations/electra_ipu.json')

ipu_model = PipelinedBertForSequenceClassification.from_transformers(model, ipu_config)



ipu_model.parallelize()
if not training_args.fp32:
    ipu_model = ipu_model.half()



# from transformers import (
#     AutoConfig,
#     AutoModelForSequenceClassification,
#     AutoTokenizer,
#     DataCollatorWithPadding,
#     EvalPrediction,
#     HfArgumentParser,
#     PretrainedConfig,
# )

# task_to_keys = {
#     "cola": ("sentence", None),
#     "mnli": ("premise", "hypothesis"),
#     "mrpc": ("sentence1", "sentence2"),
#     "qnli": ("question", "sentence"),
#     "qqp": ("question1", "question2"),
#     "rte": ("sentence1", "sentence2"),
#     "sst2": ("sentence", None),
#     "stsb": ("sentence1", "sentence2"),
#     "wnli": ("sentence1", "sentence2"),
# }


# import datasets
# import numpy as np
# from datasets import load_dataset, load_metric

# raw_datasets = load_dataset("glue", 'mnli', cache_dir=model_args.cache_dir)


# raw_datasets['train'][0]



# label_list = raw_datasets["train"].unique("label")
# label_list.sort()  # Let's sort it for determinism
# num_labels = len(label_list)


# non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]



# sentence1_key, sentence2_key = task_to_keys['mnli']
# padding=False
# max_seq_length = tokenizer.model_max_length
# label_to_id = PretrainedConfig(num_labels=num_labels).label2id




# def preprocess_function(examples):
#     # Tokenize the texts
#     args = (
#         (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
#     )
#     result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

#     # Map labels to IDs (not necessary for GLUE tasks)
#     if label_to_id is not None and "label" in examples:
#         result["label"] = examples["label"]
#         #result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
#     return result

# with training_args.main_process_first(desc="dataset map pre-processing"):
#     raw_datasets = raw_datasets.map(
#         preprocess_function,
#         batched=True,
#         load_from_cache_file=not data_args.overwrite_cache,
#         desc="Running tokenizer on dataset",
#     )


raw_datasets['train'][0]


import poptorch

opts = ipu_config.to_options()
optimizer_class = torch.optim.AdamW
optimizer = optimizer_class(model.parameters(), lr=0.004)

opts.deviceIterations(4)

training_model = poptorch.trainingModel(
    ipu_model.train(), options=opts, optimizer=optimizer
)



train_dataset = raw_datasets['train']



import inspect

signature = inspect.signature(training_model.forward)
_signature_columns = list(signature.parameters.keys())
_signature_columns += ["label", "label_ids"]

columns = [k for k in _signature_columns if k in train_dataset.column_names]
ignored_columns = list(set(train_dataset.column_names) - set(_signature_columns))


# In[26]:


train_dataset = train_dataset.remove_columns(ignored_columns)
data = train_dataset[0]

inputss = tokenizer(["Hello, my dog is cute"]*4, return_tensors="pt")

training_model.compile(**dict(inputss))



inference_model = poptorch.inferenceModel(
    ipu_model.eval(), options=opts
)


inference_model(**dict(inputss))




