import collections
from tqdm.auto import tqdm
import numpy as np
import torch
from transformers import default_data_collator, ElectraTokenizerFast

max_seq_length = 384
doc_stride = 128

tokenizer = ElectraTokenizerFast.from_pretrained("monologg/koelectra-base-v3-discriminator")
  