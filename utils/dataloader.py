import numpy as np
import torch
import torch.nn as nn
from torchtext.data import Field, Dataset, Example

def tokenize(s:str):
    tok = s.rstrip().replace("，", "#").replace("。", "#")
    return list(tok)

sentence_field = Field(sequential = True, tokenize = tokenize, lower = False,
                       batch_first = True, include_lengths = True)
label_field = Field(sequential = False, use_vocab = False)