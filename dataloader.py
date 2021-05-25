import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from collections import Counter
from torchtext.vocab import Vocab
import random

class poet_dataset():
    def __init__(self, data_path, train_batch_size, eval_batch_size):
        counter = Counter()
        all_sents = []

        with open(data_path, 'r', encoding = 'utf-8-sig') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                full = self.tokenizer(line)
                counter.update(full)
                all_sents.append(full)
        
        self.vocab = Vocab(counter)
        self.len = len(all_sents)
        self.ntoken = len(self.vocab.itos)

        random.shuffle(all_sents)
        train_sz, val_sz, test_sz = int(0.6*self.len), int(0.2*self.len), int(0.2*self.len)
        train_data = self.data_process(all_sents[:train_sz])
        val_data = self.data_process(all_sents[train_sz: train_sz + val_sz])
        test_data = self.data_process(all_sents[train_sz + val_sz:])

        #print(train_data[0].shape)

        self.train_data = self.batchify(train_data, train_batch_size)
        self.val_data = self.batchify(val_data, eval_batch_size)
        self.test_data = self.batchify(test_data, eval_batch_size)

        #print(self.train_data[0].shape)

        
    def tokenizer(self, s:str):
        tok = s.strip().replace("，", "#").replace("。", "#").split("#")[:-1]
        self.n_sents = len(tok)
        self.n_words = len(tok[0])
        full = []
        for sent in tok:
            lsent = list(sent)
            full.extend(lsent)
            full.append('#')
        full.append('*')
        return full
    
    def data_process(self, s:list):
        slen = len(s)
        x, y = [0] * slen, [0] * slen

        for i,sent in enumerate(s):
            numeric = [self.vocab[word] for word in sent]
            x[i] = torch.tensor(numeric[:-1], dtype = torch.long).unsqueeze(0)
            y[i] = torch.tensor(numeric[1: ], dtype = torch.long).unsqueeze(0)
        
        xs = torch.cat(x, dim = 0)
        ys = torch.cat(y, dim = 0)

        return xs, ys
    
    def batchify(self, data, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x, y = data
        n_batch = x.shape[0] // batch_size
        x = x.narrow(0, 0, n_batch * batch_size)
        y = y.narrow(0, 0, n_batch * batch_size)
        x = x.view(n_batch, batch_size, -1).contiguous()
        y = y.view(n_batch, batch_size, -1).contiguous()
        return x.to(device), y.to(device)
    
    def get_batch(self, data, index):
        input = data[0][index]
        target = data[1][index].reshape(-1)
        return input, target
    
    def vec2sent(self, s:list):
        sent = [self.vocab.itos[num] for num in s]
        return ''.join(sent)
    
    def info(self):
        return self.ntoken, self.n_sents, self.n_words
    
    def Test(self):
        print(self.vocab.itos)
