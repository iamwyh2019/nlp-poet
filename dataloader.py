import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from collections import Counter
from torchtext.vocab import Vocab
import random

class poet_dataset():
    def __init__(self, data_path, train_batch_size = 50, eval_batch_size = 40):
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
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        random.shuffle(all_sents)
        train_sz, val_sz, test_sz = int(0.6*self.len), int(0.2*self.len), int(0.2*self.len)
        self.ori_train_data = self.data_process(all_sents[:train_sz])
        self.ori_val_data = self.data_process(all_sents[train_sz: train_sz + val_sz])
        self.ori_test_data = self.data_process(all_sents[train_sz + val_sz:])

        #print(train_data[0].shape)

        self.train_data = self.batchify(self.ori_train_data, train_batch_size)
        self.val_data = self.batchify(self.ori_val_data, eval_batch_size)
        self.test_data = self.batchify(self.ori_test_data, eval_batch_size)

        #print(self.train_data[0].shape)

    def tokenizer(self, s:str):
        self.n_sents = s.count("，") + s.count("。")
        self.n_words = s.find("，")
        tok = list(s)
        tok.append('#')
        return tok
    
    def shuffle(self):
        random.shuffle(self.ori_train_data)
        self.train_data = self.batchify(self.ori_train_data, self.train_batch_size)
    
    def head2vec(self, s:str):
        numeric = self.vocab[s]
        ts = torch.tensor(numeric, dtype = torch.long).view(1,1)
        return ts
    
    def data_process(self, s:list):
        slen = len(s)
        ts = [0] * slen

        for i,sent in enumerate(s):
            numeric = [self.vocab[word] for word in sent]
            x = torch.tensor(numeric[:-1], dtype = torch.long).unsqueeze(0)
            y = torch.tensor(numeric[1: ], dtype = torch.long).unsqueeze(0)
            ts[i] = (x,y)

        return ts
    
    def batchify(self, data, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.cat([s[0] for s in data], dim = 0)
        y = torch.cat([s[1] for s in data], dim = 0)
        n_batch = x.shape[0] // batch_size
        x = x.narrow(0, 0, n_batch * batch_size)
        y = y.narrow(0, 0, n_batch * batch_size)
        x = x.view(n_batch, batch_size, -1).contiguous()
        y = y.view(n_batch, batch_size, -1).contiguous()
        return x.to(device), y.to(device)
    
    def get_batch(self, data, index, batch_first = False, target_flatten = True):
        input = data[0][index]
        target = data[1][index]
        if not batch_first:
            input = input.T
            target = target.T
        if target_flatten:
            target = target.reshape(-1)
        return input, target
    
    def num2word(self, idx:int):
        return self.vocab.itos[idx]
    
    def list2sent(self, s:list):
        sent = [self.vocab.itos[num] for num in s]
        return ''.join(sent)
    
    def info(self):
        return self.ntoken, self.n_sents, self.n_words
    
    def Test(self):
        print(self.vocab.itos)
