import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from collections import Counter
from torchtext.vocab import Vocab
import random
import math

class poet_dataset():
    def __init__(self, data_path, train_batch_size = 50, eval_batch_size = 40):
        counter = Counter()
        df = Counter()
        all_sents = []

        self.n_poet = 0

        with open(data_path, 'r', encoding = 'utf-8-sig') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.n_poet += 1
                full = self.tokenizer(line)
                counter.update(full)
                df.update(set(full))
                all_sents.append(full)
        
        self.vocab = Vocab(counter)
        self.len = len(all_sents)
        self.ntoken = len(self.vocab.itos)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        random.shuffle(all_sents)
        train_sz, val_sz, test_sz = int(0.7*self.len), int(0.15*self.len), int(0.15*self.len)
        self.ori_train_data = self.data_process(all_sents[:train_sz])
        self.ori_val_data = self.data_process(all_sents[train_sz: train_sz + val_sz])
        self.ori_test_data = self.data_process(all_sents[train_sz + val_sz:])

        self.train_data = self.batchify(self.ori_train_data, train_batch_size)
        self.val_data = self.batchify(self.ori_val_data, eval_batch_size)
        self.test_data = self.batchify(self.ori_test_data, eval_batch_size)

        twords = sum(counter.values())
        self.tf_idf = [0] * self.ntoken
        for i,word in enumerate(self.vocab.itos):
            tf = counter[word] / twords
            idf = math.log(self.n_poet / (df[word] + 1))
            self.tf_idf[i] = tf * idf
        self.tf_idf = torch.tensor(self.tf_idf)

    def tokenizer(self, s:str):
        tok = s.strip().replace("，", "#").replace("。", "#").split("#")[:-1]
        self.n_sents = len(tok)
        self.n_words = len(tok[0])
        full = []
        for sent in tok:
            lsent = list(sent)
            full.extend(lsent)
            full.append('#')
        return full
    
    def shuffle(self):
        random.shuffle(self.ori_train_data)
        #random.shuffle(self.ori_val_data)
        #random.shuffle(self.ori_test_data)
        self.train_data = self.batchify(self.ori_train_data, self.train_batch_size)
        #self.val_data = self.batchify(self.ori_val_data, self.eval_batch_size)
        #self.test_data = self.batchify(self.ori_test_data, self.eval_batch_size)
    
    def head2vec(self, s:str):
        if s == '。' or s == '，':
            s = '#'
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
    
    def get_batch(self, data, index, batch_first = True, flat_target = True):
        input = data[0][index]
        target = data[1][index]
        if not batch_first:
            input = input.T
            target = target.T
        if flat_target:
            target = target.reshape(-1)
        return input, target
    
    def num2word(self, idx:int):
        return self.vocab.itos[idx]
    
    def info(self):
        return self.ntoken, self.n_sents, self.n_words
    
    def Test(self):
        print(self.vocab.itos)
