import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from dataloader import poet_dataset
from model import RNNModel
import time

data_path = 'data/wuyanjueju.txt'
train_batch_size = 50
eval_batch_size = 40
epochs = 100
input_size = 300
hidden_size = 300
n_layers = 5
clip = 0.1

dataset = poet_dataset(data_path, train_batch_size, eval_batch_size)
tokens, n_sents, n_words = dataset.info()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RNNModel(
    voc_size = tokens,
    input_size = input_size,
    hidden_size = hidden_size,
    n_layers = n_layers,
    dropout = None
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
lr = 0.01

def train(model):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    log_step = 20

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    hidden = None

    n_batch = dataset.train_data[0].shape[0]

    for i in range(n_batch):
        input, target = dataset.get_batch(dataset.train_data, i)

        optimizer.zero_grad()

        output, hidden = model(input, hidden)
        hidden = model.detach_hidden(hidden)

        loss = criterion(output, target)
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        if i % log_step == 0 and i > 0:
            avg_loss = total_loss / log_step
            elapse = time.time() - start_time

            print('| epoch {:3d} | batch {:3d}/{:3d} | {:5.2f} ms/batch | loss {:5.2f} |'.format(
                epoch, i, n_batch, elapse * 1000 / log_step, avg_loss
            ))

            start_time = time.time()
            total_loss = 0.0
        


        
for epoch in range(epochs):
    train(model)