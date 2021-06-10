import torch
import torch.nn as nn
import numpy as np
from dataloader import poet_dataset
from model import TransformerModel
import time
import matplotlib.pyplot as plt

data_path = 'data/qiyanjueju.txt'
train_batch_size = 50
eval_batch_size = 40
epochs = 100
input_size = 300
hidden_size = 512
n_layers = 12
clip = 0.1
n_heads = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = poet_dataset(data_path, train_batch_size, eval_batch_size)
tokens, n_sents, n_words = dataset.info()
tf_idf = dataset.tf_idf.to(device)

model = TransformerModel(
    voc_size = tokens,
    input_size = input_size,
    n_heads = n_heads,
    hidden_size = hidden_size,
    n_layers = n_layers,
    n_sents = n_sents,
    n_words = n_words,
    data_path = data_path,
    dropout = 0.5
)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight = tf_idf).to(device)
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

train_loss_history = np.zeros(epochs)
val_loss_history = np.zeros(epochs)
train_ppl_history = np.zeros(epochs)
val_ppl_history = np.zeros(epochs)

def train(model):
    model.train()
    total_loss = 0.0
    avg_loss = 0.0
    start_time = time.time()
    log_step = 20
    n_batch, _, seq_len = dataset.train_data[0].shape
    input_mask = model.generate_square_subsequent_mask(seq_len).to(device)

    for i in range(n_batch):
        input, target = dataset.get_batch(dataset.train_data, i, batch_first = False, flat_target = True)

        optimizer.zero_grad()

        output = model(input, input_mask)

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
    
    return avg_loss
        
def evaluate(model, data):
    model.eval()
    total_loss = 0.0
    total_batch = 0
    n_batch, _, seq_len = data[0].shape
    input_mask = model.generate_square_subsequent_mask(seq_len).to(device)

    with torch.no_grad():
        for i in range(n_batch):
            input, target = dataset.get_batch(data, i, batch_first = False, flat_target = True)
            output = model(input, input_mask)

            loss = criterion(output, target)

            total_loss += loss.item() * input.shape[1]
            total_batch += input.shape[1]
    
    return total_loss / total_batch

best_val_loss = float('inf')
best_model = None

def plot_curve(train_loss, val_loss, model_name, ylab):
    x = range(len(train_loss))
    plt.figure(facecolor = 'white', edgecolor = 'black')
    plt.plot(x, train_loss, color = 'r', linewidth = 2, label = 'Training')
    plt.plot(x, val_loss, color = 'b', linewidth = 2, label = 'Validation')
    plt.title(model_name + ' ' + ylab)
    plt.xlabel('epoch')
    plt.ylabel(ylab)
    plt.legend(loc = 'lower right')
    plt.savefig(model_name + '_' + ylab + '.png')

# Main
for epoch in range(epochs):
    epoch_start_time = time.time()
    train_loss = train(model)
    val_loss = evaluate(model, dataset.val_data)

    train_loss_history[epoch] = train_loss
    val_loss_history[epoch] = val_loss
    train_ppl_history[epoch] = np.exp(train_loss)
    val_ppl_history[epoch] = np.exp(val_loss)

    np.save('train_loss_history.npy', train_loss_history)
    np.save('val_loss_history.npy', val_loss_history)
    np.save('train_ppl_history.npy', train_ppl_history)
    np.save('val_ppl_history.npy', val_ppl_history)

    print('-' * 65)
    print('| epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '.format(
        epoch, (time.time() - epoch_start_time), val_loss))
    print('-' * 65)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        torch.save(best_model, 'tr_qiyanjueju_best.pt')
    
    dataset.shuffle()

torch.save(model, 'tr_qiyanjueju_final.pt')

plot_curve(train_loss_history, val_loss_history, 'Transformer', 'loss')
plot_curve(train_ppl_history, val_ppl_history, 'Transformer', 'perplexity')