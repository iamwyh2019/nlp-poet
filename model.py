import torch
import torch.nn as nn

class PoetModel(nn.Module):
    def __init__(self, voc_size, input_size, hidden_size, n_layers, n_sents, n_words):
        super(PoetModel, self).__init__()
        self.encoder = nn.Embedding(num_embeddings = voc_size, embedding_dim = input_size)

        self.n_sents = n_sents
        self.n_words = n_words

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.drop = nn.Dropout(0.5)

        self.lstm = nn.LSTM(
                input_size = input_size,
                hidden_size = hidden_size,
                num_layers = n_layers,
                batch_first = True
            )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.decoder = nn.Linear(hidden_size, voc_size)

        self.softmax = nn.Softmax(dim = 1)
        
        self.init_weights()

    def forward(self, input, hidden = None):
        # input: batch_sz * seq_len
        batch_sz, seq_len  = input.size()
        if hidden is None:
            ht = torch.zeros((self.n_layers, batch_sz, self.hidden_size)).to(self.device)
            ct = torch.zeros((self.n_layers, batch_sz, self.hidden_size)).to(self.device)
            hidden = (ht, ct)
        
        # embedding: batch_sz * seq_len * voc_sz
        embedding = self.encoder(input)

        embedding = self.drop(embedding)

        # output: batch_sz * seq_len * hidden_size        
        output, hidden = self.lstm(embedding, hidden)

        output = self.drop(output)

        output = self.layer_norm(output)

        # decode: (batch_sz * seq_len) * voc_sz
        decode = self.decoder(output.reshape(batch_sz * seq_len, -1))

        prob = self.softmax(decode)

        return decode, hidden
    
    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.uniform_(-init_uniform, init_uniform)
    
    def detach_hidden(self, hidden):
        if isinstance(hidden, torch.Tensor):
            return hidden.detach()
        else:
            return [self.detach_hidden(v) for v in hidden]
    
    def info(self):
        return self.n_sents, self.n_words
