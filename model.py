import torch
import torch.nn as nn
import math

class PoetModel(nn.Module):
    def __init__(self, voc_size, input_size, hidden_size, n_layers, n_sents, n_words,
                data_path, pre_word):
        super(PoetModel, self).__init__()
        self.encoder = nn.Embedding(num_embeddings = voc_size, embedding_dim = input_size)

        self.n_sents = n_sents
        self.n_words = n_words
        self.data_path = data_path
        self.pre_word = pre_word

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.drop = nn.Dropout(0.5)

        self.lstm = nn.LSTM(
                input_size = input_size,
                hidden_size = hidden_size,
                num_layers = n_layers,
                batch_first = True,
                #dropout = 0.5
            )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.decoder = nn.Linear(hidden_size, voc_size)
        
        self.init_weights()

    def forward(self, input, hidden = None):
        # input: batch_sz * seq_len
        batch_sz, seq_len  = input.size()
        if hidden is None:
            ht = torch.zeros((self.n_layers, batch_sz, self.hidden_size)).to(input.device)
            ct = torch.zeros((self.n_layers, batch_sz, self.hidden_size)).to(input.device)
            hidden = (ht, ct)
        
        # embedding: batch_sz * seq_len * voc_sz
        embedding = self.encoder(input)

        embedding = self.drop(embedding)

        # output: batch_sz * seq_len * hidden_size        
        output, hidden = self.lstm(embedding, hidden)

        output = self.layer_norm(output)

        output = self.drop(output)

        # decode: (batch_sz * seq_len) * voc_sz
        decode = self.decoder(output.reshape(batch_sz * seq_len, -1))

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


class TransformerModel(nn.Module):

    def __init__(self, voc_size, input_size, n_heads, hidden_size, n_layers,
                n_sents, n_words, data_path, dropout = 0.5):

        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(num_embeddings = voc_size, embedding_dim = input_size)

        self.pos_encoder = PositionalEncoding(input_size, dropout)

        encoder_layers = nn.TransformerEncoderLayer(input_size, n_heads, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        
        self.input_size = input_size
        self.decoder = nn.Linear(input_size, voc_size)

        self.n_sents = n_sents
        self.n_words = n_words
        self.data_path = data_path

        self.init_weights()


    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self, initrange = 0.1):
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, input_mask):
        # batch_first = False
        seq_len, batch_sz = input.size()
        src = self.embedding(input) * math.sqrt(self.input_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, input_mask)
        decoded = self.decoder(output.reshape(seq_len * batch_sz, -1))

        return decoded
    
    def info(self):
        return self.n_sents, self.n_words



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout = 0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)