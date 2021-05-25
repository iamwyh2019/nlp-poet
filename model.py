import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, voc_size, input_size, hidden_size, n_layers, dropout = None):
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(num_embeddings = voc_size, embedding_dim = input_size)
        if dropout is None:
            dropout = 0
        self.dropout = nn.Dropout(p = dropout)

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
                input_size = input_size,
                hidden_size = hidden_size,
                num_layers = n_layers,
                batch_first = True
            )
        
        self.decoder = nn.Linear(hidden_size, voc_size)
        
        self.init_weights()
    
    '''
    
    def prop(self, input, hidden):
        # input: (1,batch_size)
        emb = self.encoder(input)
        # emb: (1, batch_size, input_size)

        #if self.dropout is not None:
        #    emb = self.dropout(emb)
        
        output, (h1,c1) = self.lstm(emb, hidden)

        if self.dropout is not None:
            output = self.dropout(output)
        
        output = self.decoder(output.reshape(output.size(0)*output.size(1), output.size(2)))
        # output: batch_size * voc_size
        return output, (h1,c1)
    
    def forward(self, input, hidden = None):
        # input: seq_len * batch_size
        seq_len, batch_sz = input.size()
        if hidden is None:
            ht = torch.zeros((self.n_layers, batch_sz, self.hidden_size)).to(input.device)
            ct = torch.zeros((self.n_layers, batch_sz, self.hidden_size)).to(input.device)
        else:
            ht, ct = hidden
        
        results = []

        for sent_idx in range(self.n_sents):
            # ipt: 1 * batch_size
            ipt = input[sent_idx * 2].unsqueeze(0)
            ipt_emb = self.encoder(ipt)
            print(ipt_emb.shape)
            results.append(ipt_emb)

            for word_idx in range(self.n_words - 1):
                opt, (ht,ct) = self.prop(ipt, (ht,ct))
                results.append(opt)
                print(opt.shape)
                words = torch.argmax(opt, dim = 1).unsqueeze(0)
                ipt = words
            
            # Last word
            opt, (ht,ct) = self.prop(ipt, (ht,ct))

            # EOL
            ipt = input[sent_idx * 2 + 1].unsqueeze(0)
            ipt_emb = self.encoder(ipt)
            results.append(ipt_emb)
            opt, (ht,ct) = self.prop(ipt, (ht,ct))
        
        return torch.cat(results, dim = 0)
    '''

    def forward(self, input, hidden = None):
        batch_sz, seq_len  = input.size()
        if hidden is None:
            ht = torch.zeros((self.n_layers, batch_sz, self.hidden_size)).to(input.device)
            ct = torch.zeros((self.n_layers, batch_sz, self.hidden_size)).to(input.device)
        else:
            ht, ct = hidden
        
        embedding = self.encoder(input)
        embedding = self.dropout(embedding)
        
        output, hidden = self.lstm(embedding, (ht,ct))
        output = self.dropout(output)

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
