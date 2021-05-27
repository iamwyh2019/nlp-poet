import torch
from dataloader import poet_dataset
from model import PoetModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = 'data/qiyanjueju.txt'
dataset = poet_dataset(data_path)
n_batch, _, seq_len = dataset.train_data[0].shape

model_path = 'final_model.pt'
model = torch.load(model_path, map_location = device)
hidden = None
n_sents, n_words = model.info()
print("mode: {}言{}".format(
    "零一二三四五六七八九"[n_sents],
    "绝句" if n_words == 5 else "律诗"
))

pre_word = "泉眼无声惜细流，树阴照水爱晴柔。小荷才露尖尖角，早有蜻蜓立上头。"

def pre_process(pre_word):
    global hidden
    model.eval()
    with torch.no_grad():
        for word in pre_word:
            ipt = dataset.head2vec(word).to(device)
            opt, hidden = model(ipt, hidden)

pre_process(pre_word)

while True:
    model.eval()
    heads = input('Input heads: ')
    if len(heads) != n_sents:
        print('Invalid input')
        continue
    poet = []
    cur_hidden = hidden
    #input_ = None
    n_len = 0
    with torch.no_grad():
        for i, head in enumerate(heads):
            sent = [head]
            ipt = dataset.head2vec(head).to(device)
            #input_ = ipt if input_ is None else torch.cat([input_, ipt], dim = 0)
            #n_len += 1

            for j in range(1, n_words):
                #input_mask = model.generate_square_subsequent_mask(n_len).to(device)
                opt, cur_hidden = model(ipt, cur_hidden)
                #opt = model(input_, input_mask)
                word_idx = torch.argmax(opt.squeeze()).item()
                #word_idx = torch.argmax(opt[-1]).item()
                word = dataset.num2word(word_idx)
                sent.append(word)
                ipt = dataset.head2vec(word).to(device)
                #input_ = torch.cat([input_, ipt], dim = 0)
                #n_len += 1
            
            opt, cur_hidden = model(ipt, cur_hidden)
            sep = '，' if i % 2 == 0 else '。'
            sep_emb = dataset.head2vec(sep).to(device)
            sent.append(sep)
            #input_ = torch.cat([input_, sep_emb], dim = 0)
            #n_len += 1
            opt, cur_hidden = model(sep, cur_hidden)

            poet.append(''.join(sent))
    
    print('\n'.join(poet))
    print()