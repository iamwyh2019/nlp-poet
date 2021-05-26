import torch
from dataloader import poet_dataset
from model import PoetModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = 'data/qiyanjueju.txt'
dataset = poet_dataset(data_path)
sep = dataset.head2vec('#').to(device)

model_path = 'final_model.pt'
model = torch.load(model_path, map_location = device)
hidden = None
n_sents, n_words = model.info()

pre_word = "碧玉妆成一树高，万条垂下绿丝绦。不知细叶谁裁出，二月春风似剪刀。"

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
    heads = input('Input heads:')
    if len(heads) != n_sents:
        print('Invalid input')
        continue
    poet = []
    cur_hidden = hidden
    with torch.no_grad():
        for i, head in enumerate(heads):
            ipt = dataset.head2vec(head).to(device)
            poet.append(head)

            for _ in range(n_words - 1):
                opt, cur_hidden = model(ipt, cur_hidden)
                word_idx = torch.argmax(opt.squeeze()).item()
                word = dataset.num2word(word_idx)
                poet.append(word)
                ipt = dataset.head2vec(word).to(device)
            
            opt, cur_hidden = model(ipt, cur_hidden)
            poet.append('，' if i % 2 == 0 else '。')
            opt, cur_hidden = model(sep, cur_hidden)
    
    print(''.join(poet))