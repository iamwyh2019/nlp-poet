import torch
from dataloader import poet_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = 'data/qiyanjueju.txt'
dataset = poet_dataset(data_path)
sep = dataset.head2vec('#').to(device)

model_path = 'new_final_model.pt'
model = torch.load(model_path, map_location = device)
hidden = None
n_sents, n_words = model.info()

print('mode: {}言{}'.format(
    "零一二三四五六七八九"[n_words],
    "绝句" if n_sents == 4 else "律诗"
))

pre_word = "泉眼无声惜细流，树阴照水爱晴柔。小荷才露尖尖角，早有蜻蜓立上头。"

def pre_process(pre_word):
    global hidden
    model.eval()
    with torch.no_grad():
        for word in pre_word:
            ipt = dataset.head2vec(word).to(device)
            opt, hidden = model(ipt, hidden)

#pre_process(pre_word)

while True:
    model.eval()
    heads = input('Input heads: ')
    if len(heads) != n_sents:
        print('Invalid input')
        continue
    poet = []
    cur_hidden = hidden
    with torch.no_grad():
        for i, head in enumerate(heads):
            sent = []
            ipt = dataset.head2vec(head).to(device)
            sent.append(head)

            for _ in range(n_words - 1):
                opt, cur_hidden = model(ipt, cur_hidden)
                word_idx = torch.argmax(opt.squeeze()).item()
                word = dataset.num2word(word_idx)
                sent.append(word)
                ipt = dataset.head2vec(word).to(device)
            
            opt, cur_hidden = model(ipt, cur_hidden)
            sent.append('，' if i % 2 == 0 else '。')
            opt, cur_hidden = model(sep, cur_hidden)

            poet.append(''.join(sent))
    
    print('\n'.join(poet))
    print()