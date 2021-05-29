import torch
from dataloader import poet_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = 'new_final_model.pt'
model = torch.load(model_path, map_location = device)
print(model)
hidden = None
n_sents, n_words = model.info()

data_path = model.data_path
dataset = poet_dataset(data_path)
sep = dataset.head2vec('#').to(device)

pre_word = "远上寒山石径斜，白云深处有人家。停车坐爱枫林晚，霜叶红于二月花。"

def pre_process(pre_word):
    global hidden
    model.eval()
    with torch.no_grad():
        for word in pre_word:
            ipt = dataset.head2vec(word).to(device)
            opt, hidden = model(ipt, hidden)
pre_process(pre_word)

def get_mode() -> str:
    mode = n_sents * 10 + n_words
    mode_mp = {45: "五言绝句", 47: "七言绝句", 85: "五言律诗", 87: "七言律诗"}
    return mode_mp[mode]

def entry(heads):
    if len(heads) != n_sents:
        return "Invalid input"
    model.eval()
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
    return '\n'.join(poet)
