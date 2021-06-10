import torch
from dataloader import poet_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mode_LSTM:
    def __init__(self, name, path):
        print(f'Loading model for 『{name}』...')
        self.name = name
        self.model = torch.load(path, map_location=device)
        self.hidden = None
        self.n_sents, self.n_words = self.model.info()
        self.dataset = poet_dataset(self.model.data_path)
        self.sep = self.dataset.head2vec(self.dataset.sep).to(device)
        self.model.eval()
        with torch.no_grad():
            for word in self.model.pre_word:
                ipt = self.dataset.head2vec(word).to(device)
                _, self.hidden = self.model(ipt, self.hidden)
    def entry(self, heads):
        if len(heads) != self.n_sents:
            return "Invalid input"
        self.model.eval()
        poet = []
        cur_hidden = self.hidden
        with torch.no_grad():
            for i, head in enumerate(heads):
                sent = []
                ipt = self.dataset.head2vec(head).to(device)
                sent.append(head)

                for _ in range(self.n_words - 1):
                    opt, cur_hidden = self.model(ipt, cur_hidden)
                    word_idx = torch.argmax(opt.squeeze()).item()
                    word = self.dataset.num2word(word_idx)
                    sent.append(word)
                    ipt = self.dataset.head2vec(word).to(device)

                opt, cur_hidden = self.model(ipt, cur_hidden)
                sent.append('，' if i % 2 == 0 else '。')
                opt, cur_hidden = self.model(self.sep, cur_hidden)

                poet.append(''.join(sent))
        return '\n'.join(poet)

class Mode_Transformer():
    def __init__(self, name, path):
        print(f'Loading model for 『{name}』...')
        self.name = name
        self.model = torch.load(path, map_location = device)
        self.hidden = None
        self.n_sents, self.n_words = self.model.info()
        self.dataset = poet_dataset(self.model.data_path)
        self.sep = self.dataset.head2vec(self.dataset.sep).to(device)
        self.model.eval()

    def entry(self, heads):
        if len(heads) != self.n_sents:
            return "Invalid input"
        self.model.eval()
        poet = []
        with torch.no_grad():
            input = None
            for i, head in enumerate(heads):
                sent = []
                ipt = self.dataset.head2vec(head).to(device)
                input = ipt if input is None else torch.cat([input, ipt], dim = 0)
                sent.append(head)

                for _ in range(self.n_words - 1):
                    input_mask = self.model.generate_square_subsequent_mask(input.shape[0]).to(device)
                    output = self.model(input, input_mask)
                    word_idx = torch.argmax(output[-1]).item()
                    word = self.dataset.num2word(word_idx)
                    sent.append(word)
                    ipt = self.dataset.head2vec(word).to(device)
                    input = torch.cat([input, ipt], dim = 0)

                sent.append('，' if i % 2 == 0 else '。')
                input = torch.cat([input, self.sep], dim = 0)
                poet.append(''.join(sent))

        return '\n'.join(poet)

Mode = Mode_LSTM

'''
modes = [Mode('五言绝句', 'wuyanjueju_final_model.pt'),
         Mode('七言绝句', 'qiyanjueju_final_model.pt'),
         Mode('五言律诗', 'wuyanlvshi_final_model.pt'),
         Mode('七言律诗', 'qiyanlvshi_final_model.pt')]
curmode = 0
'''

def getAllModes():
    return modes

def getCurMode():
    return modes[curmode]

def setCurModeIndex(index):
    global curmode
    curmode = index

if __name__ == "__main__":
    model = Mode_Transformer('七言绝句', 'tr_qiyanjueju_final.pt')
    while True:
        heads = input('Head: ')
        output = model.entry(heads)
        print(output)