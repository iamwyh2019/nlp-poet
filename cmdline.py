import torch
from dataloader import poet_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chn_number = '零一二三四五六七八九'

class Mode_LSTM:
    def __init__(self, path):
        print('Loading model...')
        
        self.model = torch.load(path, map_location=device)
        self.hidden = None
        self.n_sents, self.n_words = self.model.info()

        self.name = '{}言{}'.format(chn_number[self.n_sents], '绝句' if self.n_words==4 else '律诗')
        print('Loaded model for', self.name)

        self.dataset = poet_dataset(self.model.data_path)
        self.sep = self.dataset.head2vec(self.dataset.sep).to(device)
        self.model.eval()
        #print(self.model.pre_word)
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


if __name__ == "__main__":
    mode = Mode_LSTM('qiyanjueju_final_model.pt')
    while True:
        heads = input('输入诗头，由{}个字组成： '.format(mode.model.n_sents))
        output = mode.entry(heads)
        print(output)