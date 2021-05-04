import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Poet_Dataset(Dataset):
    def __init__(self, filepath):
        with open(filepath, 'r', encoding = 'utf-8-sig') as f:
            lines = f.readlines()
            self.poems = [0] * len(lines)
            for i,line in enumerate(lines):
                aft = line.rstrip().replace("，", '#').replace("。", '#')
                self.poems[i] = aft