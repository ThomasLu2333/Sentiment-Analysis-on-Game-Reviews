from torch.utils.data import Dataset
import json
import pandas as pd
from config import *
import torch

def load_data(filename):
    with open(filename, "r") as f:
        jsons = f.read().split("\n")
        decoded = [json.loads(s) for s in jsons[:-1]]
        dataframe = pd.DataFrame(decoded)[['overall', 'reviewText']]
        dataframe['label'] = dataframe.apply(lambda row: 0 if row['overall'] > 3 else 1 if row['overall'] < 3 else 2,
                                             axis=1)
        return dataframe.drop(columns=['overall']).rename(columns = {'reviewText':'text'})

class MyDataset(Dataset):
    def __init__(self, dataframe : pd.DataFrame, tokenizer : DistilBertTokenizer):
        self.tokenizer = tokenizer
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            str(self.data['text'].iloc[index]),
            None,
            **TOKENIZER_CONFIG
        )
        return {
            'ids':torch.tensor(inputs['input_ids'], dtype = torch.long),
            'mask':torch.tensor(inputs['attention_mask'] , dtype=torch.long),
            'targets':torch.tensor(self.data['label'].iloc[index], dtype=torch.long)
        }