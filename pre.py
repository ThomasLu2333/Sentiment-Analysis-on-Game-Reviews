import pandas as pd
import json
from config import *
from matplotlib import pyplot as plt
import numpy as np

def sequence_length(sequence):
    return len(TOKENIZER(str(sequence))['input_ids'])

with open("Video_Games_5.json", "r") as f:
    text = f.read()
    reviews = text.split("\n")
    reviews_jsonified = [json.loads(s) for s in reviews[:-1]]
    data = pd.DataFrame(reviews_jsonified)
    train_lengths = np.array([sequence_length(s) for s in data['reviewText']])

    mean_length = np.mean(train_lengths)
    median_length = np.median(train_lengths)
    std_dev_length = np.std(train_lengths)

    hist_data = plt.hist(train_lengths, bins=30, color='blue', alpha=0.7)
    plt.xlabel('sentence2 tokenized tokens length')
    plt.show()

    print(mean_length, median_length, std_dev_length, hist_data)






