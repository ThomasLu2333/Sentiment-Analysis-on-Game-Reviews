from transformers import DistilBertTokenizer
from torch import cuda
import torch

DEVICE = 'cuda' if cuda.is_available() else 'cpu'
MAX_LEN = 256
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 4096
EPOCHS = 1
LR = 1e-5
CHECKPOINT = 'distilbert-base-cased'
LABELS_TEXTS = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
TEXTS_LABELS = dict(zip(LABELS_TEXTS, range(3)))
TRAIN_SIZE = 0.8
FILENAME = 'Video_Games_5.json'
RANDOM_STATE = 200
TOKENIZER_CONFIG = {
    'add_special_tokens': True,
    'max_length': MAX_LEN,
    'pad_to_max_length': True,
    'return_token_type_ids': True,
    'truncation': True
}
TOKENIZER = DistilBertTokenizer.from_pretrained(CHECKPOINT)
LOSS_FUNCTION_CLASS = torch.nn.CrossEntropyLoss
OPTIMIZER_CLASS = torch.optim.Adam
OUTPUT_MODEL_FILE = 'models_bin/pytorch_distilbert_news.bin'
OUTPUT_VOCAL_FILE = 'models_bin/vocab_distilbert_news.bin'
