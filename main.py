from torch.utils.data import DataLoader

from datasets import *
from models import *
from config import *
from finetune import *

general = load_data(FILENAME)
train = general.sample(frac=TRAIN_SIZE, random_state=RANDOM_STATE)
test = general.drop(train.index).reset_index(drop=True)
train = train.reset_index(drop=True)
print("\nPreview for train:")
print(train.head())
print("\nPreview for test:")
print(test.head())

train_set = MyDataset(train, TOKENIZER)
test_set = MyDataset(test, TOKENIZER)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': TEST_BATCH_SIZE,
               'shuffle': True,
               'num_workers': 0
               }

train_loader = DataLoader(train_set, **train_params)
test_loader = DataLoader(test_set, **test_params)

model = MyDistillBERT()
model.to(DEVICE)
loss_fn = LOSS_FUNCTION_CLASS()
optimizer = OPTIMIZER_CLASS(params=model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    train_model(epoch, model, train_loader, loss_fn, optimizer)

print("printing test reports")
test_model(model, test_loader)

print("saving model")

model_to_save = model
torch.save(model_to_save, OUTPUT_MODEL_FILE)
TOKENIZER.save_vocabulary(OUTPUT_VOCAL_FILE)

print('Done')
