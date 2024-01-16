import torch
from transformers import DistilBertModel
from config import *
from sklearn.metrics import classification_report


def calcuate_accu(big_idx : torch.tensor, target):
    n_correct = (big_idx==target).sum().item()
    return n_correct

def train_model(epoch, model, training_loader, loss_function, optimizer):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    print(f'training start for epoch #{epoch}')
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(DEVICE, dtype=torch.long)
        mask = data['mask'].to(DEVICE, dtype=torch.long)
        targets = data['targets'].to(DEVICE, dtype=torch.long)

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        print("progress:" + str(_))

        loss_step = tr_loss / nb_tr_steps
        accu_step = (n_correct * 100) / nb_tr_examples
        print(f"loss: {loss_step}")
        print(f"accuracy: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'total accuracy for epoch #{epoch}: {(n_correct * 100) / nb_tr_examples}')
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"total loss: {epoch_loss}")
    print(f"total accuracy: {epoch_accu}")

def test_model(model, testing_loader):
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(DEVICE, dtype=torch.long)
            mask = data['mask'].to(DEVICE, dtype=torch.long)
            targets = data['targets'].to(DEVICE, dtype=torch.long)
            outputs = model(ids, mask).squeeze()
            bigval, big_index = torch.max(outputs.data, dim=1)
            print(classification_report(targets, big_index, target_names = LABELS_TEXTS))