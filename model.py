from transformers import BertTokenizer
import torch
import numpy as np
import pandas as pd
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
from eval import Evaluation


def get_bert_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-cased')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.tokenizer = get_bert_tokenizer()

        self.labels = []
        for label in df['type']:
            if label == 'right':
                self.labels.append(0)
            if label == 'left':
                self.labels.append(1)
            if label == 'center':
                self.labels.append(2)

        self.texts = []
        for text in df['article']:
            text = self.tokenizer(str(text), padding='max_length',
                                  max_length=512, truncation=True, return_tensors="pt")
            # print(text)
            self.texts.append(text)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], np.array(self.labels[idx])


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5, hidden_size=256):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 3)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        x, bert_out = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False)
        bert_out = self.dropout(bert_out)
        hidden_out = self.relu(self.linear1(bert_out))
        final_out = self.relu(self.linear2(hidden_out))

        return final_out


def test_model(model, test_data, batch_size=2):
    test_data = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True)

    cuda = torch.cuda.is_available()
    if not cuda:
        print("Warning: not using cuda!")

    device = torch.device("cuda" if cuda else "cpu")

    total_correct = 0
    with torch.no_grad():
        preds = []
        labels = []
        for test_input, test_label in tqdm(test_dataloader):
            test_label = test_label.to(device)
            attention_mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1)
            input_id = input_id.to(device)

            output = model(input_id, attention_mask)
            num_correct = (output.argmax(dim=1) == test_label).sum().item()
            preds.extend(output.argmax(dim=1).tolist())
            labels.extend(test_label.tolist())
            total_correct += num_correct

    eval = Evaluation(preds, labels)
    eval.all_metrics()


def train(model, train_data, val_data, learning_rate, epochs, batch_size=2):

    train = Dataset(train_data)
    validation = Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
        validation, batch_size=batch_size)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    if cuda:
        model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if cuda:
        print("Using Cuda")
    else:
        print("Warning: not using cuda!")

    for epoch_num in range(epochs):

        total_correct = 0
        total_loss = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            attention_mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, attention_mask)

            loss = criterion(output, train_label)
            total_loss += loss.item()

            num_correct = (output.argmax(dim=1) == train_label).sum().item()
            total_correct += num_correct

            model.zero_grad()
            loss.backward()
            optimizer.step()

        total_correct_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for input_val, label_val in val_dataloader:
                label_val = label_val.to(device)
                mask = input_val['attention_mask'].to(device)
                input_id = input_val['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                loss = criterion(output, label_val)
                total_loss_val += loss.item()

                num_correct = (output.argmax(dim=1) == label_val).sum().item()
                total_correct_val += num_correct

        print('Epochs: ' + str(round(epoch_num + 1, 3)) + ' | Train Loss: ' + str(round(total_loss / len(train_data), 3)) + ' | Train Accuracy: ' + str(round(total_correct /
              len(train_data), 3)) + ' | Val Loss: ' + str(round(total_loss_val / len(val_data), 3)) + '| Val Accuracy: ' + str(round(total_correct_val / len(val_data), 3)))
