from transformers import BertTokenizer
import torch
import numpy as np
import pandas as pd
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
from eval import Evaluation

labels = {
    'right': 0,
    'left': 1,
    'center': 2
}

def get_bert_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-cased')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.tokenizer = get_bert_tokenizer()
        self.labels = [labels[label] for label in df['type']]
        self.texts = [self.tokenizer(str(text), padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for text in df['article']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch_texts = self.texts[idx]
        batch_y = np.array(self.labels[idx])

        return batch_texts, batch_y

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5, hidden_size=256):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 3)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear1(dropout_output)
        hidden_layer = self.relu(linear_output)
        final_layer = self.relu(self.linear2(hidden_layer))

        return final_layer

def test_model(model, test_data, batch_size=2):
    test_data = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    total_acc_test = 0
    with torch.no_grad():
        preds = []
        labels = []
        for test_input, test_label in tqdm(test_dataloader):
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            preds.extend(output.argmax(dim=1).tolist())
            labels.extend(test_label.tolist())
            total_acc_test += acc

    eval = Evaluation(preds, labels)
    eval.all_metrics()

def train(model, train_data, val_data, learning_rate, epochs, batch_size=2):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    print("Using Cuda:", use_cuda)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
    
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}')




