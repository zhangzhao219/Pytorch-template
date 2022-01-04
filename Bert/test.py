import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

device = torch.device('cuda')

df_train = pd.read_csv('./data/disaster/train_data.csv',names=['id','sentence','target'])
df_test = pd.read_csv('./data/disaster/test_data.csv',names=['id','sentence','target'])


df_train['final_text'] = df_train['sentence']
df_test['final_text'] = df_test['sentence']

# Get text values and labels
text_values = df_train['final_text'].values
labels = df_train['target'].values

# Load the pretrained Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def encode_fn(text_list):
    all_input_ids = []
    for text in text_values:
        input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=180, pad_to_max_length=True, return_tensors='pt')
        all_input_ids.append(input_ids)
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids

epochs = 1
batch_size = 32

# Split data into train and validation
all_input_ids = encode_fn(text_values)
labels = torch.tensor(labels)

train_dataset = TensorDataset(all_input_ids, labels)

# Create train and validation dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Load the pretrained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_attentions=False, output_hidden_states=False)
model.cuda()

# create optimizer and learning rate schedule
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_dataloader) * epochs

from sklearn.metrics import f1_score, accuracy_score

def flat_accuracy(preds, labels):
    
    """A function for calculating accuracy scores"""
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)

for epoch in range(epochs):
    model.train()
    total_loss, total_val_loss = 0, 0
    total_eval_accuracy = 0
    for step, batch in enumerate(train_dataloader):
        print(step)
        model.zero_grad()
        # print(model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0]>0).to(device), labels=batch[1].to(device)))
        outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0]>0).to(device), labels=batch[1].to(device))
        # print(batch[0].size())
        # print((batch[0]>0).size())
        loss, logits = outputs.loss,outputs.logits
        total_loss += loss.item()
        loss.backward()
        optimizer.step() 
        
    model.eval()
    for i, batch in enumerate(train_dataloader):
        with torch.no_grad():
            outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0]>0).to(device), labels=batch[1].to(device))
            loss, logits = outputs.loss,outputs.logits
                
            total_val_loss += loss.item()
            
            logits = logits.detach().cpu().numpy()
            label_ids = batch[1].to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
    
    avg_train_loss = total_loss / len(train_dataloader)
    avg_val_loss = total_val_loss / len(train_dataloader)
    avg_val_accuracy = total_eval_accuracy / len(train_dataloader)
    
    print(f'Train loss     : {avg_train_loss}')
    print(f'Validation loss: {avg_val_loss}')
    print(f'Accuracy: {avg_val_accuracy}')
    print('\n')