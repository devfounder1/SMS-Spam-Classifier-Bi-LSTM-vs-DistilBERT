import torch
import torch.nn as nn 
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
import logging 
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s : %(message)s')
basic_url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

df = pd.read_csv(basic_url, sep='\t', names = ['labels', 'text'])
df['labels'] = df['labels'].map({'ham' : 0, 'spam' : 1})
logging.info(df.head(5))
logging.info(df.shape)

all_texts = df['text'].tolist()
all_labels = df['labels'].tolist()

x_train, x_test, y_train, y_test = train_test_split(all_texts, all_labels, test_size=0.2, stratify=all_labels, random_state=42)

class SimpDataset(Dataset):
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        return self.text[index], self.labels[index]

MODEL_NAME = "distilbert-base-uncased"

logging.info("Загрузка токенайзера, может занять (10-20 сек)") 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = 2) 

def make_hf_collate_fn(tokenizer):
    def collate_fn(batch):
        texts = [item[0] for item in batch]
        labels = [int(item[1]) for item in batch]
        
        encoding = tokenizer(
            texts,
            padding = True,
            max_length = 64,
            truncation = True,
            return_tensors = "pt",
        )
        encoding['labels'] = torch.tensor(labels, dtype=torch.long)
        return encoding
    return collate_fn

train_dataset = SimpDataset(x_train, y_train)
test_dataset = SimpDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32, collate_fn=make_hf_collate_fn(tokenizer))
test_loader =  DataLoader(test_dataset, shuffle=False, batch_size=32, collate_fn=make_hf_collate_fn(tokenizer))

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

EPOCHS = 5

best_val_loss = float('inf')
patience = 3
patience_counter = 0
best_weights = None

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for batch in test_loader:
            out = model(**batch)
            loss_v = out.loss
            val_loss += loss_v.item()
            
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(test_loader)
    logging.info(f"Эпоха: {epoch+1}/{EPOCHS} | AVG Val_loss : {avg_val_loss:.4f} | AVG train_loss : {avg_train_loss:.4f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_weights = copy.deepcopy(model.state_dict())
        logging.info("Найдена лучшая модель! Веса сохранены.")
    else: 
        patience_counter += 1
        logging.info(f"Улучшений нет | Patience: {patience_counter} / {patience}")
        if patience_counter >= patience:
            logging.info(f"Произошел Early Stopping : {epoch+1}. Лучший val_loss: {best_val_loss}")
            break