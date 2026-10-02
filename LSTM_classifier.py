import torch
import pandas as pd
import torch.nn as nn 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import logging 
import re 
import copy
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s : %(message)s')
basic_url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

df = pd.read_csv(basic_url, sep='\t', names = ['labels', 'text'])
df['labels'] = df['labels'].map({'ham' : 0, 'spam' : 1})
logging.info(df.head(5))
logging.info(df.shape)

all_texts = df['text'].tolist()
all_labels = df['labels'].tolist()

x_train, x_test, y_train, y_test = train_test_split(all_texts, all_labels, test_size=0.2, stratify=all_labels, random_state=42)

class LSTMclassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embeddings_dim, hidden_dim):
        super().__init__()

        self.embeddings = nn.Embedding(embedding_dim=embeddings_dim, num_embeddings=vocab_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embeddings_dim, num_layers=2, hidden_size=hidden_dim, batch_first=True, bidirectional=True, dropout=0.3)
        self.line = nn.Linear(in_features=hidden_dim * 2, out_features=num_classes)
        
    def forward(self, x):
        x = self.embeddings(x)
        outputs, (hidden, cell) = self.lstm(x)
        last_layer_lstm = torch.cat((hidden[-2],hidden[-1]), dim=1) 
        logits = self.line(last_layer_lstm)
        return logits
    
class SimpleDataset(Dataset):
    def __init__(self, text, labels, vocab_size):
        self.text = text
        self.labels = labels
        self.vocab_size = vocab_size
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        return self.text[index], self.labels[index]

def clean_text(text : str) -> str: 
    text = re.sub(r'[^a-zа-яё1-9\s]', '', text.lower())
    return text

def make_collate_fn(vocab_dict):
    def collate_fn(batch):
        text_in_batch = [item[0] for item in batch]
        labels_in_batch = [int(item[1]) for item in batch]
        
        list_of_tensors = []
        for text in text_in_batch:
            clean_t = clean_text(text)
            word = clean_t.split()
            indices = [vocab_dict.get(w, vocab_dict['<UNK>']) for w in word]
            list_of_tensors.append(torch.tensor(indices, dtype=torch.long))
        
        padded_inputs = torch.nn.utils.rnn.pad_sequence(list_of_tensors, padding_value=0, batch_first=True)
        labels_tensor = torch.tensor(labels_in_batch, dtype=torch.long)

        return padded_inputs, labels_tensor
    return collate_fn

uniq_word = set()
for text in x_train:
    for word in clean_text(text).split():
        uniq_word.add(word)
        
vocabulary = ["<PAD>", "<UNK>"] + sorted(list(uniq_word))
word_to_idx = {word : idx for idx,word in enumerate(vocabulary)}

HIDDEN_DIM =64
NUM_CLASSES = 2
VOCAB_SIZE = len(vocabulary)        
EMBEDDING_DIM = 32

model = LSTMclassifier(hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES, vocab_size=VOCAB_SIZE, embeddings_dim=EMBEDDING_DIM)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
criterion = torch.nn.CrossEntropyLoss()

train_dataset = SimpleDataset(x_train, y_train, word_to_idx)
test_dataset = SimpleDataset(x_test, y_test, word_to_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=make_collate_fn(word_to_idx))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=make_collate_fn(word_to_idx))

EPOCHS = 10
best_val_loss = float('inf')
patience = 3
patience_count = 0
best_weights = None



for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    for batch_inputs, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    model.eval()
    
    with torch.no_grad():
        val_loss = 0.0
        
        all_preds = []
        all_labels = []
        
        for inp, batch_labels in test_loader:
            out = model(inp)
            loss_v = criterion(out, batch_labels)
            val_loss += loss_v.item()

            preds = torch.argmax(out, dim = 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(test_loader)
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    logging.info(f"Метрики Val: Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f}")
    logging.info(f"Эпоха {epoch+1} / {EPOCHS} | AVG train loss : {avg_train_loss} | AVG val loss : {avg_val_loss}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_count = 0
        best_weights = copy.deepcopy(model.state_dict())
        logging.info("Найдена лучшая модель !!!!")
    else:
        patience_count += 1 
        logging.info(f"Улучшений нет : {patience_count} / {patience}")
        
        if patience_count >= patience:
            logging.info(f"Ранняя остановка на эпохе {epoch+1} | Лучший val_loss: {best_val_loss:.4f}")
            break
        
if best_weights is not None:
    model.load_state_dict(best_weights) 
    logging.info("Загружены веса лучшей модели для финального использования")
    
SAVE_DIR = "./models/lstm"
os.makedirs(SAVE_DIR, exist_ok=True)

torch.save(model.state_dict(), f"{SAVE_DIR}/model.pth")

with open(f"{SAVE_DIR}/vocab.json", "w", encoding="utf-8") as f:
    json.dump(word_to_idx, f, ensure_ascii=False, indent=2)

config = {
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": EMBEDDING_DIM,
    "hidden_dim": HIDDEN_DIM,
    "num_classes": NUM_CLASSES,
    "num_layers": 2,
    "bidirectional": True
}

with open(f"{SAVE_DIR}/config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

logging.info("Модель успешно сохранена")