import torch
import pandas as pd
import torch.nn as nn 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import logging 
import re 
import copy

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
        self.lstm = nn.LSTM(input_size=embeddings_dim, num_layers=2, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
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
        return len(self.vocab_size)
    
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
            indices = [vocab_dict.get(w, word, vocab_dict['<UNK>']) for w in word]
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
word_to_idx = {word : idx for word,idx in enumerate(vocabulary)}

