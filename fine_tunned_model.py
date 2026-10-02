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