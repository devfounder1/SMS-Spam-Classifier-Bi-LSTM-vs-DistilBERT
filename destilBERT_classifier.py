import torch
import torch.nn as nn 
import copy
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
model = AutoModelForSequenceClassification(MODEL_NAME, num_labels = 2)

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
    