import fastapi
from fastapi import HTTPException, FastAPI
import torch
from pydantic import BaseModel
import torch.nn as nn 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import re
import torch.nn.functional as F
from contextlib import asynccontextmanager
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s : %(message)s')

class LSTMclasssifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(LSTMclasssifier, self).__init__()
        
        self.embeddings = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size=hidden_dim, bidirectional=True, batch_first=True, dropout=0.3, input_size=embedding_dim,num_layers=2)
        self.line = nn.Linear(in_features=hidden_dim * 2, out_features=num_classes)
        
    def forward(self, x):
        x = self.embeddings(x)
        output, (hidden, cell) = self.lstm(x)
        last_layer = torch.cat((hidden[-2], hidden[-1]), dim=1)
        logits = self.line(last_layer)
        return logits

lstm_model = None
lstm_vocab = None
hf_tokenizer = None
hf_model = None

def clean_text(text : str) -> str: 
    re.sub(r'[^a-zа-я1-9\s]', '', text.lower())
    return text
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    
    #загрузка при старте приложения
    global lstm_model, lstm_vocab, hf_model, hf_tokenizer
    logging.info("Загрузка моделей")
    
    # загрузка LSTM 
    with open("./models/lstm/config.json", "r", encoding="utf-8") as f:
        lstm_config = json.load(f)
    with open("./models/lstm/vocab.json", "r", encoding="utf-8") as f:
        lstm_vocab = json.load(f)

    lstm_model = LSTMclasssifier(**lstm_config)
    lstm_model.load_state_dict(torch.load("./models/lstm/model.pth",  map_location="cpu"))
    lstm_model.eval()
    
    # Загрузка DestilBERT
    hf_tokenizer = AutoTokenizer.from_pretrained("./models/destilbert")
    hf_model = AutoModelForSequenceClassification.from_pretrained("./models/destilbert")
    hf_model.eval()
    
    logging.info("Обе модели загрузились успешно")
    yield
    
app = FastAPI(
    title="SMS Spam classifier API",
    description="Сравнение LSTM и DestilBERT для классификации спама",
    version="1.0.0",
    lifespan=lifespan
)

class TextRequest(BaseModel):
    text : str
    
class PredictionResponse(BaseModel):
    text : str
    model_used : str
    is_spam : bool
    confidence_spam : float
    confidence_ham : float
    
@app.post("/predict/lstm", response_model=PredictionResponse)
def predict_lstm(request : TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    
    # Очистка и токенизация
    clean_t = clean_text(request.text)
    words = clean_t.split()
    indeces = [lstm_vocab.get(w, lstm_vocab.get("<UNK>", 1)) for w in words]
    
    # Создание тензора 
    input_tensor = torch.tensor([indeces], dtype=torch.long)
    
    # Предсказание
    with torch.no_grad():
        logits = lstm_model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze().tolist()
    
    return PredictionResponse(
        text = request.text,
        model_used="Bi-LSTM (Custom)",
        is_spam = bool(probs[1] > probs[0]),
        confidence_spam=round(probs[1], 4),
        confidence_ham=round(probs[0], 4),
    )
    
@app.post("/predict/destilBERT", response_model=PredictionResponse)
def predict_destilBERT(request : TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    
    inputs = hf_tokenizer(
        request.text,
        return_tensors = "pt",
        truncation = True,
        max_length = 64,
        padding = True
    )
    
    with torch.no_grad():
        outputs = hf_model(**inputs)
        probs = F.softmax(outputs.logits, dim = 1).squeeze().tolist()
        
    return PredictionResponse(
        text=request.text,
        model_used="destilBERT (Hugging Face)",
        is_spam = bool(probs[1] > probs[0]),
        confidence_spam=round(probs[1], 4),
        confidence_ham=round(probs[0], 4),
    )

@app.get("/")
def health_check():
    return {
        "status" : "OK",
        "Models_loaded" : True,
        "Endpints" : ["/predict/lstm", "/predict/distilBERT"]
    }