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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s : %(message)s')

lstm_model = None
lstm_vocab = None
hf_tokenizer = None
hf_model = None

request_status = {
    "total_requests" : 0,
    "spam_count" : 0,
    "ham_count" : 0,
    "lstm_requests" : 0,
    "distilbert_requests" : 0,
    "compare_requests" : 0,
    "last_request" : None
}

class LSTMclasssifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, bidirectional=True):
        super(LSTMclasssifier, self).__init__()
        
        self.embeddings = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=bidirectional, 
            dropout=0.3 if num_layers > 1 else 0.0
        )
        # Если bidirectional=True, размерность умножается на 2
        out_features = hidden_dim * 2 if bidirectional else hidden_dim
        self.line = nn.Linear(in_features=out_features, out_features=num_classes)
        
    def forward(self, x):
        x = self.embeddings(x)
        output, (hidden, cell) = self.lstm(x)
        
        if self.lstm.bidirectional:
            last_layer = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            last_layer = hidden[-1]
            
        logits = self.line(last_layer)
        return logits

def clean_text(text : str) -> str: 
    return re.sub(r'[^a-zа-я1-9\s]', '', text.lower())
     
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
    lstm_model.load_state_dict(torch.load("./models/lstm/model.pth",  map_location="cpu", weights_only=False))
    lstm_model.eval()
    
    # Загрузка DistilBERT
    hf_tokenizer = AutoTokenizer.from_pretrained("./models/distilbert")
    hf_model = AutoModelForSequenceClassification.from_pretrained("./models/distilbert")
    hf_model.eval()
    
    logging.info("Обе модели загрузились успешно")
    yield
    
app = FastAPI(
    title="SMS Spam classifier API",
    description="Сравнение LSTM и DistilBERT для классификации спама",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем запросы с любых источников
    allow_credentials=True,
    allow_methods=["*"],  # Разрешаем все методы (GET, POST и т.д.)
    allow_headers=["*"],  # Разрешаем все заголовки
)

@app.get("/", response_class=HTMLResponse)
def get_root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Ошибка: файл index.html не найден в папке проекта</h1>", status_code=500)

@app.get("/health")
def health_check():
    return {
        "status" : "OK",
        "Models_loaded" : True,
        "Endpoints" : ["/predict/lstm", "/predict/distilbert"]
    }

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
    input_tensor = torch.tensor([indeces], dtype=torch.long)
    
    # Предсказание
    with torch.no_grad():
        logits = lstm_model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze().tolist()
    
    is_spam = bool(probs[1] > probs[0])
    
    request_status["total_requests"] += 1
    request_status["lstm_requests"] += 1
    request_status["spam_count"] += 1 if is_spam else 0
    request_status["ham_count"] += 1 if not is_spam else 0
    request_status["last_request"] = datetime.now().isoformat()
    
    return PredictionResponse(
        text = request.text,
        model_used="Bi-LSTM (Custom)",
        is_spam = bool(probs[1] > probs[0]),
        confidence_spam=round(probs[1], 4),
        confidence_ham=round(probs[0], 4),
    )
    
@app.post("/predict/distilbert", response_model=PredictionResponse)
def predict_distilbert(request : TextRequest):
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
    
    is_spam = bool(probs[1] > probs[0])
    
    request_status["total_requests"] += 1
    request_status["distilbert_requests"] += 1
    request_status["spam_count"] += 1 if is_spam else 0
    request_status["ham_count"] += 1 if not is_spam else 0
    request_status["last_request"] = datetime.now().isoformat()   
    
    return PredictionResponse(
        text=request.text,
        model_used="distilBERT (Hugging Face)",
        is_spam = bool(probs[1] > probs[0]),
        confidence_spam=round(probs[1], 4),
        confidence_ham=round(probs[0], 4),
    )

class CompareResponse(BaseModel):
    text : str
    lstm_prediction : PredictionResponse
    distilbert_prediction : PredictionResponse
    model_agree : bool
    final_verdict : str
    recomendation : str
    
@app.post("/predict/compare", response_model=CompareResponse)
def predict_compare(request : TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400,detail="Текст не может быть пустым")

    clean_t = clean_text(request.text)
    words = clean_t.split()
    indices = [lstm_vocab.get(w, lstm_vocab.get("<UNK>", 1)) for w in words]
    input_tensors = torch.tensor([indices], dtype=torch.long)
    
    with torch.no_grad():
        lstm_logits = lstm_model(input_tensors)
        lstm_probs = F.softmax(lstm_logits, dim = 1).squeeze().tolist()
        
    lstm_result = PredictionResponse(
        text = request.text,
        model_used="Bi-LSTM (custom)",
        is_spam=bool(lstm_probs[1] > lstm_probs[0]),
        confidence_spam=round(lstm_probs[1], 4),        
        confidence_ham=round(lstm_probs[0], 4),
    )
    
    # предсказание distilBERT
    inputs = hf_tokenizer(
       request.text,
       return_tensors = "pt",
       truncation = True,
       max_length = 64,
       padding = True 
    )
    
    with torch.no_grad():
        bert_outputs = hf_model(**inputs)
        bert_probs = F.softmax(bert_outputs.logits, dim = 1).squeeze().tolist()
        
    bert_result = PredictionResponse(
        text = request.text,
        model_used="DistilBERT (Hugging Face)",
        is_spam=bool(bert_probs[1] > bert_probs[0]),
        confidence_spam=round(bert_probs[1], 4),
        confidence_ham=round(bert_probs[0], 4),
    )
    
    model_agree = lstm_result.is_spam == bert_result.is_spam
    
    if model_agree:
        final_verdict = "SPAM" if lstm_result.is_spam else "HAM"
        recomendation = "Обе модели согласны - высокая увереннось в результате"
    else: 
        final_verdict = "SPAM" if bert_result.is_spam else "HAM"
        recomendation = "Модели разошлись во мнениях. Приоритет отдан DestilBERT (F1 = 0.97)" 
        
    request_status["total_requests"] += 1
    request_status["compare_requests"] += 1
    request_status["last_request"] = datetime.now().isoformat()
    
    return CompareResponse(
        text = request.text,
        lstm_prediction=lstm_result,
        distilbert_prediction=bert_result,
        model_agree=model_agree,
        final_verdict=final_verdict,
        recomendation=recomendation
    )
    
@app.get("/status")
def get_status():
    return {
        "total_requests": request_status["total_requests"],
        "spam_detected": request_status["spam_count"],
        "ham_detected": request_status["ham_count"],
        "lstm_requests": request_status["lstm_requests"],
        "distilbert_requests": request_status["distilbert_requests"],
        "compare_requests": request_status["compare_requests"],
        "last_request": request_status["last_request"],
        "uptime": "API работает стабильно",
    }