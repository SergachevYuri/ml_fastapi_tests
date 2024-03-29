from fastapi import FastAPI, HTTPException
from transformers import pipeline
from pydantic import BaseModel

# Модель входных данных
class Item(BaseModel):
    text: str

# Создаем обьект FastAPI
app = FastAPI()

# Загружаем предобученную модель ИИ
classifier = pipeline("sentiment-analysis")

# Корневой путь
@app.get("/")
def root():
    return {"message": "Hello World"}

# Путь для предсказания
@app.post("/predict/")
def predict(item: Item):
    # Проверяем длину текста
    if len(item.text) < 10:
        # Вызываем ошибку 400 
        raise HTTPException(status_code=400, detail="The text must contain at least 10 characters")

    # Применяем модель к тексту
    result = classifier(item.text)[0]

    # Возвращяем результат
    return result
