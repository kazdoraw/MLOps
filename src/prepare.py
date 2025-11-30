import pandas as pd
import yaml
import os
import re
import json
from sklearn.model_selection import train_test_split

def load_params():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["prepare"]

def clean_text(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def create_medical_category(text):
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['pain', 'hurt', 'ache', 'sore']):
        return 'pain'
    elif any(word in text_lower for word in ['fever', 'temperature', 'hot', 'cold', 'flu']):
        return 'fever'
    elif any(word in text_lower for word in ['skin', 'rash', 'itch', 'acne']):
        return 'skin'
    elif any(word in text_lower for word in ['stomach', 'digest', 'nausea', 'vomit']):
        return 'digestive'
    elif any(word in text_lower for word in ['breath', 'cough', 'lung', 'asthma']):
        return 'respiratory'
    elif any(word in text_lower for word in ['heart', 'chest', 'pressure', 'cardiovascular']):
        return 'cardiac'
    elif any(word in text_lower for word in ['mental', 'anxiety', 'depress', 'stress']):
        return 'mental'
    else:
        return 'general'

def prepare_data():
    params = load_params()
    
    print("Загрузка данных...")
    data_path = "data/raw/chatdoctor.json"
    
    if not os.path.exists(data_path):
        print(f"Файл {data_path} не найден. Пожалуйста, загрузите датасет.")
        return
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Загружено записей из JSON: {len(data)}")
    
    records = []
    for item in data:
        text = item.get('input', '')
        if text:
            records.append({'text': text})
    
    df = pd.DataFrame(records)
    print(f"Создано записей: {len(df)}")
    
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 10]
    print(f"После очистки коротких текстов: {len(df)}")
    
    df['category'] = df['text'].apply(create_medical_category)
    
    print(f"\nРаспределение категорий:")
    print(df['category'].value_counts())
    
    if params["sample_size"] and params["sample_size"] < len(df):
        df = df.sample(n=params["sample_size"], random_state=params["random_state"])
        print(f"\nВыбрано записей для обучения: {len(df)}")
    
    df = df[['text', 'category']]
    
    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df['category']
    )
    
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    
    print(f"\nСохранено:")
    print(f"  - train.csv: {len(train_df)} записей")
    print(f"  - test.csv: {len(test_df)} записей")
    print(f"\nФайлы готовы для обучения модели")

if __name__ == "__main__":
    prepare_data()
