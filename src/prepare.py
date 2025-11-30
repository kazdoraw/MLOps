import pandas as pd
import yaml
import os
import json
import re
from sklearn.model_selection import train_test_split

def load_params():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["prepare"]

def detect_offline_visit_needed(text):
    """
    Определяет, нужен ли офлайн визит на основе ключевых фраз в ответе врача.
    """
    if not isinstance(text, str):
        return 0
    
    text_lower = text.lower()
    
    offline_keywords = [
        'visit', 'consult', 'see a doctor', 'emergency', 'hospital',
        'clinic', 'physical exam', 'in person', 'appointment',
        'follow up', 'follow-up', 'examination', 'urgent care',
        'er', 'physician', 'specialist', 'office visit'
    ]
    
    for keyword in offline_keywords:
        if keyword in text_lower:
            return 1
    
    return 0

def parse_json_file(file_path):
    """
    Парсит JSON файл с диалогами врач-пациент.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            data = [json.loads(line) for line in lines if line.strip()]
        return data

def prepare_data():
    params = load_params()
    
    print("Загрузка данных...")
    
    json_files = []
    raw_dir = "data/raw"
    
    if os.path.exists(os.path.join(raw_dir, "chatdoctor.csv")):
        csv_path = os.path.join(raw_dir, "chatdoctor.csv")
        df = pd.read_csv(csv_path)
        print(f"Загружен CSV файл: {len(df)} записей")
    else:
        for file in os.listdir(raw_dir):
            if file.endswith('.json'):
                json_files.append(os.path.join(raw_dir, file))
        
        if not json_files:
            print("Не найдено JSON или CSV файлов в data/raw/")
            return
        
        print(f"Найдено JSON файлов: {len(json_files)}")
        
        all_data = []
        for json_file in json_files:
            try:
                data = parse_json_file(json_file)
                all_data.extend(data if isinstance(data, list) else [data])
            except Exception as e:
                print(f"Ошибка при чтении {json_file}: {e}")
        
        print(f"Загружено записей из JSON: {len(all_data)}")
        
        texts = []
        labels = []
        
        for item in all_data:
            if isinstance(item, dict):
                patient_text = item.get('input', item.get('question', item.get('patient', '')))
                doctor_text = item.get('output', item.get('answer', item.get('doctor', '')))
                
                combined_text = f"Patient: {patient_text} Doctor: {doctor_text}"
                label = detect_offline_visit_needed(doctor_text)
                
                texts.append(combined_text)
                labels.append(label)
        
        df = pd.DataFrame({
            'text': texts,
            'label': labels
        })
    
    if 'text' not in df.columns and len(df.columns) >= 2:
        text_col = df.columns[0]
        label_col = df.columns[-1] if len(df.columns) > 1 else None
        
        if label_col:
            df['label'] = df[label_col].apply(lambda x: detect_offline_visit_needed(str(x)))
        else:
            df['label'] = 0
        
        df['text'] = df[text_col].astype(str)
        df = df[['text', 'label']]
    
    df = df.dropna()
    print(f"После очистки пропусков: {len(df)}")
    print(f"Распределение меток: {df['label'].value_counts().to_dict()}")
    
    if params["sample_size"] and params["sample_size"] < len(df):
        df = df.sample(n=params["sample_size"], random_state=params["random_state"])
        print(f"Выбрано записей для обучения: {len(df)}")
    
    if len(df['label'].unique()) > 1:
        stratify = df['label']
    else:
        stratify = None
        print("Внимание: только один класс в данных, stratify отключен")
    
    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=stratify
    )
    
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    
    print(f"\nСохранено:")
    print(f"  - train.csv: {len(train_df)} записей")
    print(f"  - test.csv: {len(test_df)} записей")
    print(f"\nРаспределение в train: {train_df['label'].value_counts().to_dict()}")
    print(f"Распределение в test: {test_df['label'].value_counts().to_dict()}")

if __name__ == "__main__":
    prepare_data()
