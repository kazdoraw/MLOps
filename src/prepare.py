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
    Определяет необходимость офлайн визита на основе КОНКРЕТНЫХ фраз.
    Использует regex для точного поиска рекомендаций визита.
    """
    if not isinstance(text, str):
        return 0
    
    text_lower = text.lower()
    
    # Срочные случаи
    urgent_patterns = [
        r'\b(urgent|immediate|emergency)\s+(care|visit|consultation|exam)',
        r'\bsee\s+(a\s+)?doctor\s+(immediately|urgently|soon|asap)',
        r'\bvisit\s+(the\s+)?(hospital|emergency|er|clinic)',
        r'\b(emergency\s+room|er\s+visit)',
        r'\bgo\s+to\s+(the\s+)?(hospital|emergency|doctor)',
        r'\bseek\s+(immediate|urgent|medical)\s+(attention|help|care)',
    ]
    
    # Рекомендации визита
    visit_recommendations = [
        r'\brecommend.{0,30}(see|visit|consult).{0,20}(doctor|physician|specialist)',
        r'\badvise.{0,30}(see|visit|consult).{0,20}(doctor|physician)',
        r'\bsuggest.{0,30}(see|visit|consult).{0,20}(doctor|physician)',
        r'\bshould\s+(see|visit|consult)\s+(a\s+)?(doctor|physician|specialist)',
        r'\bneed\s+to\s+(see|visit|consult)\s+.{0,20}(doctor|physician|specialist)',
        r'\bphysical\s+examination\s+(is\s+)?(required|needed|necessary|recommended)',
        r'\bin[- ]person\s+(visit|consultation|exam)',
        r'\b(follow[- ]up|followup)\s+.{0,20}(visit|appointment|with\s+doctor)',
        r'\bconsult\s+(a\s+|your\s+)?(doctor|physician|specialist)',
        r'\bget\s+.{0,20}(checked|examined|evaluated)\s+by\s+(a\s+)?doctor',
    ]
    
    for pattern in urgent_patterns:
        if re.search(pattern, text_lower):
            return 1
    
    for pattern in visit_recommendations:
        if re.search(pattern, text_lower):
            return 1
    
    return 0

def parse_json_to_csv():
    """
    Парсит JSON файлы и создает единый CSV файл с метками.
    """
    raw_dir = "data/raw"
    all_data = []
    
    print("Парсинг JSON файлов...")
    for file in sorted(os.listdir(raw_dir)):
        if file.endswith('.json'):
            filepath = os.path.join(raw_dir, file)
            print(f"  Обработка {file}...")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
                    print(f"    Загружено записей: {len(data) if isinstance(data, list) else 1}")
                except Exception as e:
                    print(f"    Ошибка: {e}")
    
    print(f"Всего записей: {len(all_data)}")
    
    # Создание датасета
    texts = []
    labels = []
    
    for item in all_data:
        if isinstance(item, dict):
            patient_text = item.get('input', item.get('question', item.get('patient', '')))
            doctor_text = item.get('output', 
                                   item.get('answer', 
                                   item.get('answer_icliniq',
                                   item.get('doctor', ''))))
            
            if patient_text and doctor_text:
                combined_text = f"{patient_text} {doctor_text}"
                label = detect_offline_visit_needed(doctor_text)
                
                texts.append(combined_text)
                labels.append(label)
    
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    print(f"Создан датасет: {df.shape}")
    print(f"Распределение меток: {df['label'].value_counts().to_dict()}")
    
    # Сохранение единого CSV
    csv_path = os.path.join(raw_dir, "chatdoctor.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Сохранено в {csv_path}\n")
    
    return df

def prepare_data():
    """
    Основная функция подготовки данных:
    1. Парсит JSON → создает chatdoctor.csv
    2. Делает train/test split
    """
    params = load_params()
    
    # Шаг 1: Парсинг JSON → CSV (если CSV еще нет)
    csv_path = "data/raw/chatdoctor.csv"
    if not os.path.exists(csv_path):
        df = parse_json_to_csv()
    else:
        print(f"Загрузка существующего CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Загружено записей: {len(df)}")
        print(f"Распределение меток: {df['label'].value_counts().to_dict()}\n")
    
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
