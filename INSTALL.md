# Установка и быстрый старт

## Требования

- Python 3.8+
- Git
- (Опционально) Conda для изолированного окружения

## Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/kazdoraw/MLOps.git
cd MLOps/HW1
```

### 2. Создание виртуального окружения (рекомендуется)

**Вариант A: Python venv**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

**Вариант B: Conda**
```bash
conda create -n mlops python=3.12
conda activate mlops
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Загрузка данных

Данные автоматически загружаются из Kaggle при первом запуске notebook или можно загрузить вручную:

```bash
# Откройте hw1_final.ipynb и запустите ячейку 2 для загрузки данных
# или используйте kagglehub напрямую
```

## Быстрый старт

### Вариант 1: Jupyter Notebook (рекомендуется для первого запуска)

```bash
jupyter notebook hw1_final.ipynb
```

Выполните все ячейки последовательно. Notebook:
- Загрузит данные из Kaggle
- Инициализирует DVC
- Выполнит предобработку данных
- Обучит модель
- Залогирует результаты в MLflow

### Вариант 2: DVC Pipeline (для воспроизведения)

После первого запуска notebook, когда данные уже загружены:

```bash
# Запуск всего пайплайна
dvc repro

# Только подготовка данных
dvc repro prepare

# Только обучение модели
dvc repro train
```

### Вариант 3: Прямой запуск скриптов

```bash
# Предобработка
python src/prepare.py

# Обучение
python src/train.py
```

## Просмотр результатов

### MLflow UI

```bash
# Запуск MLflow UI
./start_mlflow.sh

# Или вручную
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Откройте браузер: **http://127.0.0.1:5001**

В UI вы увидите:
- Все запуски экспериментов
- Параметры моделей
- Метрики (accuracy, precision, recall, f1-score)
- Артефакты (confusion matrix, модели)

### Файлы с результатами

После обучения:
- `models/model.pkl` - обученная модель
- `models/confusion_matrix.png` - матрица ошибок
- `mlflow.db` - база данных с экспериментами

## Изменение параметров

Отредактируйте `params.yaml`:

```yaml
prepare:
  test_size: 0.2
  random_state: 42
  sample_size: 50000  # Размер выборки для обучения

train:
  model_type: RandomForest  # или LogisticRegression
  n_estimators: 100
  max_depth: 15
  random_state: 42
  max_features: sqrt
  tfidf_max_features: 5000
  ngram_range: [1, 2]
  min_df: 2
  max_df: 0.95
```

После изменения параметров:

```bash
dvc repro  # Переобучит только необходимые этапы
```

## Структура проекта

```
HW1/
├── data/
│   ├── raw/                    # Исходные данные (из Kaggle)
│   │   ├── HealthCareMagic-100k.json
│   │   ├── iCliniq.json
│   │   └── chatdoctor.csv      # Объединенный CSV
│   └── processed/              # Обработанные данные
│       ├── train.csv
│       └── test.csv
├── models/                     # Модели и артефакты
│   ├── model.pkl
│   └── confusion_matrix.png
├── src/
│   ├── prepare.py             # Предобработка данных
│   └── train.py               # Обучение модели
├── hw1_final.ipynb            # Jupyter notebook
├── dvc.yaml                   # DVC пайплайн
├── params.yaml                # Параметры
├── requirements.txt           # Зависимости Python
├── start_mlflow.sh            # Скрипт запуска MLflow UI
└── README.md                  # Основная документация
```

## Решение проблем

### Ошибка: MLflow UI не запускается

```bash
# Проверьте, не занят ли порт
lsof -i :5001

# Используйте другой порт
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5002
```

### Ошибка: DVC не находит данные

```bash
# Убедитесь, что данные загружены
ls data/raw/

# Должны быть файлы:
# - HealthCareMagic-100k.json
# - iCliniq.json
# - chatdoctor.csv
```

### Ошибка: Модуль не найден

```bash
# Переустановите зависимости
pip install -r requirements.txt --force-reinstall
```

## Дополнительная информация

- **README.md** - Описание проекта и архитектуры
- **QUICKSTART.md** - Краткое руководство
- **MLFLOW_GUIDE.md** - Подробное руководство по MLflow
