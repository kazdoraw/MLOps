# MLOps HW1: ML Pipeline с DVC и MLflow

[![GitHub](https://img.shields.io/badge/GitHub-MLOps-blue?logo=github)](https://github.com/kazdoraw/MLOps)
[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-945DD6?logo=dvc)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?logo=mlflow)](https://mlflow.org)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)

## Репозиторий

**GitHub**: https://github.com/kazdoraw/MLOps.git

## Быстрый старт

```bash
git clone https://github.com/kazdoraw/MLOps.git
cd MLOps/HW1
pip install -r requirements.txt
jupyter notebook hw1_final.ipynb
```

Подробная инструкция: **[INSTALL.md](INSTALL.md)**

## Цель проекта

Создание воспроизводимого ML-pipeline для **бинарной классификации медицинских консультаций**: определение необходимости очного визита к врачу на основе текста диалога пациент-врач из датасета ChatDoctor.

### Задача классификации

- **Класс 0**: Консультация может быть решена онлайн
- **Класс 1**: Требуется офлайн-визит к врачу

Метки генерируются автоматически на основе ключевых фраз в ответах врачей (visit, consult, emergency, etc.)

## Структура проекта

```
HW1/
├── data/
│   ├── raw/              # Исходные данные из Kaggle (версионируются через DVC)
│   └── processed/        # Обработанные данные train/test (версионируются через DVC)
├── src/
│   ├── prepare.py        # Парсинг JSON/CSV, генерация меток, train/test split
│   └── train.py          # TF-IDF + RandomForest/LogisticRegression + MLflow
├── models/               # Сохраненные модели и графики (версионируются через DVC)
├── dvc.yaml              # Описание пайплайна DVC (2 стадии)
├── params.yaml           # Гиперпараметры модели и TF-IDF
├── requirements.txt      # Зависимости проекта
└── README.md             # Документация
```

## Как запустить

### Вариант 1: Jupyter Notebook (рекомендуется)

```bash
git clone https://github.com/kazdoraw/MLOps.git
cd MLOps/HW1
pip install -r requirements.txt
jupyter notebook hw1_final.ipynb
```

Выполните ячейки последовательно - notebook автоматически:
- Загрузит данные из Kaggle
- Инициализирует DVC
- Выполнит предобработку
- Обучит модель
- Залогирует результаты в MLflow

### Вариант 2: DVC Pipeline

```bash
git clone https://github.com/kazdoraw/MLOps.git
cd MLOps/HW1
pip install -r requirements.txt

# После загрузки данных через notebook:
dvc repro

# Запуск MLflow UI
./start_mlflow.sh
```

Откройте браузер: **http://127.0.0.1:5001**

**Полная документация**: [INSTALL.md](INSTALL.md) | [QUICKSTART.md](QUICKSTART.md)

## Описание пайплайна

Пайплайн состоит из двух основных стадий:

1. **prepare** (src/prepare.py):
   - Загрузка датасета ChatDoctor (JSON/CSV)
   - Парсинг структуры: извлечение текстов пациентов и врачей
   - Генерация меток: автоматическая разметка на основе ключевых фраз
   - Формирование датасета: [text, label]
   - Train/test split (80/20) со стратификацией
   - Сохранение в data/processed/

2. **train** (src/train.py):
   - Загрузка обработанных данных
   - Построение Pipeline:
     * TfidfVectorizer (max_features=5000, ngrams=(1,2))
     * Классификатор (RandomForest или LogisticRegression)
   - Обучение модели
   - Оценка метрик: accuracy, precision, recall, f1-score
   - Создание confusion matrix
   - Логирование всех параметров/метрик/артефактов в MLflow
   - Сохранение модели в models/

## Технологии

- **DVC** - версионирование данных и пайплайнов
- **MLflow** - трекинг экспериментов и моделей
- **scikit-learn** - машинное обучение
- **pandas** - обработка данных

## MLflow UI

```bash
./start_mlflow.sh
# или
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Интерфейс доступен: **http://127.0.0.1:5001**

В UI вы увидите:
- Все запуски экспериментов с параметрами
- Метрики качества (accuracy, precision, recall, f1-score)
- Метрики для класса 1 (офлайн-визит)
- Артефакты (confusion matrix, модели)
- Сравнение экспериментов

## Результаты

С использованием `class_weight='balanced'` для обработки дисбаланса классов:

| Метрика | Класс 0 | Класс 1 (офлайн-визит) |
|---------|---------|------------------------|
| Precision | 0.97 | 0.86 |
| Recall | 0.99 | 0.66 |
| F1-score | 0.98 | 0.75 |

**Общая accuracy**: 96.7%

Распределение классов:
- Train: 36,985 (класс 0) / 3,015 (класс 1)
- Test: 9,246 (класс 0) / 754 (класс 1)

## Документация

- **[INSTALL.md](INSTALL.md)** - Полная инструкция по установке и настройке
- **[QUICKSTART.md](QUICKSTART.md)** - Краткое руководство
- **[MLFLOW_GUIDE.md](MLFLOW_GUIDE.md)** - Работа с MLflow
- **[hw1.md](hw1.md)** - Описание задания

## Лицензия

MIT License
