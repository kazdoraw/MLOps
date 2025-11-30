#!/bin/bash
# Скрипт для запуска MLflow UI в окружении ml-python312

echo "Запуск MLflow UI..."
cd "$(dirname "$0")"

# Активация conda окружения ml-python312
eval "$(conda shell.bash hook)"
conda activate ml-python312

# Проверка активации
echo "Окружение: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Python: $(which python)"
echo "MLflow версия: $(python -m mlflow --version)"
echo ""
echo "MLflow UI будет доступен на: http://127.0.0.1:5001"
echo "Для остановки нажмите Ctrl+C"
echo "=" * 60

# Запуск MLflow UI на порту 5001
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001 --host 127.0.0.1
