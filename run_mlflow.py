#!/usr/bin/env python3
"""
Скрипт для запуска MLflow UI
Использование: python run_mlflow.py
"""
import subprocess
import sys
import os

# Переходим в директорию проекта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("Запуск MLflow UI...")
print("Директория:", os.getcwd())
print("База данных: sqlite:///mlflow.db")
print("Адрес: http://127.0.0.1:5000")
print("\nДля остановки нажмите Ctrl+C")
print("=" * 60)

# Запускаем MLflow UI
try:
    subprocess.run([
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--port", "5000"
    ])
except KeyboardInterrupt:
    print("\n\nMLflow UI остановлен")
except Exception as e:
    print(f"\nОшибка: {e}")
    print("\nУбедитесь, что MLflow установлен:")
    print("   pip install mlflow")
