import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

def load_params():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["train"]

def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """
    Создает и сохраняет матрицу ошибок.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Confusion matrix сохранена: {save_path}")

def train_model():
    params = load_params()
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("chatdoctor_classification")
    
    print("Загрузка обработанных данных...")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    X_train = train_df['text']
    y_train = train_df['label']
    X_test = test_df['text']
    y_test = test_df['label']
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Распределение классов в train: {y_train.value_counts().to_dict()}")
    print(f"Распределение классов в test: {y_test.value_counts().to_dict()}")
    
    with mlflow.start_run():
        print("\nОбучение модели...")
        
        if params["model_type"] == "RandomForestClassifier":
            classifier = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=params["random_state"],
                max_features=params.get("max_features", "sqrt"),
                class_weight='balanced',  # Балансировка классов!
                n_jobs=-1,
                verbose=1
            )
        elif params["model_type"] == "LogisticRegression":
            classifier = LogisticRegression(
                max_iter=params.get("max_iter", 1000),
                random_state=params["random_state"],
                class_weight='balanced',  # Балансировка классов!
                n_jobs=-1,
                verbose=1
            )
        else:
            raise ValueError(f"Неизвестный тип модели: {params['model_type']}")
        
        # Конвертируем ngram_range из списка в tuple
        ngram_range = params.get("ngram_range", [1, 2])
        if isinstance(ngram_range, list):
            ngram_range = tuple(ngram_range)
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=params.get("tfidf_max_features", 5000),
                ngram_range=ngram_range,
                min_df=params.get("min_df", 2),
                max_df=params.get("max_df", 0.8)
            )),
            ('clf', classifier)
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        
        precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        
        # Метрики для каждого класса
        precision_per_class = precision_score(y_test, y_pred_test, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred_test, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred_test, average=None, zero_division=0)
        
        print(f"\nМетрики на Train:")
        print(f"  Accuracy:  {accuracy_train:.4f}")
        
        print(f"\nМетрики на Test:")
        print(f"  Accuracy:  {accuracy_test:.4f}")
        print(f"  Precision (weighted): {precision:.4f}")
        print(f"  Recall (weighted):    {recall:.4f}")
        print(f"  F1-score (weighted):  {f1:.4f}")
        
        print(f"\nМетрики для класса 1 (офлайн визит):")
        print(f"  Precision: {precision_per_class[1]:.4f}")
        print(f"  Recall:    {recall_per_class[1]:.4f}")
        print(f"  F1-score:  {f1_per_class[1]:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test, zero_division=0))
        
        mlflow.log_param("model_type", params["model_type"])
        mlflow.log_param("n_estimators", params.get("n_estimators", "N/A"))
        mlflow.log_param("max_depth", params.get("max_depth", "N/A"))
        mlflow.log_param("max_features", params.get("max_features", "N/A"))
        mlflow.log_param("random_state", params["random_state"])
        mlflow.log_param("tfidf_max_features", params.get("tfidf_max_features", 5000))
        
        mlflow.log_metric("accuracy_train", accuracy_train)
        mlflow.log_metric("accuracy_test", accuracy_test)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Метрики для класса 1 (важные!)
        mlflow.log_metric("precision_class_1", precision_per_class[1])
        mlflow.log_metric("recall_class_1", recall_per_class[1])
        mlflow.log_metric("f1_score_class_1", f1_per_class[1])
        
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.pkl"
        joblib.dump(pipeline, model_path)
        
        cm_path = "models/confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred_test, cm_path)
        
        mlflow.sklearn.log_model(pipeline, "model")
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(cm_path)
        
        print(f"\nМодель сохранена: {model_path}")
        print(f"Артефакты залогированы в MLflow")

if __name__ == "__main__":
    train_model()
