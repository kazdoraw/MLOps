import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

def load_params():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["train"]

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix сохранена: {save_path}")

def plot_feature_importance(model, save_path, top_n=20):
    if hasattr(model.named_steps['clf'], 'feature_importances_'):
        importances = model.named_steps['clf'].feature_importances_
        feature_names = model.named_steps['tfidf'].get_feature_names_out()
        
        indices = importances.argsort()[-top_n:][::-1]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_importances)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Feature importance сохранена: {save_path}")

def train_model():
    params = load_params()
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("chatdoctor_classification")
    
    print("Загрузка обработанных данных...")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    X_train = train_df['text']
    y_train = train_df['category']
    X_test = test_df['text']
    y_test = test_df['category']
    
    labels = sorted(y_train.unique())
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Категории: {labels}")
    
    with mlflow.start_run():
        print("\nОбучение модели...")
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )),
            ('clf', RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=params["random_state"],
                max_features=params["max_features"],
                n_jobs=-1
            ))
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\nМетрики:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=labels))
        
        mlflow.log_param("model_type", params["model_type"])
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_param("max_features", params["max_features"])
        mlflow.log_param("random_state", params["random_state"])
        mlflow.log_param("num_categories", len(labels))
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        os.makedirs("models", exist_ok=True)
        os.makedirs("models/plots", exist_ok=True)
        
        cm_path = "models/plots/confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, labels, cm_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")
        
        fi_path = "models/plots/feature_importance.png"
        plot_feature_importance(pipeline, fi_path)
        mlflow.log_artifact(fi_path, artifact_path="plots")
        
        model_path = "models/model.pkl"
        joblib.dump(pipeline, model_path)
        
        mlflow.sklearn.log_model(pipeline, "model")
        mlflow.log_artifact(model_path)
        
        print(f"\nМодель сохранена: {model_path}")
        print(f"Визуализации сохранены в: models/plots/")

if __name__ == "__main__":
    train_model()
