import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dagshub

def train_with_tuning():
    # Inisialisasi DagsHub
    dagshub.init(repo_owner='masbroumail', repo_name='SistemML01', mlflow=True)
    
    # Muat data
    print("Loading data...")
    try:
        df = pd.read_csv('train_processed.csv')
    except FileNotFoundError:
        df = pd.read_csv('Membangun_model/train_processed.csv')
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("Titanic_Tuned_Model")
    
    # Hyperparameter untuk di-tuning
    n_estimators_list = [50, 100]
    max_depth_list = [5, 10]
    
    best_acc = 0
    best_params = {}
    
    for n_est in n_estimators_list:
        for depth in max_depth_list:
            with mlflow.start_run(run_name=f"RF_n{n_est}_d{depth}"):
                # Logging Manual: Parameter
                params = {"n_estimators": n_est, "max_depth": depth}
                mlflow.log_params(params)
                
                model = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                # Logging Manual: Metrik (Sama seperti autolog)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1_score", f1)
                
                print(f"Run: n_est={n_est}, depth={depth} -> Accuracy: {acc}")
                
                # Advanced: Log Artefak (Confusion Matrix)
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6,4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                
                # Simpan plot sementara
                os.makedirs("temp_artifacts", exist_ok=True)
                cm_path = "temp_artifacts/confusion_matrix.png"
                plt.savefig(cm_path)
                plt.close()
                
                # Log artefak
                mlflow.log_artifact(cm_path)
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                if acc > best_acc:
                    best_acc = acc
                    best_params = params

    print(f"Best Accuracy: {best_acc} with params {best_params}")

if __name__ == "__main__":
    train_with_tuning()
