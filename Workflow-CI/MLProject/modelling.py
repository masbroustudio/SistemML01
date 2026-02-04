import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model():
    # Muat data
    print("Loading data...")
    try:
        df = pd.read_csv('train_processed.csv')
    except FileNotFoundError:
        # Fallback jika dijalankan dari root
        df = pd.read_csv('Membangun_model/train_processed.csv')
    
    X = df.drop('Survived', axis=1)
    # Ubah ke float untuk hindari peringatan skema MLflow
    X = X.astype(float)
    y = df['Survived']
    
    # Bagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Aktifkan autolog
    mlflow.sklearn.autolog()
    
    # Cek jika berjalan via 'mlflow run' (MLFLOW_RUN_ID akan diset)
    if os.environ.get("MLFLOW_RUN_ID"):
        print("Running within MLflow Project context...")
        # JANGAN set eksperimen, karena sudah ditentukan oleh mlflow run
        # CATATAN: Saat MLFLOW_RUN_ID ada, mlflow.start_run() akan melanjutkan run itu
        # Jangan buat nested run jika tujuannya menggunakan project run
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc}")
        
        # Log model secara eksplisit untuk memastikan tercatat
        # Tapi hati-hati jangan duplikasi jika autolog bekerja
        # signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        # mlflow.sklearn.log_model(model, "model", signature=signature)
            
    # Cek jika kita sudah dalam run aktif (misal via code wrapper)
    elif mlflow.active_run():
        print("Training model in existing run...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc}")
        
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)
    else:
        # Set eksperimen hanya jika berjalan lokal/manual
        mlflow.set_experiment("Titanic_Basic_Model")
        with mlflow.start_run():
            print("Training model in new run...")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {acc}")
            
            signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(model, "model", signature=signature)

if __name__ == "__main__":
    train_model()
