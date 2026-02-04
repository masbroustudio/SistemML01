import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model():
    # Load data
    print("Loading data...")
    try:
        df = pd.read_csv('train_processed.csv')
    except FileNotFoundError:
        # Fallback if running from root
        df = pd.read_csv('Membangun_model/train_processed.csv')
    
    X = df.drop('Survived', axis=1)
    # Cast to float to avoid MLflow schema warning about integer columns with potential missing values
    X = X.astype(float)
    y = df['Survived']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Enable autolog
    mlflow.sklearn.autolog()
    
    # Check if running via 'mlflow run' (MLFLOW_RUN_ID will be set)
    if os.environ.get("MLFLOW_RUN_ID"):
        print("Running within MLflow Project context...")
        # Do NOT set experiment, as it's already determined by mlflow run
        # NOTE: When MLFLOW_RUN_ID is present, mlflow.start_run() will resume that run automatically
        # We should not create a nested run if the intention is to use the project run
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc}")
        
        # Explicitly log model to ensure it's captured even if autolog fails on permissions
        # But be careful not to duplicate if autolog works
        # signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        # mlflow.sklearn.log_model(model, "model", signature=signature)
            
    # Check if we are already in an active run (e.g. invoked via code wrapper)
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
        # Set experiment only if running locally/manually
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
