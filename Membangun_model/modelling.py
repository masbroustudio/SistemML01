import pandas as pd
import mlflow
import mlflow.sklearn
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
    # Ubah ke float untuk hindari peringatan skema MLflow tentang kolom integer
    X = X.astype(float)
    y = df['Survived']
    
    # Bagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Set eksperimen
    mlflow.set_experiment("Titanic_Basic_Model")
    
    # Aktifkan autolog
    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        print("Training model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi (Autolog menangani ini, tapi bagus untuk dicetak)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc}")
        
        # Infer signature (opsional tapi praktik yang baik)
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)

if __name__ == "__main__":
    train_model()
