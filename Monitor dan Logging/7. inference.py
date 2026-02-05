import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class ModelInference:
    def __init__(self):
        # Dalam skenario nyata, muat model dari artefak
        # self.model = mlflow.sklearn.load_model("...")
        # Untuk demonstrasi, kita akan melatih model dummy saat init
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        # Fit dummy untuk memungkinkan prediksi
        # Diperbarui ke 8 fitur untuk mencocokkan output preprocessing Titanic
        # (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S)
        X_dummy = np.random.rand(10, 8) 
        y_dummy = np.random.randint(0, 2, 10)
        self.model.fit(X_dummy, y_dummy)
        
    def predict(self, data):
        """
        Prediksi menggunakan model yang dimuat.
        Args:
            data (list atau np.array): Fitur input.
        Returns:
            int: Hasil prediksi.
        """
        # Pastikan input 2D
        data = np.array(data).reshape(1, -1)
        return self.model.predict(data)[0]

if __name__ == "__main__":
    inference = ModelInference()
    # Data sampel diperbarui menjadi 8 fitur
    sample_data = [1, 0, 22.0, 1, 0, 7.25, 1, 0]
    print(f"Prediction for {sample_data}: {inference.predict(sample_data)}")
