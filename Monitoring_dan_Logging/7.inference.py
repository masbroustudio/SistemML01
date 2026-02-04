import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class ModelInference:
    def __init__(self):
        # In a real scenario, load the model from artifact
        # self.model = mlflow.sklearn.load_model("...")
        # For demonstration, we'll train a dummy model on init
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        # Dummy fit to allow prediction
        # Updated to 8 features to match Titanic preprocessing output
        # (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S)
        X_dummy = np.random.rand(10, 8) 
        y_dummy = np.random.randint(0, 2, 10)
        self.model.fit(X_dummy, y_dummy)
        
    def predict(self, data):
        """
        Predicts using the loaded model.
        Args:
            data (list or np.array): Input features.
        Returns:
            int: Prediction result.
        """
        # Ensure input is 2D
        data = np.array(data).reshape(1, -1)
        return self.model.predict(data)[0]

if __name__ == "__main__":
    inference = ModelInference()
    # Updated sample data to 8 features
    sample_data = [1, 0, 22.0, 1, 0, 7.25, 1, 0]
    print(f"Prediction for {sample_data}: {inference.predict(sample_data)}")
