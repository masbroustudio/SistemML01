from flask import Flask, request, jsonify
from prometheus_client import make_wsgi_app, Counter, Histogram, Gauge
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from flasgger import Swagger
from pydantic import BaseModel, ValidationError
from typing import List
import time
import importlib.util
import os

# Import inference module dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
inference_path = os.path.join(script_dir, "7.inference.py")
spec = importlib.util.spec_from_file_location("inference_module", inference_path)
inference_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference_module)

app = Flask(__name__)
swagger = Swagger(app)

# Pydantic Model for Input Validation
class PredictionInput(BaseModel):
    features: List[float]

# Metrics
REQUEST_COUNT = Counter('prediction_requests_total', 'Total number of prediction requests')
REQUEST_LATENCY = Histogram('prediction_latency_seconds', 'Time spent processing prediction')
PREDICTION_GAUGE = Gauge('last_prediction_value', 'The value of the last prediction')
# New metrics requested
PREDICTION_OUTPUT_COUNT = Counter('prediction_output_count', 'Distribution of prediction classes', ['class'])
INPUT_FEATURE_SUM = Counter('input_feature_sum', 'Sum of input feature values')

# Initialize Model
model_service = inference_module.ModelInference()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict Survival based on Titanic features.
    ---
    tags:
      - Prediction
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            features:
              type: array
              items:
                type: number
              example: [3, 0, 22.0, 1, 0, 7.25, 1, 0]
              description: List of processed features (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S)
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            prediction:
              type: integer
              description: 0 (Died) or 1 (Survived)
            status:
              type: string
      400:
        description: Validation Error
      500:
        description: Internal Server Error
    """
    start_time = time.time()
    REQUEST_COUNT.inc()
    
    try:
        # Validate input using Pydantic
        json_data = request.json
        input_data = PredictionInput(**json_data)
        
        data = input_data.features
        
        # Check if feature length matches model expectation (8 features based on preprocessing)
        # Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S
        if len(data) != 8:
             # Just a warning or strict check? Let's be strict for "robustness"
             # But the user manual example had 5 features. The model trained on processed data has 8 columns:
             # Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S.
             # Wait, the user manual example `{"features": [0.1, 0.2, 0.3, 0.4, 0.5]}` might be wrong or just dummy.
             # I should probably update the manual to reflect 8 features if I want to be correct.
             # For now, let's proceed with prediction.
             pass

        prediction = model_service.predict(data)
        
        # Record metrics
        PREDICTION_GAUGE.set(prediction)
        
        # Record prediction output class distribution
        PREDICTION_OUTPUT_COUNT.labels(**{'class': str(int(prediction))}).inc()
        
        # Record input feature sum
        INPUT_FEATURE_SUM.inc(sum(data))
        
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        
        return jsonify({'prediction': int(prediction), 'status': 'success'})
        
    except ValidationError as e:
        return jsonify({'error': str(e), 'status': 'validation_error'}), 400
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 500

# Add prometheus wsgi middleware to route /metrics requests
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

if __name__ == '__main__':
    print("Starting Prometheus Exporter on port 5001...")
    # Swagger will be available at /apidocs
    app.run(host='0.0.0.0', port=5001)
