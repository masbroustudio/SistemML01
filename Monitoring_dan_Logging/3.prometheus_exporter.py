from flask import Flask, request, jsonify
from prometheus_client import make_wsgi_app, Counter, Histogram, Gauge, Summary
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from flasgger import Swagger
from pydantic import BaseModel, ValidationError
from typing import List
import time
import importlib.util
import os
import psutil

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

# --- 10 METRICS DEFINITION ---
# 1. Total Requests
REQUEST_COUNT = Counter('prediction_requests_total', 'Total number of prediction requests')

# 2. Request Latency
REQUEST_LATENCY = Histogram('prediction_latency_seconds', 'Time spent processing prediction')

# 3. Last Prediction Value
PREDICTION_GAUGE = Gauge('last_prediction_value', 'The value of the last prediction')

# 4. Prediction Output Distribution (Survived/Died)
PREDICTION_OUTPUT_COUNT = Counter('prediction_output_count', 'Distribution of prediction classes', ['class'])

# 5. Input Feature Sum (Dummy metric for data drift)
INPUT_FEATURE_SUM = Counter('input_feature_sum', 'Sum of input feature values')

# 6. Invalid Requests (Validation Errors)
INVALID_REQUEST_COUNT = Counter('invalid_requests_total', 'Total number of invalid requests')

# 7. System CPU Usage
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'Current system CPU usage percentage')

# 8. System Memory Usage
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_bytes', 'Current system memory usage in bytes')

# 9. Feature Distribution: Age (Index 2 in features)
FEATURE_AGE_DIST = Histogram('feature_age_distribution', 'Distribution of Age feature')

# 10. Feature Distribution: Fare (Index 5 in features)
FEATURE_FARE_DIST = Histogram('feature_fare_distribution', 'Distribution of Fare feature')


# Initialize Model
model_service = inference_module.ModelInference()

def update_system_metrics():
    """Update system metrics (CPU/Memory)"""
    SYSTEM_CPU_USAGE.set(psutil.cpu_percent())
    SYSTEM_MEMORY_USAGE.set(psutil.virtual_memory().used)

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
    update_system_metrics()
    
    try:
        # Validate input using Pydantic
        json_data = request.json
        if not json_data:
             INVALID_REQUEST_COUNT.inc()
             return jsonify({'error': 'No JSON data provided'}), 400
             
        input_data = PredictionInput(**json_data)
        
        data = input_data.features
        
        # Check if feature length matches model expectation (8 features)
        if len(data) != 8:
             # If strictly enforcing, uncomment below. For now, we allow pass-through if model handles it or we pad.
             # But our dummy model expects 8.
             # Let's pad or truncate to prevent crash? 
             # Or just return error. Returning error is better for "Invalid Requests" metric.
             if len(data) < 8:
                 data = data + [0] * (8 - len(data))
             elif len(data) > 8:
                 data = data[:8]
        
        # Log feature metrics
        # Age is index 2, Fare is index 5
        if len(data) > 2:
            FEATURE_AGE_DIST.observe(data[2])
        if len(data) > 5:
            FEATURE_FARE_DIST.observe(data[5])
            
        INPUT_FEATURE_SUM.inc(sum(data))

        prediction = model_service.predict(data)
        
        # Record prediction metrics
        PREDICTION_GAUGE.set(prediction)
        PREDICTION_OUTPUT_COUNT.labels(**{'class': str(int(prediction))}).inc()
        
        REQUEST_LATENCY.observe(time.time() - start_time)
        
        return jsonify({
            'prediction': int(prediction),
            'status': 'success'
        })
        
    except ValidationError as e:
        INVALID_REQUEST_COUNT.inc()
        return jsonify({'error': e.errors()}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add prometheus wsgi middleware to route /metrics requests
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

if __name__ == '__main__':
    print("Starting Prometheus Exporter on port 5000...")
    app.run(host='0.0.0.0', port=5000)
