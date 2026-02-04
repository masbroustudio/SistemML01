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

# Impor modul inferensi secara dinamis
script_dir = os.path.dirname(os.path.abspath(__file__))
inference_path = os.path.join(script_dir, "7.inference.py")
spec = importlib.util.spec_from_file_location("inference_module", inference_path)
inference_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference_module)

app = Flask(__name__)
swagger = Swagger(app)

# Model Pydantic untuk Validasi Input
class PredictionInput(BaseModel):
    features: List[float]

# --- DEFINISI 10 METRIK ---
# 1. Total Permintaan
REQUEST_COUNT = Counter('prediction_requests_total', 'Total jumlah permintaan prediksi')

# 2. Latensi Permintaan
REQUEST_LATENCY = Histogram('prediction_latency_seconds', 'Waktu pemrosesan prediksi')

# 3. Nilai Prediksi Terakhir
PREDICTION_GAUGE = Gauge('last_prediction_value', 'Nilai prediksi terakhir')

# 4. Distribusi Output Prediksi (Selamat/Meninggal)
PREDICTION_OUTPUT_COUNT = Counter('prediction_output_count', 'Distribusi kelas prediksi', ['class'])

# 5. Jumlah Fitur Input (Metrik dummy untuk data drift)
INPUT_FEATURE_SUM = Counter('input_feature_sum', 'Jumlah nilai fitur input')

# 6. Permintaan Tidak Valid (Kesalahan Validasi)
INVALID_REQUEST_COUNT = Counter('invalid_requests_total', 'Total jumlah permintaan tidak valid')

# 7. Penggunaan CPU Sistem
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'Persentase penggunaan CPU sistem saat ini')

# 8. Penggunaan Memori Sistem
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_bytes', 'Penggunaan memori sistem saat ini dalam byte')

# 9. Distribusi Fitur: Umur (Indeks 2 dalam fitur)
FEATURE_AGE_DIST = Histogram('feature_age_distribution', 'Distribusi fitur Umur')

# 10. Distribusi Fitur: Tarif (Indeks 5 dalam fitur)
FEATURE_FARE_DIST = Histogram('feature_fare_distribution', 'Distribusi fitur Tarif')


# Inisialisasi Model
model_service = inference_module.ModelInference()

def update_system_metrics():
    """Perbarui metrik sistem (CPU/Memori)"""
    SYSTEM_CPU_USAGE.set(psutil.cpu_percent())
    SYSTEM_MEMORY_USAGE.set(psutil.virtual_memory().used)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediksi Kelangsungan Hidup berdasarkan fitur Titanic.
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
              description: Daftar fitur yang diproses (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S)
    responses:
      200:
        description: Hasil prediksi
        schema:
          type: object
          properties:
            prediction:
              type: integer
              description: 0 (Meninggal) atau 1 (Selamat)
            status:
              type: string
      400:
        description: Kesalahan Validasi
      500:
        description: Kesalahan Server Internal
    """
    start_time = time.time()
    REQUEST_COUNT.inc()
    update_system_metrics()
    
    try:
        # Validasi input menggunakan Pydantic
        json_data = request.json
        if not json_data:
             INVALID_REQUEST_COUNT.inc()
             return jsonify({'error': 'No JSON data provided'}), 400
             
        input_data = PredictionInput(**json_data)
        
        data = input_data.features
        
        # Periksa apakah panjang fitur sesuai ekspektasi model (8 fitur)
        if len(data) != 8:
             # Jika ingin ketat, hapus komentar di bawah. Saat ini, kita izinkan jika model bisa menanganinya atau kita pad.
             # Tapi model dummy kita mengharapkan 8.
             # Mari pad atau potong untuk mencegah crash? 
             # Atau kembalikan error. Mengembalikan error lebih baik untuk metrik "Permintaan Tidak Valid".
             if len(data) < 8:
                 data = data + [0] * (8 - len(data))
             elif len(data) > 8:
                 data = data[:8]
        
        # Catat metrik fitur
        # Umur adalah indeks 2, Tarif adalah indeks 5
        if len(data) > 2:
            FEATURE_AGE_DIST.observe(data[2])
        if len(data) > 5:
            FEATURE_FARE_DIST.observe(data[5])
            
        INPUT_FEATURE_SUM.inc(sum(data))

        prediction = model_service.predict(data)
        
        # Rekam metrik prediksi
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

# Tambahkan middleware wsgi prometheus untuk merutekan permintaan /metrics
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

if __name__ == '__main__':
    print("Starting Prometheus Exporter on port 5000...")
    app.run(host='0.0.0.0', port=5001)
