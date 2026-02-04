# Instruksi Pengerjaan Proyek

## Kriteria 1: Eksperimen & Preprocessing
**Lokasi**: `Eksperimen_SML_Trae/`

1.  **Notebook**: `preprocessing/Eksperimen_Trae.ipynb`
    *   Buka di Jupyter Notebook/VS Code.
    *   Jalankan semua cell untuk melihat proses EDA dan Preprocessing manual.
2.  **Automasi**: `preprocessing/automate_Trae.py`
    *   Script ini mengonversi langkah di notebook menjadi fungsi Python.
    *   Cara run: `python Eksperimen_SML_Trae/preprocessing/automate_Trae.py`
    *   Output: `Eksperimen_SML_Trae/preprocessing/train_processed.csv`

## Kriteria 2: Membangun Model
**Lokasi**: `Membangun_model/`

1.  **Modelling Dasar**: `modelling.py`
    *   Melatih model Random Forest dengan autologging MLflow.
    *   Cara run: `cd Membangun_model && python modelling.py`
2.  **Modelling Tuning**: `modelling_tuning.py`
    *   Melatih dengan Hyperparameter Tuning + Manual Logging.
    *   Cara run: `cd Membangun_model && python modelling_tuning.py`
3.  **Melihat Hasil MLflow**:
    *   Jalankan `mlflow ui` di terminal.
    *   Buka `http://localhost:5000` di browser.

## Kriteria 3: Workflow CI
**Lokasi**: `Workflow-CI/` & `.github/workflows/`

1.  **MLProject**: Folder `Workflow-CI/MLProject` berisi definisi proyek MLflow.
2.  **GitHub Actions**:
    *   File `.github/workflows/preprocessing.yml`: Otomatisasi preprocessing.
    *   File `.github/workflows/training.yml`: Otomatisasi training model via MLflow Project.
    *   Actions akan berjalan otomatis saat Anda push ke GitHub.

## Kriteria 4: Monitoring & Logging
**Lokasi**: `Monitoring_dan_Logging/`

1.  **Inference & Exporter**:
    *   Jalankan service: `python Monitoring_dan_Logging/3.prometheus_exporter.py`
    *   Service akan berjalan di `http://localhost:5000`.
    *   Endpoint Prediksi: `POST http://localhost:5000/predict`
    *   Endpoint Metrics: `GET http://localhost:5000/metrics`
2.  **Prometheus**:
    *   Gunakan config `2.prometheus.yml`.
    *   Pastikan Prometheus terinstall dan jalankan dengan config tersebut.
3.  **Grafana**:
    *   Hubungkan ke Prometheus.
    *   Buat dashboard sesuai metrics yang muncul.
4.  **Bukti**:
    *   Simpan screenshot ke folder `4.bukti_monitoring_Prometheus`, `5.bukti_monitoring_Grafana`, dll.
