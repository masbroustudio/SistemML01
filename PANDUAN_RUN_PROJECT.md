# Panduan Menjalankan dan Menguji Proyek Sistem Machine Learning

Dokumen ini berisi langkah-langkah detail untuk menjalankan dan menguji setiap kriteria proyek sesuai dengan `PROJECT_BRIEF.md`.

## Persiapan Awal
Pastikan Anda berada di root directory proyek: `C:\Users\NITRO5\Documents\trae_projects\SistemML01`
Dan pastikan environment Python sudah aktif serta dependencies terinstall:
```bash
pip install -r Membangun_model/requirements.txt
```

---

## Kriteria 1: Eksperimen & Preprocessing
**Tujuan**: Melakukan eksperimen data dan otomatisasi preprocessing.

### 1. Menjalankan Notebook Eksperimen (Manual)
*   **File**: `Eksperimen_SML_Trae/preprocessing/Eksperimen_Trae.ipynb`
*   **Cara**: Buka file ini menggunakan VS Code atau Jupyter Notebook, lalu jalankan semua cell ("Run All").
*   **Verifikasi**: Pastikan tidak ada error pada cell dan proses EDA serta preprocessing berjalan.

### 2. Menjalankan Script Automasi
*   **File**: `Eksperimen_SML_Trae/preprocessing/automate_Trae.py`
*   **Deskripsi**: Script ini mengonversi langkah-langkah di notebook menjadi fungsi Python otomatis.
*   **Perintah**:
    ```powershell
    python Eksperimen_SML_Trae/preprocessing/automate_Trae.py
    ```
*   **Verifikasi**: Periksa apakah file output baru berhasil dibuat di:
    `Eksperimen_SML_Trae/preprocessing/train_processed.csv`

### 3. (Optional) Menjalankan Unit Testing
*   **File**: `tests/test_preprocessing.py`
*   **Tujuan**: Memastikan script preprocessing berjalan dengan benar (bebas error dan output valid).
*   **Perintah**:
    ```powershell
    python -m pytest
    ```
*   **Hasil**: Anda harus melihat status **PASSED** (hijau).

---

## Kriteria 2: Membangun Model Machine Learning
**Tujuan**: Melatih model dan melakukan tracking eksperimen menggunakan MLflow.

### 1. Melatih Model Dasar (Autolog)
*   **File**: `Membangun_model/modelling.py`
*   **Perintah**:
    ```powershell
    cd Membangun_model
    python modelling.py
    ```
*   **Hasil**: Model akan dilatih menggunakan `RandomForestClassifier` dan metrik akan dicatat otomatis oleh MLflow.

### 2. Melatih Model dengan Tuning (Hyperparameter Tuning) & DagsHub (Advanced)
*   **File**: `Membangun_model/modelling_tuning.py`
*   **Persiapan DagsHub**:
    1.  Pastikan Anda sudah install library dagshub: `pip install dagshub` (sudah ada di requirements).
    2.  Login ke DagsHub melalui terminal (gunakan full path jika perintah 'dagshub' tidak dikenali):
        ```powershell
        C:\Users\NITRO5\AppData\Roaming\Python\Python313\Scripts\dagshub.exe login
        ```
    3.  Ikuti instruksi di terminal (klik link autentikasi).
*   **Perintah**:
    ```powershell
    # Pastikan masih di dalam folder Membangun_model
    python modelling_tuning.py
    ```
*   **Hasil**: 
    *   Eksperimen akan dicatat secara otomatis ke repository DagsHub Anda: [https://dagshub.com/masbroumail/SistemML01](https://dagshub.com/masbroumail/SistemML01).
    *   Anda bisa melihat metrik dan artefak langsung di website DagsHub tanpa perlu menjalankan `mlflow ui` lokal.

### 3. Melihat Hasil di MLflow UI (Lokal)
*   **Perintah**:
    ```powershell
    python -m mlflow ui
    ```
    *(Gunakan `python -m mlflow` jika perintah `mlflow` langsung tidak dikenali)*
*   **Verifikasi**:
    1.  Buka browser ke [http://localhost:5000](http://localhost:5000).
    2.  Anda akan melihat eksperimen bernama **"Titanic_Basic_Model"** dan **"Titanic_Tuned_Model"**.
    3.  Klik salah satu run untuk melihat parameter, metrik, dan artefak (gambar confusion matrix).

---

## Kriteria 3: Membuat Workflow CI
**Tujuan**: Otomatisasi training menggunakan GitHub Actions dan MLflow Project.

### 1. Tes Lokal (MLflow Project)
*   Anda bisa mensimulasikan apa yang akan dijalankan oleh CI server di lokal.
*   **Perintah**:
    ```powershell
    cd ../Workflow-CI/MLProject
    python -m mlflow run . --env-manager=local
    ```
    *(Gunakan `python -m mlflow` agar lebih aman dari masalah PATH)*
*   **Verifikasi**: MLflow akan menjalankan entry point yang didefinisikan di file `MLProject` (yaitu menjalankan `modelling.py`).

### 2. GitHub Actions (CI Online)
*   Workflow file sudah disiapkan di:
    *   `.github/workflows/preprocessing.yml`
    *   `.github/workflows/training.yml`
*   **Cara**: Push seluruh folder proyek ini ke repository GitHub Anda.
*   **Verifikasi**: Buka tab "Actions" di repository GitHub Anda untuk melihat pipeline berjalan otomatis.

---

## Kriteria 4: Monitoring dan Logging
**Tujuan**: Melakukan serving model dan monitoring menggunakan Prometheus & Grafana.

### 1. Menjalankan Model Serving & Exporter
*   Saya telah membuat script yang menggabungkan endpoint prediksi (Flask) dengan Prometheus Exporter, serta dilengkapi dokumentasi API (Swagger) dan validasi input.
*   **Perintah**:
    ```powershell
    # Kembali ke root folder
    cd ../../
    python Monitoring_dan_Logging/3.prometheus_exporter.py
    ```
*   **Note**: Biarkan terminal ini tetap berjalan.

### 2. Dokumentasi API (Swagger UI)
*   Setelah server berjalan, akses dokumentasi interaktif di:
    [http://localhost:5001/apidocs](http://localhost:5001/apidocs)
*   Anda bisa mencoba endpoint `/predict` langsung dari halaman web ini tanpa perlu perintah cURL/PowerShell.

### 3. Menguji Endpoint Prediksi (Manual)
*   **Perintah**:
    ```powershell
    Invoke-RestMethod -Uri "http://localhost:5001/predict" -Method Post -ContentType "application/json" -Body '{"features": [3, 0, 22.0, 1, 0, 7.25, 1, 0]}'
    ```
    *Catatan: Pastikan jumlah fitur sesuai (8 fitur: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S)*
*   **Hasil**: JSON berisi hasil prediksi (misal: `{"prediction": 1, "status": "success"}`).

### 4. Menguji Endpoint Metrics
*   **Perintah**:
    ```powershell
    Invoke-RestMethod -Uri "http://localhost:5001/metrics" -Method Get
    ```
*   **Hasil**: Anda akan melihat daftar metrik format Prometheus (teks plain).

### 5. Setup Prometheus & Grafana (Manual)
Tahapan ini memerlukan instalasi aplikasi monitoring pihak ketiga. Ikuti langkah detail berikut:

#### A. Setup Prometheus (Pengumpul Metrik)
1.  **Download Prometheus**:
    *   Kunjungi [Halaman Download Prometheus](https://prometheus.io/download/).
    *   Pilih versi **Windows** (misal: `prometheus-2.53.0.windows-amd64.zip`).
    *   Download dan Extract file zip tersebut ke folder yang mudah diakses (misal: `C:\Prometheus`).

2.  **Konfigurasi**:
    *   Kita akan menggunakan file konfigurasi yang sudah ada di project ini: `Monitoring_dan_Logging/2.prometheus.yml`.
    *   Pastikan file tersebut menargetkan aplikasi Flask kita (`localhost:5001`).

3.  **Jalankan Prometheus**:
    *   Buka terminal baru (PowerShell/CMD).
    *   Arahkan ke folder hasil extract Prometheus tadi.
    *   Jalankan perintah berikut (sesuaikan path project Anda):
        ```powershell
        # Contoh jika file 2.prometheus.yml ada di Documents\trae_projects\SistemML01\Monitoring_dan_Logging
        .\prometheus.exe --config.file="C:\Users\NITRO5\Documents\trae_projects\SistemML01\Monitoring_dan_Logging\2.prometheus.yml"
        ```
    *   *Tips*: Pastikan tidak ada error di terminal.

4.  **Verifikasi**:
    *   Buka browser: [http://localhost:9090/targets](http://localhost:9090/targets).
    *   Pastikan status target `ml_model_monitoring` adalah **UP** (hijau). Jika `DOWN` (merah), pastikan script `3.prometheus_exporter.py` sedang berjalan.

#### B. Setup Grafana (Visualisasi)
1.  **Download Grafana**:
    *   Kunjungi [Halaman Download Grafana OSS](https://grafana.com/grafana/download?platform=windows).
    *   Pilih versi **Windows** (Standalone binary zip lebih mudah).
    *   Download dan Extract (misal: `C:\Grafana`).

2.  **Jalankan Grafana**:
    *   Masuk ke folder `bin` di dalam folder Grafana (misal: `C:\Grafana\bin`).
    *   Jalankan file `grafana-server.exe` (klik 2x atau via terminal).

3.  **Login**:
    *   Buka browser: [http://localhost:3000](http://localhost:3000).
    *   Login default: User=`admin`, Password=`admin`. (Anda akan diminta ganti password, boleh skip).

4.  **Hubungkan ke Prometheus**:
    *   Di menu kiri, pilih **Connections** -> **Data sources** -> **Add data source**.
    *   Pilih **Prometheus**.
    *   Di bagian **Connection** -> **Prometheus server URL**, isi: `http://localhost:9090`.
    *   Scroll ke bawah, klik **Save & test**. Pastikan muncul centang hijau "Successfully queried the Prometheus API".

5.  **Buat Dashboard Monitoring (Minimal 5 Metrik)**:
    *   Klik menu **Dashboards** -> **New** -> **New dashboard** -> **Add visualization**.
    *   Pilih data source **Prometheus** yang baru dibuat.
    *   Masukkan query metrik berikut satu per satu (ulangi langkah Add visualization untuk setiap panel):
        1.  **Total Prediksi**: `prediction_requests_total`
        2.  **Distribusi Kelas Prediksi**: `prediction_output_count`
        3.  **Latency (Histogram)**: `prediction_latency_seconds_bucket`
        4.  **Input Values (Fitur)**: `input_feature_sum` (opsional jika ada custom metric lain)
        5.  **Process CPU Seconds**: `process_cpu_seconds_total` (metrik bawaan Prometheus client)
    *   Klik **Apply** untuk setiap panel.
    *   Simpan dashboard (ikon disket di kanan atas).

#### C. Setup Alerting di Grafana (Minimal 1 Alert)
1.  Edit salah satu panel (misal: Total Prediksi).
2.  Masuk tab **Alert** (di bawah grafik saat edit panel).
3.  Klik **Create alert rule from this panel**.
4.  Set kondisi: Misal "When `prediction_requests_total` is above 10".
5.  Set folder dan group (bebas).
6.  Simpan rule. Alert ini akan memicu notifikasi (di UI Grafana) jika jumlah prediksi melebihi 10.

---
**Catatan**: Ambil screenshot dari setiap tahapan (MLflow UI, Grafana Dashboard, Prometheus Targets, Swagger UI, Unit Test Result) dan simpan ke dalam folder bukti yang sesuai di `Monitoring_dan_Logging/` untuk kelengkapan submission.
