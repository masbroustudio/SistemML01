# To-Do List Proyek Sistem Machine Learning

## Persiapan Awal
- [ ] Konfirmasi nama siswa untuk penamaan file/folder (Default dari brief: `YudhaElfransyah`)
- [ ] Pilih dan unduh dataset yang akan digunakan
- [ ] Siapkan environment Python 3.12.7
- [ ] Instalasi dependencies awal (`mlflow==2.19.0`, `pandas`, `scikit-learn`, dll)

## Kriteria 1: Eksperimen & Preprocessing
- [ ] Buat struktur folder `Eksperimen_SML_[Nama-siswa]`
    - [ ] Folder `[namadataset]_raw`
    - [ ] Folder `preprocessing`
- [ ] Buat Notebook Eksperimen: `preprocessing/Eksperimen_[Nama-siswa].ipynb`
    - [ ] Load Data
    - [ ] Exploratory Data Analysis (EDA)
    - [ ] Data Preprocessing Manual
- [ ] Buat Script Automasi: `preprocessing/automate_[Nama-siswa].py`
    - [ ] Fungsi preprocessing otomatis
    - [ ] Output: Data siap latih
- [ ] Setup GitHub Repository & Local
- [ ] Buat GitHub Actions Workflow untuk preprocessing otomatis

## Kriteria 2: Membangun Model Machine Learning
- [ ] Buat struktur folder `Membangun_model`
- [ ] Implementasi `modelling.py` (Basic)
    - [ ] Train model (Scikit-Learn)
    - [ ] MLflow Tracking (Local)
    - [ ] MLflow Autolog
- [ ] Implementasi `modelling_tuning.py` (Skilled/Advanced)
    - [ ] Hyperparameter Tuning
    - [ ] Manual Logging (Metrics sama dengan autolog)
- [ ] Integrasi DagsHub (Advanced)
    - [ ] Setup DagsHub Repository
    - [ ] Config MLflow tracking URI ke DagsHub
    - [ ] Manual logging + 2 artefak tambahan
- [ ] Dokumentasi Bukti
    - [ ] `screenshoot_dashboard.jpg`
    - [ ] `screenshoot_artifak.jpg`
    - [ ] `requirements.txt`
    - [ ] `DagsHub.txt`

## Kriteria 3: Workflow CI
- [ ] Buat struktur folder `Workflow-CI`
- [ ] Siapkan folder `MLProject`
    - [ ] `modelling.py` (disesuaikan)
    - [ ] `conda.yaml`
    - [ ] File `MLProject` definition
- [ ] Buat Workflow CI (GitHub Actions)
    - [ ] Trigger re-training model
    - [ ] Simpan artefak ke Repo/Drive
- [ ] Docker Integration (Advanced)
    - [ ] `mlflow build-docker`
    - [ ] Push ke Docker Hub

## Kriteria 4: Monitoring dan Logging
- [ ] Serving Model
    - [ ] `inference.py` atau `mlflow model serve`
- [ ] Setup Prometheus
    - [ ] `prometheus.yml`
    - [ ] `prometheus_exporter.py`
    - [ ] Monitoring minimal 3 metriks
- [ ] Setup Grafana
    - [ ] Koneksi ke Prometheus
    - [ ] Dashboard Monitoring (Min 5-10 metriks)
    - [ ] Alerting (1-3 rules)
- [ ] Dokumentasi Bukti
    - [ ] Screenshot serving, monitoring, dan alerting sesuai struktur folder
