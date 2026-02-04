# Membangun Sistem Machine Learning

Membangun sebuah sistem machine learning yang andal dan siap masuk tahap produksi, mencakup seluruh tahapan mulai dari pengumpulan data, pelatihan model, penelusuran metadata, hingga deployment dan monitoring yang aktif. 

## Kriteria 1: Melakukan Eksperimen terhadap Dataset Pelatihan


-  wajib menggunakan Template Eksperimen MSML (https://colab.research.google.com/drive/1vSTQWWgGqPGBGHvv8lbeGdoa5N92D_UC?usp=sharing) sebagai panduan awal sebelum membuat file untuk melakukan otomatisasi data preprocessing.
- Pastikan template tersebut diikuti dengan benar untuk memastikan proses berjalan sesuai standar yang ditetapkan. 
- Setelah melakukan eksplorasi, Anda telah memiliki panduan utama untuk membuat file yang dapat melakukan preprocessing data secara otomatis.
- Selanjutnya, silakan konversi langkah-langkah yang ada pada notebook eksperimen untuk membuat file otomatisasi tersebut.

Harus membuat sebuah repository (GitHub dan lokal) dengan struktur seperti ini:
Eksperimen_SML_Nama-siswa
├── .workflow (jika menerapkan advance)
├── namadataset_raw (bisa berupa file atau folder)
├── preprocessing
    └── Eksperimen_Nama-siswa.ipynb
    └── automate_Nama-siswa.py (jika menerapkan skilled)
    └── namadataset_preprocessing (bisa berupa file atau folder)

Anda disarankan menggunakan environment berikut untuk menunjang submission:

Python 3.12.7

mlflow==2.19.0

Berikut adalah hal yang harus dilengkapi untuk kriteria 1:
Melakukan tahapan experimentation secara manual.

Melakukan data loading pada notebook.

Melakukan EDA pada notebook.

Melakukan preprocessing pada notebook.

Membuat sebuah file automate_Nama-siswa.py yang berisikan fungsi untuk melakukan preprocessing secara otomatis sehingga mengembalikan data yang siap dilatih.

Pada tahap ini Anda harus melakukan konversi dari proses eksperimen sebelumnya, sehingga tahapannya harus sama tetapi memiliki struktur yang berbeda.

Membuat sebuah workflow pada GitHub Actions agar dapat melakukan preprocessing setiap kali trigger terpantik.

Anda harus membuat sebuah repository dengan nama Eksperimen_SML_YudhaElfransyah  berisi seluruh file yang sama dengan rekomendasi struktur folder pada kriteria 1.

Pastikan Actions yang dibuat mengembalikan sebuah dataset terbaru yang sudah diproses sedemikian rupa.

## Kriteria 2: Membangun Model Machine Learning

- Setelah selesai melalui tahapan preprocessing, Harus melatih model menggunakan dataset yang sudah siap digunakan (bukan raw). Nantinya Anda harus membuat sebuah folder yang berisikan file modelling.py beserta dependencies nya dengan struktur seperti berikut:
Membangun_model
├── modelling.py
├── modelling_tuning.py (jika menerapkan skilled/advanced)
├── namadataset_preprocessing (bisa berupa file atau folder)
├── screenshoot_dashboard.jpg
├── screenshoot_artifak.jpg
├── requirements.txt
├── DagsHub.txt (berisikan tautan DagsHub jika menerapkan advanced)

- Sebagai informasi, tahapan ini dapat Anda jalankan pada lokal environment sebagai jembatan penghubung ke kriteria tiga.


Berikut adalah hal yang harus dilengkapi untuk kriteria 2:

Melatih model machine learning (Scikit-Learn) menggunakan MLflow Tracking UI yang disimpan secara lokal tanpa menggunakan hyperparameter tuning.

Menggunakan autolog dari MLflow pada file modelling.py.

Melatih model machine learning/deep learning menggunakan MLflow Tracking UI yang disimpan secara lokal dengan menerapkan hyperparameter tuning.

Alih-alih menggunakan autolog, Anda diharapkan menggunakan manual logging dengan metriks yang sama dengan autolog.

Pastikan kamu melakukan checklist ini pada file modelling_tuning (bukan pada modelling.py)

Melatih model machine learning/deep learning menggunakan MLflow Tracking UI yang disimpan secara online dengan DagsHub.

Alih-alih menggunakan autolog, siswa diharapkan menggunakan manual logging dengan metriks yang tidak hanya tercover pada autolog (autolog + minimal 2 artefak tambahan).


## Kriteria 3: Membuat Workflow CI

- Setelah membuat dan memastikan file modelling.py berjalan dengan baik, selanjutnya harus membuat workflow CI menggunakan MLflow Project agar dapat melakukan re-training model secara otomatis ketika trigger dipantik. 

- buat sebuah project repository baru di GitHub dengan struktur seperti berikut ini:
Workflow-CI
├── .workflow
├── MLProject (folder)
    └── modelling.py
    └── conda.yaml
    └── MLProject
    └── namadataset_preprocessing (bisa berupa file atau folder)
    └── Tautan ke Docker Hub
    └── (file tambahan jika diperlukan)

Anda dapat menggunakan file modelling.py, conda.yaml serta dataset yang sudah siap dilatih dari hasil eksperimen sebelumnya. Pada tahap ini, Anda hanya perlu membuat struktur yang diminta beserta file MLProjectnya saja. Namun, tidak menutup kemungkinan Anda harus menyesuaikan file modelling.py ketika masuk ke tahap ini.

Berikut adalah hal yang harus dilengkapi untuk kriteria 3:

Membuat folder MLProject.

Membuat Worflow CI yang dapat membuat model machine learning ketika trigger terpantik.

Membuat workflow CI dan menyimpan artefak ke suatu repositori (GitHub yang sama atau Google Drive).

Membuat workflow CI dan menyimpan artefak ke suatu repositori (GitHub yang sama atau Google Drive) serta membuat Docker Images ke Docker Hub menggunakan fungsi mlflow build-docker.


## Kriteria 4: Membuat Sistem Monitoring dan Logging


Nantinya, Anda hanya akan mengumpulkan tangkapan layar mengenai skill yang diampu dengan struktur seperti berikut ini:
Monitoring dan Logging
├── 1.bukti_serving
├── 2.prometheus.yml
├── 3.prometheus_exporter.py
├── 4.bukti monitoring Prometheus (folder)
    └── 1.monitoring_<metriks>
    └── 2.monitoring_<metriks>
    └── dst (sesuaikan dengan poin yang diraih)
├── 5.bukti monitoring Grafana (folder)
    └── 1.monitoring_<metriks>
    └── 2.monitoring_<metriks>
    └── dst (sesuaikan dengan poin yang diraih)
├── 6.bukti alerting Grafana (folder)
    └── 1.rules_<metriks>
    └── 2.notifikasi_<metriks>
    └── 3.rules_<metriks>
    └── 4.notifikasi_<metriks>
    └── dst (sesuaikan dengan poin yang diraih)
├── 7.inference.py
├── folder/file tambahan



Berikut adalah hal yang harus dilengkapi untuk kriteria 4:

Melakukan serving model baik itu melalui artefak yang sudah dibuat atau pull Images (jika menerapkan kriteria CI untuk melakukan push ke Docker Hub)

Bisa melalui mlflow model serve, mlflow deployments, atau pull images jika memenuhi kriteria 3 advanced.

Melakukan monitoring menggunakan Prometheus minimal dengan tiga metriks yang berbeda.

Melakukan monitoring menggunakan Grafana dengan metriks yang sama dengan Prometheus.

Melakukan monitoring menggunakan Grafana dengan minimal 5 metriks yang berbeda.

Membuat satu alerting menggunakan Grafana.

Melakukan monitoring menggunakan Grafana dengan minimal 10 metriks yang berbeda.

Membuat tiga alerting menggunakan Grafana.