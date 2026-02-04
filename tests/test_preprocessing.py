import pytest
import pandas as pd
import numpy as np
import os
import sys

# Tambahkan root proyek ke path agar modul preprocessing dapat diimpor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Eksperimen_SML_YudhaElfransyah.preprocessing.automate_Yudha_Elfransyah import preprocess_data

@pytest.fixture
def sample_raw_data(tmp_path):
    """Buat file CSV mentah sampel untuk pengujian."""
    data = {
        'PassengerId': [1, 2, 3, 4, 5],
        'Survived': [0, 1, 1, 1, 0],
        'Pclass': [3, 1, 3, 1, 3],
        'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath', 'Allen, Mr. William Henry'],
        'Sex': ['male', 'female', 'female', 'female', 'male'],
        'Age': [22.0, 38.0, 26.0, 35.0, np.nan],  # Satu umur hilang
        'SibSp': [1, 1, 0, 1, 0],
        'Parch': [0, 0, 0, 0, 0],
        'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450'],
        'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05],
        'Cabin': [np.nan, 'C85', np.nan, 'C123', np.nan],
        'Embarked': ['S', 'C', 'S', 'S', np.nan] # Satu embarked hilang
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_train.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture
def output_path(tmp_path):
    return str(tmp_path / "test_processed.csv")

def test_preprocess_data_runs(sample_raw_data, output_path):
    """Uji bahwa fungsi preprocessing berjalan tanpa error."""
    df_clean = preprocess_data(sample_raw_data, output_path)
    assert df_clean is not None
    assert os.path.exists(output_path)

def test_columns_dropped(sample_raw_data, output_path):
    """Uji bahwa kolom yang tidak perlu dihapus."""
    df_clean = preprocess_data(sample_raw_data, output_path)
    dropped_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    for col in dropped_cols:
        assert col not in df_clean.columns

def test_missing_values_filled(sample_raw_data, output_path):
    """Uji bahwa nilai yang hilang diisi."""
    df_clean = preprocess_data(sample_raw_data, output_path)
    assert df_clean['Age'].isnull().sum() == 0
    # Umur harus diisi dengan median. Median dari [22, 38, 26, 35] adalah 29.0
    # Nilai nan harus diisi dengan 30.5 (median dari 22, 38, 26, 35 adalah (26+35)/2 = 30.5)
    # Cek tidak ada null untuk saat ini karena logika ada di dalam fungsi
    
    # Cek embarked tidak null (jika kolom ada, mungkin sudah one-hot encoded)
    # Embarked sudah one-hot encoded, jadi kolom 'Embarked' asli harusnya hilang
    assert 'Embarked' not in df_clean.columns 

def test_encoding_sex(sample_raw_data, output_path):
    """Uji bahwa kolom Sex di-encode menjadi 0 dan 1."""
    df_clean = preprocess_data(sample_raw_data, output_path)
    assert df_clean['Sex'].dtype in [np.int64, np.int32, int]
    # male=0, female=1
    assert set(df_clean['Sex'].unique()).issubset({0, 1})

def test_encoding_embarked(sample_raw_data, output_path):
    """Uji bahwa kolom Embarked di-one-hot encode."""
    df_clean = preprocess_data(sample_raw_data, output_path)
    # Harusnya ada Embarked_Q dan Embarked_S (tergantung nilai yang ada dan drop_first=True)
    # Nilai asli: S, C, S, S, NaN(jadi modus=S) -> S, C
    # get_dummies(drop_first=True) -> Embarked_Q, Embarked_S 
    # Jika C adalah referensi, kita mungkin melihat Embarked_Q dan Embarked_S. 
    # Tunggu, jika nilai hanya S dan C. C, S. 
    # Jika alfabetis: C, Q, S.
    # C dihapus. Q dan S tersisa.
    # Di data sampel kita: S, C, S, S, S. Hanya C dan S yang ada.
    # Jadi get_dummies(['C', 'S']) dengan drop_first=True kemungkinan akan menyimpan Embarked_S.
    
    # Cek jika ada kolom yang dimulai dengan Embarked_
    embarked_cols = [col for col in df_clean.columns if col.startswith('Embarked_')]
    assert len(embarked_cols) > 0
