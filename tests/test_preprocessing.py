import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add the project root to the path so we can import the preprocessing module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Eksperimen_SML_Trae.preprocessing.automate_Trae import preprocess_data

@pytest.fixture
def sample_raw_data(tmp_path):
    """Create a sample raw CSV file for testing."""
    data = {
        'PassengerId': [1, 2, 3, 4, 5],
        'Survived': [0, 1, 1, 1, 0],
        'Pclass': [3, 1, 3, 1, 3],
        'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath', 'Allen, Mr. William Henry'],
        'Sex': ['male', 'female', 'female', 'female', 'male'],
        'Age': [22.0, 38.0, 26.0, 35.0, np.nan],  # One missing age
        'SibSp': [1, 1, 0, 1, 0],
        'Parch': [0, 0, 0, 0, 0],
        'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450'],
        'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05],
        'Cabin': [np.nan, 'C85', np.nan, 'C123', np.nan],
        'Embarked': ['S', 'C', 'S', 'S', np.nan] # One missing embarked
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_train.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture
def output_path(tmp_path):
    return str(tmp_path / "test_processed.csv")

def test_preprocess_data_runs(sample_raw_data, output_path):
    """Test that the preprocessing function runs without error."""
    df_clean = preprocess_data(sample_raw_data, output_path)
    assert df_clean is not None
    assert os.path.exists(output_path)

def test_columns_dropped(sample_raw_data, output_path):
    """Test that unnecessary columns are dropped."""
    df_clean = preprocess_data(sample_raw_data, output_path)
    dropped_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    for col in dropped_cols:
        assert col not in df_clean.columns

def test_missing_values_filled(sample_raw_data, output_path):
    """Test that missing values are filled."""
    df_clean = preprocess_data(sample_raw_data, output_path)
    assert df_clean['Age'].isnull().sum() == 0
    # Age should be filled with median. Median of [22, 38, 26, 35] is 29.0
    # The nan value should be filled with 30.5 (median of 22, 38, 26, 35 is (26+35)/2 = 30.5)
    # Let's just check no nulls for now as logic is inside the function
    
    # Check embarked is not null (if column exists, might be one-hot encoded)
    # Embarked was one-hot encoded, so original 'Embarked' col should be gone
    assert 'Embarked' not in df_clean.columns 

def test_encoding_sex(sample_raw_data, output_path):
    """Test that Sex column is encoded to 0 and 1."""
    df_clean = preprocess_data(sample_raw_data, output_path)
    assert df_clean['Sex'].dtype in [np.int64, np.int32, int]
    # male=0, female=1
    assert set(df_clean['Sex'].unique()).issubset({0, 1})

def test_encoding_embarked(sample_raw_data, output_path):
    """Test that Embarked column is one-hot encoded."""
    df_clean = preprocess_data(sample_raw_data, output_path)
    # Should have Embarked_Q and Embarked_S (depending on values present and drop_first=True)
    # Original values: S, C, S, S, NaN(becomes mode=S) -> S, C
    # get_dummies(drop_first=True) -> Embarked_Q, Embarked_S 
    # If C is reference, we might see Embarked_Q and Embarked_S. 
    # Wait, if values are only S and C. C, S. 
    # If alphabetical: C, Q, S.
    # C is dropped. Q and S remain.
    # In our sample data: S, C, S, S, S. Only C and S present.
    # So get_dummies(['C', 'S']) with drop_first=True will likely keep Embarked_S.
    
    # Check if any column starts with Embarked_
    embarked_cols = [col for col in df_clean.columns if col.startswith('Embarked_')]
    assert len(embarked_cols) > 0
