import pandas as pd
import os

def preprocess_data(input_path, output_path):
    """
    Melakukan preprocessing data Titanic secara otomatis.
    Args:
        input_path (str): Path ke file data raw (csv).
        output_path (str): Path untuk menyimpan file data hasil preprocessing (csv).
    Returns:
        pd.DataFrame: Dataframe yang sudah diproses.
    """
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File {input_path} not found.")
        return None

    # 1. Drop columns
    print("Dropping unnecessary columns...")
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # 2. Fill missing values
    print("Filling missing values...")
    # Age: median
    if 'Age' in df_clean.columns:
        df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
    
    # Embarked: mode
    if 'Embarked' in df_clean.columns:
        mode_embarked = df_clean['Embarked'].mode()[0]
        df_clean['Embarked'] = df_clean['Embarked'].fillna(mode_embarked)

    # 3. Encoding
    print("Encoding categorical variables...")
    # Sex
    if 'Sex' in df_clean.columns:
        df_clean['Sex'] = df_clean['Sex'].map({'male': 0, 'female': 1})
    
    # Embarked
    if 'Embarked' in df_clean.columns:
        df_clean = pd.get_dummies(df_clean, columns=['Embarked'], drop_first=True)

    # Save
    print(f"Saving processed data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print("Preprocessing completed successfully.")
    
    return df_clean

if __name__ == "__main__":
    # Define paths relative to this script execution or absolute
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'titanic_raw', 'train.csv')
    output_file = os.path.join(base_dir, 'preprocessing', 'train_processed.csv')
    
    preprocess_data(input_file, output_file)
