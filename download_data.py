import pandas as pd
import os

# Buat direktori jika belum ada
os.makedirs('Eksperimen_SML_YudhaElfransyah/titanic_raw', exist_ok=True)

# Unduh dataset Titanic
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
try:
    df = pd.read_csv(url)
    df.to_csv('Eksperimen_SML_YudhaElfransyah/titanic_raw/train.csv', index=False)
    print("Dataset downloaded successfully to Eksperimen_SML_YudhaElfransyah/titanic_raw/train.csv")
    print(df.head())
except Exception as e:
    print(f"Error downloading dataset: {e}")
