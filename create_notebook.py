import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Cell 1: Import Libraries
text_1 = """\
# Eksperimen Preprocessing Data Titanic
Notebook ini berisi tahapan eksperimen mulai dari loading data, EDA, hingga preprocessing untuk menyiapkan data bagi model Machine Learning.
"""
code_1 = """\
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
"""

# Cell 2: Load Data
text_2 = "## 1. Load Data"
code_2 = """\
df = pd.read_csv('../titanic_raw/train.csv')
df.head()
"""

# Cell 3: EDA
text_3 = "## 2. Exploratory Data Analysis (EDA)"
code_3 = """\
df.info()
"""
code_4 = """\
df.describe()
"""
code_5 = """\
# Cek missing values
df.isnull().sum()
"""

# Cell 4: Preprocessing
text_4 = "## 3. Data Preprocessing"
code_6 = """\
# Drop kolom yang tidak diperlukan
df_clean = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df_clean.head()
"""

code_7 = """\
# Fill missing values
# Age: isi dengan median
df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())

# Embarked: isi dengan modus
mode_embarked = df_clean['Embarked'].mode()[0]
df_clean['Embarked'] = df_clean['Embarked'].fillna(mode_embarked)

df_clean.isnull().sum()
"""

code_8 = """\
# Encoding Categorical Variables
# Sex: Label Encoding
df_clean['Sex'] = df_clean['Sex'].map({'male': 0, 'female': 1})

# Embarked: One Hot Encoding
df_clean = pd.get_dummies(df_clean, columns=['Embarked'], drop_first=True)

df_clean.head()
"""

code_9 = """\
# Simpan hasil preprocessing
df_clean.to_csv('train_processed.csv', index=False)
print("Data processed saved to train_processed.csv")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(text_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_markdown_cell(text_3),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_markdown_cell(text_4),
    nbf.v4.new_code_cell(code_6),
    nbf.v4.new_code_cell(code_7),
    nbf.v4.new_code_cell(code_8),
    nbf.v4.new_code_cell(code_9)
]

os.makedirs('Eksperimen_SML_Trae/preprocessing', exist_ok=True)
with open('Eksperimen_SML_Trae/preprocessing/Eksperimen_Trae.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully.")
