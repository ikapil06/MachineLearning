import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW = Path('data/raw')
PROC = Path('data/processed')
PROC.mkdir(parents=True, exist_ok=True)

# Load the raw dataset
df = pd.read_csv(RAW / 'data.csv')

"""
Split the dataset into train and test sets. If the dataset contains a binary classification
target (either named ``target_class`` or a column named ``target`` with only two unique
values), perform a stratified split to ensure both classes are represented in the train
and test splits. Otherwise, perform a standard random split.
"""
# Determine if we should stratify based on classification target
stratify_col = None
if 'target_class' in df.columns:
    stratify_col = df['target_class']
elif 'target' in df.columns and df['target'].nunique() <= 2:
    stratify_col = df['target']

if stratify_col is not None:
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=stratify_col)
else:
    train, test = train_test_split(df, test_size=0.2, random_state=42)

train.to_csv(PROC / 'train.csv', index=False)
test.to_csv(PROC / 'test.csv', index=False)
print('Data split into train and test sets.')
