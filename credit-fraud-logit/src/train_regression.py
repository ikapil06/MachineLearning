import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import argparse

PROC = Path('data/processed')
REPORTS = Path('reports'); REPORTS.mkdir(exist_ok=True)
MODELS = Path('models'); MODELS.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['linear','ridge','lasso'], default='linear')
args = parser.parse_args()

train = pd.read_csv(PROC / 'train.csv')
test = pd.read_csv(PROC / 'test.csv')
# Determine target column
if 'target_reg' in train.columns:
    target = 'target_reg'
elif 'target' in train.columns:
    target = 'target'
else:
    raise ValueError('No continuous target column found.')

X_train = train.drop(columns=[col for col in ['target', 'target_reg', 'target_class'] if col in train.columns])
y_train = train[target]
X_test = test.drop(columns=[col for col in ['target', 'target_reg', 'target_class'] if col in test.columns])
y_test = test[target]

num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
preprocess = ColumnTransformer([
    ('num', StandardScaler(with_mean=False), num_cols)
], remainder='drop')

if args.model == 'linear':
    model = LinearRegression()
elif args.model == 'ridge':
    model = Ridge(alpha=1.0)
else:
    model = Lasso(alpha=0.001)

pipeline = Pipeline([
    ('preprocess', preprocess),
    ('model', model)
])

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
with open(REPORTS / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

joblib.dump(pipeline, MODELS / f"{args.model}.pkl")
print(metrics)
