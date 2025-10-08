import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import argparse

PROC = Path('data/processed')
REPORTS = Path('reports'); REPORTS.mkdir(exist_ok=True)
MODELS = Path('models'); MODELS.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--penalty', choices=['l1','l2','none'], default='l2')
parser.add_argument('--class_weight', choices=['balanced','none'], default='balanced')
args = parser.parse_args()

train = pd.read_csv(PROC / 'train.csv')
test = pd.read_csv(PROC / 'test.csv')

# Determine target column
if 'target_class' in train.columns:
    target = 'target_class'
elif 'target' in train.columns:
    target = 'target'
else:
    raise ValueError('No classification target column found.')

X_train = train.drop(columns=[col for col in ['target', 'target_reg', 'target_class'] if col in train.columns])
y_train = train[target]
X_test = test.drop(columns=[col for col in ['target', 'target_reg', 'target_class'] if col in test.columns])
y_test = test[target]

num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
preprocess = ColumnTransformer([
    ('num', StandardScaler(with_mean=False), num_cols)
], remainder='drop')

class_weight = 'balanced' if args.class_weight == 'balanced' else None

model = LogisticRegression(penalty='none' if args.penalty=='none' else args.penalty,
                           solver='saga', max_iter=1000, class_weight=class_weight)

pipeline = Pipeline([
    ('preprocess', preprocess),
    ('model', model)
])

pipeline.fit(X_train, y_train)

proba = pipeline.predict_proba(X_test)[:, 1]
preds = (proba >= 0.5).astype(int)

metrics = {
    'ROC_AUC': roc_auc_score(y_test, proba),
    'PR_AUC': average_precision_score(y_test, proba),
    'F1': f1_score(y_test, preds)
}

with open(REPORTS / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

joblib.dump(pipeline, MODELS / 'logit.pkl')
print(metrics)
