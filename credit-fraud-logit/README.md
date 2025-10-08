# Credit Fraud Logit

This project demonstrates a simple logistic regression using synthetic data.

## Data
Synthetic data is generated with features and target variables. Files are stored in `data/raw/data.csv`.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare data: `python src/data_prep.py`
3. Train model(s):
   - Regression: `python src/train_regression.py --model ridge`
   - Classification: `python src/train_logit.py --penalty l2` (if applicable)

## Repository Structure
- `data/raw/` – contains the raw synthetic data.
- `data/processed/` – contains train/test splits after running data_prep.
- `src/` – Python scripts for data prep and model training.
- `models/` – saved model artifacts.
- `reports/` – contains metrics and figures.

## Notes
These examples use synthetic data and simple models. Modify the scripts and data generation for real-world use cases.