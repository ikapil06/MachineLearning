# Credit Fraud Detection - Comprehensive Analysis & Machine Learning

A complete end-to-end fraud detection project featuring comprehensive Exploratory Data Analysis (EDA) and multiple machine learning models implementation.

## 🎯 Project Overview

This project implements a robust credit fraud detection system using machine learning techniques. The analysis includes comprehensive EDA, feature engineering, multiple model comparison, and production-ready model deployment artifacts.

## 📊 Dataset

- **Size**: 1,000 transactions × 5 features
- **Target**: Binary classification (0 = No Fraud, 1 = Fraud)
- **Class Distribution**: 86% Non-fraud, 14% Fraud (Imbalanced dataset)
- **Features**:
  - `amount`: Transaction amount
  - `transaction_time`: Time of transaction
  - `location`: Location code (0-99)
  - `merchant_category`: Category of merchant (0-49)
  - `target`: Fraud indicator (0 = No fraud, 1 = Fraud)

## 🔍 Key Findings

### Data Quality
- ✅ No missing values
- ✅ No duplicate records
- ✅ No outliers detected
- ✅ Clean, ready-to-use dataset

### Fraud Patterns Discovered
- **Amount**: Fraud cases have 41.6% higher average amounts ($3,280 vs $2,317)
- **Transaction Time**: Fraud cases occur 14.7% later in time
- **Location**: Fraud cases slightly more concentrated in higher location codes
- **Class Imbalance**: 6.14:1 ratio (Non-fraud:Fraud) requiring special handling

### Feature Importance
1. **Amount** (0.229): Strong positive correlation with fraud
2. **Transaction Time** (0.087): Weak positive correlation
3. **Location** (0.052): Very weak positive correlation
4. **Merchant Category** (-0.035): Very weak negative correlation

## 🚀 Machine Learning Implementation

### Models Tested
- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **Neural Network (MLPClassifier)**

### Sampling Strategies
1. **Original Dataset** (Imbalanced)
2. **SMOTE** (Synthetic Minority Oversampling)
3. **Random Undersampling**

### Best Performing Model

🏆 **Logistic Regression with Undersampled Data**
- **F1-Score**: 0.455
- **Recall**: 0.821 (82.1% fraud detection rate)
- **Precision**: 0.315
- **ROC-AUC**: 0.822
- **Training Time**: <0.01 seconds

### Performance Metrics Summary

| Model | Dataset | F1-Score | Recall | Precision | ROC-AUC |
|-------|---------|----------|--------|-----------|---------|
| Logistic Regression | Undersampled | **0.455** | **0.821** | 0.315 | **0.822** |
| Logistic Regression | SMOTE | 0.454 | 0.786 | 0.319 | 0.821 |
| Random Forest | Undersampled | 0.385 | 0.750 | 0.259 | 0.791 |
| Neural Network | Undersampled | 0.404 | 0.821 | 0.267 | 0.808 |

## 📁 Project Structure

```
credittrial/
├── credit.ipynb                              # Main analysis notebook
├── README.md                                 # Project documentation
├── requirement.txt                           # Dependencies
├── best_fraud_model_logistic_regression.pkl  # Trained model
├── fraud_detection_scaler.pkl               # Feature scaler
└── fraud_detection_features.pkl             # Feature list
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Dependencies Installation
```bash
pip install -r requirement.txt
```

### Required Libraries
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly
- imbalanced-learn
- xgboost

## 🚀 Usage

### 1. Run the Complete Analysis
```bash
jupyter notebook credit.ipynb
```

### 2. Load Trained Model for Predictions
```python
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model and preprocessor
with open('best_fraud_model_logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)

with open('fraud_detection_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('fraud_detection_features.pkl', 'rb') as f:
    features = pickle.load(f)

# Make predictions on new data
# new_data = pd.DataFrame(...)  # Your new transaction data
# scaled_data = scaler.transform(new_data[features])
# predictions = model.predict(scaled_data)
# probabilities = model.predict_proba(scaled_data)[:, 1]
```

## 📈 Analysis Highlights

### Comprehensive EDA Includes:
- **18+ Visualizations**: Histograms, box plots, correlation heatmaps, scatter plots
- **Statistical Analysis**: Distribution characteristics, skewness, kurtosis
- **Target Analysis**: Class distribution and imbalance assessment
- **Feature Engineering**: Risk scores based on location and merchant categories
- **Data Quality**: Missing values, duplicates, and outlier detection

### Advanced Visualizations:
- Correlation matrices and heatmaps
- Feature distribution comparisons by fraud status
- Confusion matrices for model evaluation
- Performance metric comparisons across models
- Business impact analysis

## 🎯 Business Impact

### Model Performance:
- **Fraud Detection Rate**: 82.1% (only 5 out of 28 fraud cases missed)
- **False Alarm Rate**: 29.1% (manageable for business operations)
- **Processing Speed**: <0.01 seconds per prediction

### Business Value:
- Significantly reduces financial losses from undetected fraud
- Balances fraud detection with customer experience
- Provides interpretable results for business stakeholders
- Fast inference suitable for real-time applications

## 🔮 Future Improvements

1. **Feature Engineering**: Incorporate time-based features, velocity checks
2. **Ensemble Methods**: Combine multiple models for improved performance
3. **Real-time Processing**: Implement streaming fraud detection
4. **Model Monitoring**: Track performance drift and data quality
5. **Advanced Techniques**: Deep learning, anomaly detection methods

## 📊 Model Deployment Considerations

- **Threshold Tuning**: Adjust based on business risk tolerance
- **Regular Retraining**: Update with new fraud patterns
- **Performance Monitoring**: Track metrics in production
- **Scalability**: Consider distributed processing for large volumes
- **Explainability**: Maintain interpretable predictions for compliance

## 👥 Contributing

Feel free to fork this repository and submit pull requests for improvements.

## 📄 License

This project is open source and available under the MIT License.

## 📞 Contact

- **Author**: Kapil
- **Repository**: [MachineLearning/credittrial](https://github.com/ikapil06/MachineLearning/tree/main/credittrial)

---

*This project demonstrates end-to-end machine learning implementation for fraud detection, from data exploration to production-ready model deployment.*