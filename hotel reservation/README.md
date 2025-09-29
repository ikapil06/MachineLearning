# Hotel Reservation Analysis and Prediction

This project contains machine learning models for analyzing hotel reservation data and predicting booking status. The project includes exploratory data analysis (EDA), data preprocessing, and predictive modeling using various machine learning techniques.

## Project Structure

```
hotel reservation/
├── README.md                           # Project documentation
├── Hotel Reservations.csv              # Main dataset
├── hotel_reservation_model.ipynb       # Comprehensive ML workflow notebook
├── hotelres.ipynb                      # EDA and logistic regression analysis
├── archive.zip                         # Additional data archive
└── venv/                               # Virtual environment (created locally)
```

## Dataset

The project uses the **Hotel Reservations Dataset** which contains information about hotel bookings including:

- Booking details (ID, dates, duration)
- Guest information (adults, children, weekend/weekday nights)
- Room and pricing information
- Special requests and meal plans
- Market segment and distribution channel
- **Target variable**: `booking_status` (prediction target)

## Features

### Data Analysis
- **Exploratory Data Analysis (EDA)** with comprehensive visualizations
- **Data preprocessing** including date parsing and categorical encoding
- **Statistical analysis** of booking patterns and trends
- **Correlation analysis** between different features

### Visualizations
- Distribution histograms for numeric features
- Count plots for categorical variables
- Pair plots for feature relationships
- Correlation heatmaps
- Box, violin, and strip plots for detailed distribution analysis

### Machine Learning Models
- **Logistic Regression** for binary classification
- **Random Forest Classifier** for enhanced prediction accuracy
- **Model evaluation** using multiple metrics
- **ROC curve analysis** for performance assessment

## Setup and Installation

### Prerequisites
- Python 3.7 or higher
- Virtual environment (recommended)

### Installation Steps

1. **Clone or download the project**
   ```bash
   cd "d:\python\MachineLearning\hotel reservation"
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment (Windows PowerShell)
   .\venv\Scripts\Activate.ps1
   ```

3. **Install required packages**
   ```bash
   pip install pandas numpy matplotlib scikit-learn kagglehub jupyter ipykernel seaborn
   ```

### Required Libraries

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning algorithms and tools
- **jupyter**: Notebook environment
- **kagglehub**: Dataset management (optional)

## Usage

### Running the Analysis

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the notebooks**
   - `hotelres.ipynb`: For quick EDA and logistic regression analysis
   - `hotel_reservation_model.ipynb`: For comprehensive ML workflow

3. **Run the cells sequentially** to execute the complete analysis

### Key Notebooks

#### `hotelres.ipynb`
- **Focus**: Exploratory Data Analysis and Logistic Regression
- **Features**:
  - Data loading and preprocessing
  - Comprehensive visualizations
  - Logistic regression model
  - Performance evaluation with confusion matrix and ROC curve

#### `hotel_reservation_model.ipynb`
- **Focus**: End-to-end machine learning workflow
- **Features**:
  - Advanced data cleaning and preprocessing
  - Multiple model implementations
  - Feature engineering
  - Model comparison and evaluation

## Model Performance

The project implements multiple evaluation metrics:

- **Accuracy Score**: Overall prediction accuracy
- **Confusion Matrix**: Detailed prediction breakdown
- **ROC Curve**: Performance visualization for binary classification
- **Precision, Recall, F1-Score**: Comprehensive performance metrics

## Data Preprocessing

The notebooks include comprehensive data preprocessing:

1. **Date Processing**: Combining date columns into datetime format
2. **Missing Value Handling**: Detection and treatment of missing data
3. **Categorical Encoding**: One-hot encoding for categorical variables
4. **Feature Selection**: Removal of identifier and non-predictive columns

## Visualizations

The project generates multiple types of visualizations:

- **Distribution Analysis**: Histograms with KDE for numeric features
- **Target Analysis**: Count plots for booking status distribution
- **Relationship Analysis**: Pair plots and correlation heatmaps
- **Comparative Analysis**: Box, violin, and strip plots by booking status

## Results and Insights

The analysis provides insights into:

- **Booking Patterns**: Seasonal and temporal trends
- **Price Analysis**: Relationship between pricing and booking status
- **Guest Behavior**: Impact of guest characteristics on bookings
- **Predictive Factors**: Most important features for prediction

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes. Please respect the dataset's original licensing terms.

## Contact

For questions or suggestions regarding this project, please create an issue in the repository.

## Acknowledgments

- Dataset source: Hotel Reservations Classification Dataset
- Built using scikit-learn, pandas, and matplotlib
- Visualization powered by seaborn and matplotlib

---

**Note**: Make sure to activate your virtual environment before running the notebooks to ensure all dependencies are properly loaded.