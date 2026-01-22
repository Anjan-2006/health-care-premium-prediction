# Healthcare Premium Prediction System

A machine learning-based prediction system that estimates annual healthcare insurance premiums based on personal health and demographic information. The system uses age-specific models to provide accurate predictions for different age groups.

## 📋 Project Overview

This project implements a healthcare premium prediction system that leverages machine learning to forecast insurance premium amounts. It distinguishes between two age groups:
- **Young** (Age ≤ 25): Uses Linear Regression model
- **Rest** (Age > 25): Uses XGBoost Regressor model

The system provides an interactive web interface built with Streamlit for easy access and predictions.

## ✨ Features

- **Age-Based Model Selection**: Automatically selects the appropriate model based on patient age
- **Comprehensive Input Parameters**: Collects 12 key health and demographic features
- **Medical History Processing**: Intelligently processes medical history into normalized risk scores
- **Data Scaling**: Applies MinMax scaling with age-specific scalers for optimal model performance
- **Interactive Web Interface**: User-friendly Streamlit application for predictions
- **Feature Engineering**: Includes normalized risk score calculation based on medical conditions

## 📁 Project Structure

```
mlp-1/app/
├── main.py                          # Streamlit web application
├── main.ipynb                       # Training notebook for "Rest" model (age > 25)
├── main(young).ipynb                # Training notebook for "Young" model (age ≤ 25)
├── prediction_helper.py             # Helper functions for predictions and preprocessing
├── requirements.txt                 # Project dependencies
├── artifacts/
│   ├── model_rest.joblib           # XGBoost model for age > 25
│   ├── model_young.joblib          # Linear Regression model for age ≤ 25
│   ├── scaler_rest.joblib          # MinMax scaler for age > 25
│   └── scaler_young.joblib         # MinMax scaler for age ≤ 25
└── README.md                        # Project documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or navigate to the project directory**
   ```bash
   cd /app
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import streamlit, pandas, sklearn, xgboost; print('All dependencies installed successfully!')"
   ```

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.3.3 | Data manipulation and analysis |
| numpy | 2.4.0 | Numerical computations |
| joblib | 1.5.3 | Model and scaler serialization |
| scikit-learn | 1.8.0 | Machine learning algorithms |
| xgboost | 3.1.3 | Gradient boosting models |
| streamlit | 1.53.0 | Web application framework |

## 🚀 Running the Application

### Start the Streamlit App
```bash
streamlit run main.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Web Interface

1. **Age**: Enter patient age (18-100 years)
2. **Number of Dependants**: Specify number of dependents (0-20)
3. **Genetical Risk Score**: Enter genetic risk score (0-5)
4. **Income (in Lakhs)**: Annual income in lakhs of rupees
5. **Gender**: Select Male or Female
6. **Marital Status**: Choose Unmarried or Married
7. **Region**: Select from Northeast, Northwest, Southeast, Southwest
8. **BMI Category**: Choose from Underweight, Normal, Overweight, Obesity
9. **Smoking Status**: Select Regular, Occasional, or No Smoking
10. **Employment Status**: Choose Salaried, Self-Employed, or Freelancer
11. **Insurance Plan**: Select Bronze, Silver, or Gold
12. **Medical History**: Select from predefined medical conditions (including combinations with & operator)

Click **Predict** to get the estimated annual premium amount.

## 📊 Data Processing & Model Training

### Data Cleaning
- Removed null values and duplicates
- Converted negative dependency values to positive
- Filtered outliers (age > 100, income extreme values)
- Applied IQR method for outlier detection

### Feature Engineering
- **Medical History Processing**: Converts medical conditions into normalized risk scores
  - Diabetes: 6 points
  - Heart Disease: 8 points
  - High Blood Pressure: 6 points
  - Thyroid: 5 points
  - No Disease: 0 points
- **Categorical Encoding**: One-hot encoding for nominal features
- **Scaling**: MinMax scaling applied to numerical features

### Model Details

#### Young Model (Age ≤ 25)
- **Algorithm**: Linear Regression
- **Features**: 18 engineered features
- **Performance**: Good generalization for younger demographic
- **VIF Check**: Multi-collinearity assessed and `income_level` dropped

#### Rest Model (Age > 25)
- **Algorithm**: XGBoost Regressor
- **Features**: 18 engineered features
- **Hyperparameter Tuning**: RandomizedSearchCV applied
- **Best Parameters**: Optimized n_estimators, learning_rate, and max_depth

### Model Performance Metrics
- **Error Analysis**: Residual percentage distribution analyzed
- **Extreme Error Identification**: Cases with >50% error identified and analyzed

## 🔄 Input Preprocessing Pipeline

1. **Medical History Normalization**: Converts medical conditions to normalized risk scores (0-1)
2. **Categorical Encoding**: One-hot encoding for categorical variables
3. **Feature Mapping**: Insurance plan and income level mapped to numerical values
4. **Age-Based Scaling**: Applies age-appropriate MinMax scaler
5. **Feature Selection**: Ensures only required 18 features are passed to model

## 📈 Model Prediction Flow

```
User Input → Preprocessing → Feature Engineering → Age-Based Scaler Selection
→ Model Selection (Young/Rest) → Prediction → Premium Amount Output
```

## 🔧 Prediction Helper Functions

### Key Functions in `prediction_helper.py`

- **`calculate_normalized_risk()`**: Converts medical history to normalized risk score
- **`preprocessing_input()`**: Transforms user input to model-compatible format
- **`handle_scaling()`**: Applies age-appropriate scaling
- **`predict()`**: Main prediction function that orchestrates the workflow

## 💡 Usage Example

```python
from prediction_helper import predict

input_dict = {
    "age": 35,
    "number_of_dependants": 2,
    "income_lakhs": 10,
    "genetical_risk": 2,
    "gender": "Male",
    "marital_status": "Married",
    "region": "Southeast",
    "bmi_category": "Normal",
    "smoking_status": "No Smoking",
    "employment_status": "Salaried",
    "insurance_plan": "Silver",
    "medical_history": "No Disease"
}

prediction = predict(input_dict)
print(f"Predicted Premium: ₹{prediction}")
```

## 🔍 Model Validation

- **Train-Test Split**: 70-30 split with random_state=42
- **Cross-Validation**: 3-fold CV used in RandomizedSearchCV
- **Error Analysis**: Residual percentage distribution analyzed to identify systematic biases

## ⚠️ Important Notes

- Models are optimized for Indian insurance market (amounts in Lakhs)
- Young model (≤25) uses simpler Linear Regression for interpretability
- Rest model (>25) uses XGBoost for better non-linear relationship capture
- Age threshold of 25 selected based on demographic analysis

## 📝 Training Notebooks

### main(young).ipynb
Comprehensive training pipeline for age ≤ 25 demographic:
- Data exploration and visualization
- Statistical analysis
- Feature engineering
- Model training and comparison
- Error analysis and insights

### main.ipynb
Comprehensive training pipeline for age > 25 demographic:
- Data exploration with extended analysis
- Hyperparameter tuning with RandomizedSearchCV
- Multiple model comparison (Linear, Ridge, XGBoost)
- Feature importance analysis
- Error analysis with outlier detection

## 🎯 Next Steps & Improvements

- Model retraining with new data
- A/B testing with alternative models
- Feature importance analysis for business insights
- API deployment for production use
- Enhanced error handling and validation

## 📄 License

This project is provided as-is for healthcare analytics and prediction purposes.

## 📧 Support

For issues or questions regarding the project, please refer to the training notebooks for detailed methodology and analysis.

---

**Version**: 1.0  
**Last Updated**: January 2026
