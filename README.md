# Eksperimen Stroke Prediction - Sundari Oktamiyunda

## 📋 Project Overview

This repository contains a complete machine learning experimentation pipeline for stroke prediction using healthcare data. The project demonstrates both manual experimentation and automated preprocessing capabilities.

## 🎯 Project Objectives

- **Primary Goal**: Predict stroke occurrence based on patient health indicators
- **Learning Focus**: Complete ML experimentation pipeline from raw data to model-ready dataset
- **Technical Skills**: Data preprocessing, automation, and software engineering best practices

## 📊 Dataset Information

**Source**: [Stroke Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

**Characteristics**:
- **Size**: 5,110 patients × 12 features
- **Type**: Binary classification problem
- **Target**: Stroke occurrence (0 = No Stroke, 1 = Stroke)
- **Challenge**: Highly imbalanced dataset (95.1% vs 4.9%)

**Features**:
- Patient demographics (age, gender, marriage status)
- Health conditions (hypertension, heart disease, BMI)
- Lifestyle factors (work type, residence, smoking status)
- Clinical measurements (glucose level)

## 📁 Repository Structure

```
Eksperimen_SML_Sundari-Oktamiyunda/
├── stroke_data_raw/
│   └── healthcare-dataset-stroke-data.csv          # Raw dataset
└── preprocessing/
    ├── Eksperimen_Sundari-Oktamiyunda.ipynb        # Manual experimentation
    ├── automate_Sundari-Oktamiyunda.py             # Automated preprocessing
    ├── test_automation.py                          # Unit tests
    ├── requirements.txt                            # Dependencies
    └── stroke_data_preprocessing/                  # Output folder
        ├── train_data_processed.csv                # Processed training data
        ├── test_data_processed.csv                 # Processed test data
        └── feature_info.json                       # Feature metadata
```

## 🚀 Getting Started

### Prerequisites

```bash
# Install required packages
pip install -r preprocessing/requirements.txt
```

### Manual Experimentation

1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook preprocessing/Eksperimen_Sundari-Oktamiyunda.ipynb
   ```

2. **Run all cells** to see complete experimentation process:
   - Data loading and exploration
   - Comprehensive EDA
   - Step-by-step preprocessing

### Automated Preprocessing

1. **Navigate to preprocessing folder**:
   ```bash
   cd preprocessing/
   ```

2. **Run automation script**:
   ```bash
   python automate_Sundari-Oktamiyunda.py
   ```

3. **Run unit tests**:
   ```bash
   python test_automation.py
   ```

## 🔧 Preprocessing Pipeline

### Automated Steps

1. **Data Loading**: Load raw CSV with validation
2. **Missing Value Handling**: Median imputation for BMI
3. **Feature Engineering**: One-hot encoding for categorical variables
4. **Feature Scaling**: StandardScaler for numerical features
5. **Class Balancing**: SMOTE oversampling for minority class
6. **Data Splitting**: Stratified train-test split (80:20)
7. **Output Generation**: Save processed datasets and metadata

### Key Transformations

- **Missing Values**: 201 BMI 'N/A' values → Median imputation
- **Categorical Encoding**: 5 categorical features → 25 encoded features  
- **Class Balance**: 95.1% vs 4.9% → 50% vs 50% (training only)
- **Feature Scaling**: Age, glucose, BMI normalized to standard distribution

## 📈 Results Summary

### Dataset Transformation
- **Input**: 5,110 × 11 features (imbalanced)
- **Output**: 
  - Training: 7,718 × 25 features (balanced)
  - Testing: 1,022 × 25 features (original distribution)

### Key Insights from EDA
- **Age** is the strongest predictor of stroke
- **Hypertension** and **heart disease** significantly increase risk
- **Gender differences** exist in stroke rates
- **Glucose levels** show distinct patterns between groups

## 🧪 Testing

Comprehensive unit tests cover:
- Data loading and validation
- Missing value handling
- Feature encoding and scaling
- Class balancing effectiveness
- Pipeline integration
- Error handling and edge cases

Run tests: `python test_automation.py`

## 📚 Technical Implementation

### Object-Oriented Design
- **StrokeDataPreprocessor** class with modular methods
- **Error handling** for robust production use
- **Configurable parameters** for reproducibility

### Software Engineering Best Practices
- **Unit testing** with comprehensive coverage
- **Documentation** with clear docstrings
- **Modular functions** for reusability
- **Version control** ready structure

## 🎯 Achievement Level

**Skilled Level (3 points)**:
- ✅ Manual experimentation in Jupyter notebook
- ✅ Automated preprocessing pipeline
- ✅ Function-based architecture
- ✅ Unit testing implementation
- ✅ Same preprocessing logic in both formats

## 🔄 Usage Examples

### Quick Start
```python
from automate_Sundari_Oktamiyunda import StrokeDataPreprocessor

# Initialize preprocessor
preprocessor = StrokeDataPreprocessor(random_state=42)

# Run complete pipeline
result = preprocessor.preprocess_pipeline(
    input_file='../stroke_data_raw/healthcare-dataset-stroke-data.csv',
    output_dir='stroke_data_preprocessing'
)

print(f"Processed data saved to: {result['train_file']}")
```

### Custom Configuration
```python
# Custom preprocessing steps
preprocessor = StrokeDataPreprocessor(random_state=123)

# Load and clean data
df = preprocessor.load_data('path/to/data.csv')
df_clean = preprocessor.handle_missing_values(df)

# Split and encode
X, y = preprocessor.split_features_target(df_clean)
X_encoded = preprocessor.encode_categorical_variables(X)

# Continue with custom pipeline...
```

## 📞 Contact

**Author**: Sundari Oktamiyunda  
**Project**: ML Experimentation - Kriteria 1  
**Repository**: Eksperimen_SML_Sundari-Oktamiyunda

## 🔗 Next Steps

This preprocessed dataset is ready for:
- **Kriteria 2**: Model training with MLflow
- **Kriteria 3**: CI/CD pipeline implementation  
- **Kriteria 4**: Model monitoring and logging

## 📝 License

This project is for educational purposes. Dataset source: Kaggle Stroke Prediction Dataset.

---

*Generated as part of ML System and MLOps Learning - Kriteria 1 (Skilled Level)*