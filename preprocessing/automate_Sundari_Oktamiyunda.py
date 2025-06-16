import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


class StrokeDataPreprocessor:
    """
    Automated preprocessing pipeline for stroke prediction dataset
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.numerical_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
        self.categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        self.feature_names = None
        
    def load_data(self, file_path):
        """Load stroke dataset from CSV file"""
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {file_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        df_clean = df.copy()
        
        # Remove ID column
        if 'id' in df_clean.columns:
            df_clean = df_clean.drop('id', axis=1)
        
        # Handle BMI 'N/A' values
        bmi_na_count = (df_clean['bmi'] == 'N/A').sum()
        df_clean['bmi'] = pd.to_numeric(df_clean['bmi'], errors='coerce')
        
        # Impute missing BMI with median
        bmi_median = df_clean['bmi'].median()
        df_clean['bmi'].fillna(bmi_median, inplace=True)
        
        print(f"✅ Missing values handled: {bmi_na_count} BMI values imputed")
        return df_clean
    
    def encode_categorical_variables(self, X):
        """Encode categorical variables using one-hot encoding"""
        # Only encode columns that exist in the dataset
        existing_categorical = [col for col in self.categorical_features if col in X.columns]
        
        if existing_categorical:
            X_encoded = pd.get_dummies(X, columns=existing_categorical, drop_first=False)
            print(f"✅ Categorical encoding: {X.shape[1]} → {X_encoded.shape[1]} features")
        else:
            X_encoded = X.copy()
            print(f"⚠️ No categorical columns found - returning original data")
        
        return X_encoded
    
    def scale_numerical_features(self, X_train, X_test):
        """Scale numerical features using StandardScaler"""
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Identify numerical columns in encoded dataset
        numerical_cols_encoded = [col for col in X_train.columns if col in self.numerical_features]
        
        # Fit scaler on training data only
        X_train_scaled[numerical_cols_encoded] = self.scaler.fit_transform(X_train[numerical_cols_encoded])
        X_test_scaled[numerical_cols_encoded] = self.scaler.transform(X_test[numerical_cols_encoded])
        
        print(f"✅ Feature scaling applied to {len(numerical_cols_encoded)} numerical features")
        return X_train_scaled, X_test_scaled
    
    def balance_classes(self, X_train, y_train):
        """Balance classes using SMOTE oversampling"""
        original_distribution = y_train.value_counts()
        
        # Check if we have enough samples and classes for SMOTE
        if len(original_distribution) < 2:
            print(f"⚠️ Only one class present - skipping class balancing")
            return X_train, y_train
        
        # Check if minority class has enough samples for SMOTE (need at least 6 for default k_neighbors=5)
        min_class_count = original_distribution.min()
        if min_class_count < 6:
            print(f"⚠️ Insufficient samples for SMOTE (min class: {min_class_count}) - skipping balancing")
            return X_train, y_train
        
        # Apply SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        new_distribution = pd.Series(y_train_balanced).value_counts()
        print(f"✅ Class balancing: {len(X_train)} → {len(X_train_balanced)} samples")
        
        return X_train_balanced, y_train_balanced
    
    def split_features_target(self, df):
        """Split dataset into features and target variable"""
        if 'stroke' not in df.columns:
            raise ValueError("Target column 'stroke' not found in dataset")
        
        X = df.drop('stroke', axis=1)
        y = df['stroke']
        return X, y
    
    def train_test_split_data(self, X, y, test_size=0.2):
        """Split data into training and testing sets with stratification"""
        # Check if we have enough samples for stratification
        min_class_count = y.value_counts().min()
        
        if min_class_count < 2:
            # Use random split without stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            print(f"⚠️ Insufficient samples for stratification - using random split")
        else:
            # Use stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        
        print(f"✅ Data split: Train {len(X_train)}, Test {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, output_dir):
        """Save processed data to specified directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training data
        train_data = pd.concat([X_train, y_train], axis=1)
        train_path = os.path.join(output_dir, 'train_data_processed.csv')
        train_data.to_csv(train_path, index=False)
        
        # Save test data
        test_data = pd.concat([X_test, y_test], axis=1)
        test_path = os.path.join(output_dir, 'test_data_processed.csv')
        test_data.to_csv(test_path, index=False)
        
        # Save feature information
        feature_info = {
            'total_features': list(X_train.columns),
            'numerical_features': [col for col in X_train.columns if col in self.numerical_features],
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'preprocessing_steps': [
                'Missing value imputation',
                'Categorical encoding',
                'Feature scaling',
                'Class balancing (SMOTE)'
            ]
        }
        
        info_path = os.path.join(output_dir, 'feature_info.json')
        with open(info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"✅ Processed data saved to: {output_dir}")
        return train_path, test_path, info_path
    
    def preprocess_pipeline(self, input_file, output_dir, test_size=0.2):
        """
        Complete preprocessing pipeline
        
        Args:
            input_file (str): Path to raw dataset CSV file
            output_dir (str): Directory to save processed data
            test_size (float): Proportion of data for testing
            
        Returns:
            dict: Paths to saved files and processing statistics
        """
        print("🚀 Starting stroke data preprocessing pipeline...")
        
        # Step 1: Load data
        df = self.load_data(input_file)
        
        # Step 2: Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Step 3: Split features and target
        X, y = self.split_features_target(df_clean)
        
        # Step 4: Encode categorical variables
        X_encoded = self.encode_categorical_variables(X)
        
        # Step 5: Train-test split
        X_train, X_test, y_train, y_test = self.train_test_split_data(X_encoded, y, test_size)
        
        # Step 6: Scale numerical features
        X_train_scaled, X_test_scaled = self.scale_numerical_features(X_train, X_test)
        
        # Step 7: Balance classes (only on training data)
        X_train_balanced, y_train_balanced = self.balance_classes(X_train_scaled, y_train)
        
        # Step 8: Save processed data
        train_path, test_path, info_path = self.save_processed_data(
            X_train_balanced, X_test_scaled, y_train_balanced, y_test, output_dir
        )
        
        # Return processing summary
        result = {
            'train_file': train_path,
            'test_file': test_path,
            'feature_info_file': info_path,
            'original_shape': df.shape,
            'final_train_shape': X_train_balanced.shape,
            'final_test_shape': X_test_scaled.shape,
            'class_distribution': {
                'original': y.value_counts().to_dict(),
                'train_balanced': pd.Series(y_train_balanced).value_counts().to_dict(),
                'test': y_test.value_counts().to_dict()
            }
        }
        
        print("🎉 Preprocessing pipeline completed successfully!")
        return result


def main():
    """Main function to run preprocessing pipeline"""
    
    # Configuration
    input_file = '../dataset_raw/healthcare-dataset-stroke-data.csv'
    output_dir = 'data_preprocessing'
    
    # Initialize preprocessor
    preprocessor = StrokeDataPreprocessor(random_state=42)
    
    # Run preprocessing pipeline
    try:
        result = preprocessor.preprocess_pipeline(input_file, output_dir)
        
        # Print summary
        print("\n" + "="*50)
        print("PREPROCESSING SUMMARY")
        print("="*50)
        print(f"📊 Original dataset: {result['original_shape']}")
        print(f"📊 Final train set: {result['final_train_shape']}")
        print(f"📊 Final test set: {result['final_test_shape']}")
        print(f"📁 Files saved:")
        print(f"  - {result['train_file']}")
        print(f"  - {result['test_file']}")
        print(f"  - {result['feature_info_file']}")
        
    except Exception as e:
        print(f"❌ Error in preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    main()