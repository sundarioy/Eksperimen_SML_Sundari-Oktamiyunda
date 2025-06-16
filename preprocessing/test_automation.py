import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from automate_Sundari_Oktamiyunda import StrokeDataPreprocessor


class TestStrokeDataPreprocessor(unittest.TestCase):
    """Test cases for StrokeDataPreprocessor class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.preprocessor = StrokeDataPreprocessor(random_state=42)
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'gender': ['Male', 'Female', 'Male', 'Female', 'Other'],
            'age': [67, 61, 80, 49, 79],
            'hypertension': [0, 0, 0, 0, 1],
            'heart_disease': [1, 0, 1, 0, 0],
            'ever_married': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
            'work_type': ['Private', 'Self-employed', 'Private', 'Private', 'Self-employed'],
            'Residence_type': ['Urban', 'Rural', 'Rural', 'Urban', 'Rural'],
            'avg_glucose_level': [228.69, 202.21, 105.92, 171.23, 174.12],
            'bmi': [36.6, 'N/A', 32.5, 34.4, 24.0],
            'smoking_status': ['formerly smoked', 'never smoked', 'never smoked', 'smokes', 'never smoked'],
            'stroke': [1, 1, 1, 1, 1]
        })
        
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.temp_dir, 'test_data.csv')
        self.sample_data.to_csv(self.test_csv_path, index=False)
    
    def tearDown(self):
        """Clean up after each test method"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_load_data(self):
        """Test data loading functionality"""
        df = self.preprocessor.load_data(self.test_csv_path)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)
        self.assertEqual(len(df.columns), 12)
        self.assertIn('stroke', df.columns)
    
    def test_load_data_file_not_found(self):
        """Test error handling for missing file"""
        with self.assertRaises(FileNotFoundError):
            self.preprocessor.load_data('nonexistent_file.csv')
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        df_clean = self.preprocessor.handle_missing_values(self.sample_data)
        
        # ID column should be removed
        self.assertNotIn('id', df_clean.columns)
        
        # BMI should be numeric and no missing values
        self.assertTrue(pd.api.types.is_numeric_dtype(df_clean['bmi']))
        self.assertEqual(df_clean['bmi'].isnull().sum(), 0)
        
        # Check that 'N/A' was replaced
        self.assertNotIn('N/A', df_clean['bmi'].values)
    
    def test_split_features_target(self):
        """Test feature-target splitting"""
        X, y = self.preprocessor.split_features_target(self.sample_data)
        
        self.assertNotIn('stroke', X.columns)
        self.assertEqual(len(y), len(self.sample_data))
        self.assertTrue(all(y.isin([0, 1])))
    
    def test_split_features_target_no_stroke_column(self):
        """Test error handling when stroke column is missing"""
        df_no_stroke = self.sample_data.drop('stroke', axis=1)
        
        with self.assertRaises(ValueError):
            self.preprocessor.split_features_target(df_no_stroke)
    
    def test_encode_categorical_variables(self):
        """Test categorical encoding"""
        df_clean = self.preprocessor.handle_missing_values(self.sample_data)
        X, y = self.preprocessor.split_features_target(df_clean)
        X_encoded = self.preprocessor.encode_categorical_variables(X)
        
        # Should have more columns after encoding
        self.assertGreater(X_encoded.shape[1], X.shape[1])
        
        # Original categorical columns should be gone
        for cat_col in self.preprocessor.categorical_features:
            self.assertNotIn(cat_col, X_encoded.columns)
        
        # Should have dummy columns
        dummy_cols = [col for col in X_encoded.columns if '_' in col]
        self.assertGreater(len(dummy_cols), 0)
    
    def test_train_test_split_data(self):
        """Test train-test splitting"""
        df_clean = self.preprocessor.handle_missing_values(self.sample_data)
        X, y = self.preprocessor.split_features_target(df_clean)
        
        X_train, X_test, y_train, y_test = self.preprocessor.train_test_split_data(X, y, test_size=0.2)
        
        # Check sizes
        total_samples = len(X)
        expected_train_size = int(total_samples * 0.8)
        expected_test_size = total_samples - expected_train_size
        
        self.assertEqual(len(X_train), expected_train_size)
        self.assertEqual(len(X_test), expected_test_size)
        self.assertEqual(len(y_train), expected_train_size)
        self.assertEqual(len(y_test), expected_test_size)
    
    def test_scale_numerical_features(self):
        """Test feature scaling"""
        df_clean = self.preprocessor.handle_missing_values(self.sample_data)
        X, y = self.preprocessor.split_features_target(df_clean)
        X_encoded = self.preprocessor.encode_categorical_variables(X)
        X_train, X_test, y_train, y_test = self.preprocessor.train_test_split_data(X_encoded, y)
        
        X_train_scaled, X_test_scaled = self.preprocessor.scale_numerical_features(X_train, X_test)
        
        # Check that scaling was applied
        numerical_cols = [col for col in X_train.columns if col in self.preprocessor.numerical_features]
        for col in numerical_cols:
            # Scaled values should have different mean/std than original
            if X_train[col].std() > 0:  # Avoid division by zero
                self.assertNotAlmostEqual(X_train[col].mean(), X_train_scaled[col].mean(), places=1)
    
    def test_balance_classes(self):
        """Test class balancing with SMOTE"""
        df_clean = self.preprocessor.handle_missing_values(self.sample_data)
        X, y = self.preprocessor.split_features_target(df_clean)
        X_encoded = self.preprocessor.encode_categorical_variables(X)
        
        # Create imbalanced data with sufficient samples for SMOTE
        # SMOTE needs at least 6 samples for minority class (k_neighbors=5 default)
        imbalanced_X = pd.concat([X_encoded] * 3, ignore_index=True)  # 15 samples
        imbalanced_y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 9:6 ratio
        
        X_balanced, y_balanced = self.preprocessor.balance_classes(imbalanced_X, imbalanced_y)
        
        # Should have balanced classes
        class_counts = pd.Series(y_balanced).value_counts()
        self.assertEqual(class_counts[0], class_counts[1])
        
        # Should have more samples than original
        self.assertGreaterEqual(len(X_balanced), len(imbalanced_X))
    
    def test_balance_classes_insufficient_samples(self):
        """Test class balancing with insufficient samples"""
        df_clean = self.preprocessor.handle_missing_values(self.sample_data)
        X, y = self.preprocessor.split_features_target(df_clean)
        X_encoded = self.preprocessor.encode_categorical_variables(X)
        
        # Create data with insufficient samples for SMOTE
        insufficient_y = pd.Series([0, 0, 0, 0, 1])  # Only 1 minority sample
        
        X_result, y_result = self.preprocessor.balance_classes(X_encoded, insufficient_y)
        
        # Should return original data unchanged
        self.assertEqual(len(X_result), len(X_encoded))
        self.assertTrue(y_result.equals(insufficient_y))
    
    def test_balance_classes_single_class(self):
        """Test class balancing with single class"""
        df_clean = self.preprocessor.handle_missing_values(self.sample_data)
        X, y = self.preprocessor.split_features_target(df_clean)
        X_encoded = self.preprocessor.encode_categorical_variables(X)
        
        # Create single class data
        single_class_y = pd.Series([0, 0, 0, 0, 0])
        
        X_result, y_result = self.preprocessor.balance_classes(X_encoded, single_class_y)
        
        # Should return original data unchanged
        self.assertEqual(len(X_result), len(X_encoded))
        self.assertTrue(y_result.equals(single_class_y))
    
    def test_save_processed_data(self):
        """Test saving processed data"""
        df_clean = self.preprocessor.handle_missing_values(self.sample_data)
        X, y = self.preprocessor.split_features_target(df_clean)
        X_train, X_test, y_train, y_test = self.preprocessor.train_test_split_data(X, y)
        
        output_dir = os.path.join(self.temp_dir, 'output')
        train_path, test_path, info_path = self.preprocessor.save_processed_data(
            X_train, X_test, y_train, y_test, output_dir
        )
        
        # Check files exist
        self.assertTrue(os.path.exists(train_path))
        self.assertTrue(os.path.exists(test_path))
        self.assertTrue(os.path.exists(info_path))
        
        # Check file contents
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        self.assertIn('stroke', train_data.columns)
        self.assertIn('stroke', test_data.columns)
        self.assertEqual(len(train_data), len(X_train))
        self.assertEqual(len(test_data), len(X_test))
    
    def test_preprocess_pipeline_integration(self):
        """Test complete preprocessing pipeline"""
        # Create larger sample data for proper testing
        larger_sample_data = pd.concat([self.sample_data] * 4, ignore_index=True)  # 20 samples
        
        # Create balanced classes for testing
        larger_sample_data.loc[:9, 'stroke'] = 0   # 10 no stroke
        larger_sample_data.loc[10:, 'stroke'] = 1  # 10 stroke
        
        # Update IDs to be unique
        larger_sample_data['id'] = range(1, len(larger_sample_data) + 1)
        
        # Save to temp file
        temp_csv_path = os.path.join(self.temp_dir, 'larger_test_data.csv')
        larger_sample_data.to_csv(temp_csv_path, index=False)
        
        output_dir = os.path.join(self.temp_dir, 'pipeline_output')
        
        result = self.preprocessor.preprocess_pipeline(temp_csv_path, output_dir)
        
        # Check return structure
        required_keys = ['train_file', 'test_file', 'feature_info_file', 
                        'original_shape', 'final_train_shape', 'final_test_shape']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check files were created
        self.assertTrue(os.path.exists(result['train_file']))
        self.assertTrue(os.path.exists(result['test_file']))
        self.assertTrue(os.path.exists(result['feature_info_file']))
        
        # Check data integrity
        train_data = pd.read_csv(result['train_file'])
        test_data = pd.read_csv(result['test_file'])
        
        self.assertIn('stroke', train_data.columns)
        self.assertIn('stroke', test_data.columns)
        self.assertGreater(len(train_data), 0)
        self.assertGreater(len(test_data), 0)


class TestPreprocessorEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        self.preprocessor = StrokeDataPreprocessor()
        
        # Create sample test data for edge cases
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'gender': ['Male', 'Female', 'Male', 'Female', 'Other'],
            'age': [67, 61, 80, 49, 79],
            'hypertension': [0, 0, 0, 0, 1],
            'heart_disease': [1, 0, 1, 0, 0],
            'ever_married': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
            'work_type': ['Private', 'Self-employed', 'Private', 'Private', 'Self-employed'],
            'Residence_type': ['Urban', 'Rural', 'Rural', 'Urban', 'Rural'],
            'avg_glucose_level': [228.69, 202.21, 105.92, 171.23, 174.12],
            'bmi': [36.6, 'N/A', 32.5, 34.4, 24.0],
            'smoking_status': ['formerly smoked', 'never smoked', 'never smoked', 'smokes', 'never smoked'],
            'stroke': [1, 1, 1, 1, 1]
        })
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(Exception):
            self.preprocessor.handle_missing_values(empty_df)
    
    def test_missing_required_columns(self):
        """Test handling of missing required columns"""
        incomplete_df = pd.DataFrame({
            'age': [25, 30],
            'stroke': [0, 1]
        })
        
        X, y = self.preprocessor.split_features_target(incomplete_df)
        
        # Should handle missing categorical columns gracefully
        X_encoded = self.preprocessor.encode_categorical_variables(X)
        
        # If no categorical columns exist, should return original data
        self.assertEqual(X_encoded.shape, X.shape)
        
        # Should contain the same columns
        self.assertTrue(all(col in X_encoded.columns for col in X.columns))
    
    def test_train_test_split_stratification(self):
        """Test train-test split with sufficient samples for stratification"""
        # Create larger balanced dataset
        larger_df = pd.concat([self.sample_data] * 4, ignore_index=True)
        larger_df.loc[:9, 'stroke'] = 0   # 10 no stroke  
        larger_df.loc[10:, 'stroke'] = 1  # 10 stroke
        
        df_clean = self.preprocessor.handle_missing_values(larger_df)
        X, y = self.preprocessor.split_features_target(df_clean)
        
        X_train, X_test, y_train, y_test = self.preprocessor.train_test_split_data(X, y, test_size=0.2)
        
        # Check that both classes are represented in train and test
        self.assertEqual(len(y_train.unique()), 2)
        self.assertEqual(len(y_test.unique()), 2)
    
    def test_train_test_split_insufficient_for_stratification(self):
        """Test train-test split with insufficient samples"""
        small_df = pd.DataFrame({
            'age': [25, 30],
            'bmi': [22.0, 25.0],
            'stroke': [0, 1]  # Only one sample per class
        })
        
        X, y = self.preprocessor.split_features_target(small_df)
        
        # Should work without stratification
        X_train, X_test, y_train, y_test = self.preprocessor.train_test_split_data(X, y, test_size=0.5)
        
        # Check basic functionality
        self.assertEqual(len(X_train) + len(X_test), len(X))
        self.assertEqual(len(y_train) + len(y_test), len(y))


def run_tests():
    """Run all unit tests"""
    print("🧪 Running unit tests for stroke data preprocessing...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestStrokeDataPreprocessor))
    test_suite.addTest(unittest.makeSuite(TestPreprocessorEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed")
        print(f"❌ {len(result.errors)} error(s) occurred")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()