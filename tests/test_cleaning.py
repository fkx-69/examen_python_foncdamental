import unittest
import pandas as pd
import numpy as np
import os
from ds_toolkit.cleaning import DataCleaner

class TestDataCleaner(unittest.TestCase):
    
    def setUp(self):
        # Create a dummy CSV file
        self.filename = "test_data.csv"
        df = pd.DataFrame({
            'A': [1, 2, 2, 4, 100],  # 100 is an outlier?
            'B': [5, 5, 5, 5, 5],
            'C': [None, 1, 1, 1, 1],
            'cat': ['x', 'y', 'y', 'x', 'x']
        })
        df.to_csv(self.filename, index=False)
        self.cleaner = DataCleaner(self.filename)
        self.cleaner.load_data()

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)
            
    def test_load_data(self):
        self.assertIsNotNone(self.cleaner.df)
        self.assertEqual(len(self.cleaner.df), 5)

    def test_remove_duplicates(self):
        # Row 1 and 2 are almost same but A is different (2 vs 2).
        # Let's add a real duplicate
        df = pd.DataFrame({'A': [1, 1], 'B': [2, 2]})
        cleaner = DataCleaner()
        cleaner.df = df
        cleaner.remove_duplicates()
        self.assertEqual(len(cleaner.df), 1)

    def test_handle_missing_values(self):
        self.cleaner.handle_missing_values()
        self.assertEqual(self.cleaner.df['C'].isnull().sum(), 0)
        # Median of [1, 1, 1, 1] is 1.0
        self.assertEqual(self.cleaner.df['C'].iloc[0], 1.0)

    def test_remove_outliers_iqr(self):
        # A: 1, 2, 2, 4, 100.
        # Q1=2, Q3=4. IQR=2. Upper=4+3=7. 100 should be removed.
        # Actually in 5 items: 1, 2, 2, 4, 100
        # Q1 (25%) is 2.0
        # Q3 (75%) is 4.0
        # IQR = 2
        # Upper = 4 + 1.5*2 = 7
        # Lower = 2 - 1.5*2 = -1
        
        self.cleaner.remove_outliers_iqr(columns=['A'])
        self.assertFalse(100 in self.cleaner.df['A'].values)
        self.assertTrue(4 in self.cleaner.df['A'].values)

    def test_encode_categorical(self):
        self.cleaner.encode_categorical('cat', method='label')
        self.assertIn('cat_Encoded', self.cleaner.df.columns)
        
    def test_clean_pipeline(self):
        cleaned_df = self.cleaner.clean()
        self.assertIsNotNone(cleaned_df)

if __name__ == '__main__':
    unittest.main()
