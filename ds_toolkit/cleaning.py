"""
Data Cleaning Module.
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from .utils import logging_decorator, timing_decorator

class DataCleaner:
    """
    Class to clean and transform data in a reusable way.
    
    Encapsulates all data cleaning operations.
    """
    
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        self.df = None
        
    @logging_decorator
    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Loads data from a CSV file."""
        if filepath:
            self.filepath = filepath
            
        if not self.filepath:
            raise ValueError("No filepath provided")
            
        self.df = pd.read_csv(self.filepath)
        print(f"✓ Data loaded: {len(self.df)} rows, {len(self.df.columns)} columns")
        return self.df
    
    @timing_decorator
    def remove_duplicates(self) -> 'DataCleaner':
        """Removes duplicate rows."""
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        duplicates_removed = initial_rows - len(self.df)
        
        print(f"✓ {duplicates_removed} duplicates removed")
        return self
    
    def handle_missing_values(self, columns: Optional[List[str]] = None) -> 'DataCleaner':
        """Handles missing values (median for numeric, mode for categorical)."""
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        cols_to_process = columns or self.df.columns
        missing_before = self.df.isnull().sum().sum()
        
        for col in cols_to_process:
            if col not in self.df.columns:
                continue
                
            if self.df[col].isnull().sum() == 0:
                continue
            
            if self.df[col].dtype in ['float64', 'int64']:
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                mode_value = self.df[col].mode()
                if len(mode_value) > 0:
                    self.df[col] = self.df[col].fillna(mode_value[0])
        
        missing_after = self.df.isnull().sum().sum()
        print(f"✓ {missing_before - missing_after} missing values handled")
        return self
    
    def remove_outliers_iqr(self, columns: Optional[List[str]] = None) -> 'DataCleaner':
        """Removes outliers using the IQR method."""
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        else:
            numeric_cols = [col for col in columns if col in self.df.columns]
        
        initial_rows = len(self.df)
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            self.df = self.df[
                (self.df[col] >= lower_bound) & 
                (self.df[col] <= upper_bound)
            ]
        
        outliers_removed = initial_rows - len(self.df)
        print(f"✓ {outliers_removed} outliers removed (IQR method)")
        return self
    
    @logging_decorator
    @timing_decorator
    def clean(self) -> pd.DataFrame:
        """Executes the full cleaning pipeline."""
        print("\n=== Starting Data Cleaning ===\n")
        
        self.remove_duplicates()
        self.handle_missing_values()
        self.remove_outliers_iqr()
        
        print("\n=== Cleaning Finished ===")
        print(f"Final rows: {len(self.df)}")
        
        return self.df
    
    def save_data(self, output_path: str, index: bool = False) -> None:
        """Saves cleaned data to a CSV file."""
        if self.df is None:
            raise ValueError("No data to save")
        
        self.df.to_csv(output_path, index=index)
        print(f"✓ Data saved to: {output_path}")
    
    def get_data(self) -> pd.DataFrame:
        """Returns the current DataFrame."""
        if self.df is None:
            raise ValueError("No data loaded")
        return self.df

    def encode_categorical(self, column: str, method: str = 'label', prefix: Optional[str] = None) -> 'DataCleaner':
        """Encodes a categorical variable."""
        if self.df is None:
            raise ValueError("Load data first")
        
        if column not in self.df.columns:
            print(f"⚠ Column '{column}' not found")
            return self
        
        if method == 'label':
            unique_values = self.df[column].dropna().unique()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            self.df[f'{column}_Encoded'] = self.df[column].map(mapping)
            print(f"✓ Variable '{column}' encoded (label encoding)")
        
        elif method == 'onehot':
            prefix = prefix or column
            dummies = pd.get_dummies(self.df[column], prefix=prefix)
            self.df = pd.concat([self.df, dummies], axis=1)
            print(f"✓ Variable '{column}' encoded (one-hot encoding)")
        
        return self
