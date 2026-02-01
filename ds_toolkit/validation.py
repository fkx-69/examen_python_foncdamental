"""
Data Validation Framework.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict

class ValidationRule(ABC):
    """Abstract class defining a validation rule."""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        pass


class NoMissingValuesRule(ValidationRule):
    """Checks for absence of missing values."""
    
    def __init__(self, columns: List[str] = None):
        super().__init__("No Missing Values")
        self.columns = columns
        
    def validate(self, df: pd.DataFrame) -> bool:
        cols = self.columns if self.columns else df.columns
        missing = df[cols].isnull().sum().sum()
        if missing > 0:
            print(f"Rule '{self.name}' failed: {missing} missing values")
            return False
        print(f"Rule '{self.name}' passed")
        return True


class DataTypeRule(ValidationRule):
    """Checks data types."""
    
    def __init__(self, expected_types: Dict[str, str]):
        super().__init__("Data Types Check")
        self.expected_types = expected_types
        
    def validate(self, df: pd.DataFrame) -> bool:
        passed = True
        for col, dtype in self.expected_types.items():
            if str(df[col].dtype) != dtype:
                print(f"Rule '{self.name}' failed for {col}: expected {dtype}, got {df[col].dtype}")
                passed = False
        if passed:
            print(f"Rule '{self.name}' passed")
        return passed


class DataValidator:
    """Validator orchestrating rules execution."""
    
    def __init__(self):
        self.rules = []
        
    def add_rule(self, rule: ValidationRule):
        self.rules.append(rule)
        
    def validate(self, df: pd.DataFrame) -> bool:
        print("\n--- Starting Validation ---")
        all_passed = True
        for rule in self.rules:
            if not rule.validate(df):
                all_passed = False
        
        print("--- Validation Finished ---")
        return all_passed
