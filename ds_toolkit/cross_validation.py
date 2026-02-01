"""
Cross-Validation Strategy Pattern.
"""

from abc import ABC, abstractmethod
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
import pandas as pd
import numpy as np

class CrossValidationStrategy(ABC):
    """Interface for cross-validation strategies."""
    
    @abstractmethod
    def validate(self, model, X, y, n_splits=5):
        pass


class KFoldStrategy(CrossValidationStrategy):
    """Standard K-Fold strategy."""
    
    def validate(self, model, X, y, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf)
        return scores


class StratifiedKFoldStrategy(CrossValidationStrategy):
    """Stratified K-Fold strategy."""
    
    def validate(self, model, X, y, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=skf)
        return scores


class ModelEvaluator:
    """Context using a validation strategy."""
    
    def __init__(self, strategy: CrossValidationStrategy):
        self.strategy = strategy
        
    def set_strategy(self, strategy: CrossValidationStrategy):
        """Allows changing strategy dynamically."""
        self.strategy = strategy
        
    def evaluate(self, model, X, y):
        print(f"Evaluating with {self.strategy.__class__.__name__}...")
        scores = self.strategy.validate(model, X, y)
        print(f"Scores: {scores}")
        print(f"Mean Score: {scores.mean():.4f}")
        return scores
