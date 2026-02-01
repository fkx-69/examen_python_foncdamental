"""
Exercice 2.2: Pattern Strategy pour Validation Croisée
Implémentation du pattern Strategy pour changer d'algorithme de CV.
Minimisé aux stratégies essentielles.
"""

from abc import ABC, abstractmethod
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


class CrossValidationStrategy(ABC):
    """Interface pour les stratégies de validation croisée."""
    
    @abstractmethod
    def validate(self, model, X, y, n_splits=5):
        pass


class KFoldStrategy(CrossValidationStrategy):
    """Stratégie K-Fold standard."""
    
    def validate(self, model, X, y, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf)
        return scores


class StratifiedKFoldStrategy(CrossValidationStrategy):
    """Stratégie Stratified K-Fold (conserve la distribution des classes)."""
    
    def validate(self, model, X, y, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=skf)
        return scores


class ModelEvaluator:
    """Context utilisant une stratégie de validation."""
    
    def __init__(self, strategy: CrossValidationStrategy):
        self.strategy = strategy
        
    def set_strategy(self, strategy: CrossValidationStrategy):
        """Permet de changer de stratégie dynamiquement."""
        self.strategy = strategy
        
    def evaluate(self, model, X, y):
        print(f"Évaluation avec {self.strategy.__class__.__name__}...")
        scores = self.strategy.validate(model, X, y)
        print(f"Scores: {scores}")
        print(f"Moyenne: {scores.mean():.4f}")


if __name__ == "__main__":
    # Données d'exemple
    X = pd.DataFrame(np.random.rand(100, 5))
    y = pd.Series(np.random.randint(0, 2, 100))
    model = RandomForestClassifier()
    
    # Stratégie 1: K-Fold
    evaluator = ModelEvaluator(KFoldStrategy())
    evaluator.evaluate(model, X, y)
    
    print("-" * 30)
    
    # Changement de stratégie -> Stratified K-Fold
    evaluator.set_strategy(StratifiedKFoldStrategy())
    evaluator.evaluate(model, X, y)
