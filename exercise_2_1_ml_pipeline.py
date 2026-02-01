"""
Exercice 2.1: Pipeline ML Modulaire
Refactorisation du code procédural en composants OOP.
Uniquement les fonctionnalités demandées: loader, splitter, scaler, model, pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from typing import Tuple, Any, Dict


class DataLoader:
    """Charge les données et sépare features/target."""
    def __init__(self, filepath: str, target_column: str):
        self.filepath = filepath
        self.target_column = target_column
        
    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        data = pd.read_csv(self.filepath)
        X = data.drop(self.target_column, axis=1)
        y = data[self.target_column]
        return X, y


class DataSplitter:
    """Divise les données en ensembles d'entraînement et de test."""
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        
    def split(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        return train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )


class Scaler:
    """Gère la mise à l'échelle des données (StandardScaler)."""
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        return self.scaler.fit_transform(X_train)
        
    def transform(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(X_test)


class ModelHandler:
    """Gère l'entraînement et l'évaluation du modèle (RandomForest)."""
    def __init__(self, n_estimators: int = 100):
        self.model = RandomForestClassifier(n_estimators=n_estimators)
        
    def train(self, X_train: np.ndarray, y_train: pd.Series) -> None:
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)
        
    def evaluate(self, X_test: np.ndarray, y_test: pd.Series) -> str:
        predictions = self.predict(X_test)
        return classification_report(y_test, predictions)


class MLPipeline:
    """Facade orchestrant le pipeline complet."""
    def __init__(self, loader, splitter, scaler, model_handler):
        self.loader = loader
        self.splitter = splitter
        self.scaler = scaler
        self.model_handler = model_handler
        
    def run(self):
        # 1. Load
        X, y = self.loader.load()
        
        # 2. Split
        X_train, X_test, y_train, y_test = self.splitter.split(X, y)
        
        # 3. Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 4. Train
        self.model_handler.train(X_train_scaled, y_train)
        
        # 5. Evaluate
        report = self.model_handler.evaluate(X_test_scaled, y_test)
        print("Rapport de classification:")
        print(report)
