"""
Exercice 3.2: Framework de Validation de Données
Structure modulaire pour valider des datasets.
Simplifié aux composants essentiels.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any


class ValidationRule(ABC):
    """Classe abstraite définissant une règle de validation."""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        pass


class NoMissingValuesRule(ValidationRule):
    """Vérifie l'absence de valeurs manquantes."""
    
    def __init__(self, columns: List[str] = None):
        super().__init__("No Missing Values")
        self.columns = columns
        
    def validate(self, df: pd.DataFrame) -> bool:
        cols = self.columns if self.columns else df.columns
        missing = df[cols].isnull().sum().sum()
        if missing > 0:
            print(f"Règle '{self.name}' échouée: {missing} valeurs manquantes")
            return False
        print(f"Règle '{self.name}' réussie")
        return True


class DataTypeRule(ValidationRule):
    """Vérifie les types de données."""
    
    def __init__(self, expected_types: Dict[str, str]):
        super().__init__("Data Types Check")
        self.expected_types = expected_types
        
    def validate(self, df: pd.DataFrame) -> bool:
        passed = True
        for col, dtype in self.expected_types.items():
            if str(df[col].dtype) != dtype:
                print(f"Règle '{self.name}' échouée pour {col}: attendu {dtype}, reçu {df[col].dtype}")
                passed = False
        if passed:
            print(f"Règle '{self.name}' réussie")
        return passed


class DataValidator:
    """Valideur orchestrant l'exécution des règles."""
    
    def __init__(self):
        self.rules = []
        
    def add_rule(self, rule: ValidationRule):
        self.rules.append(rule)
        
    def validate(self, df: pd.DataFrame) -> bool:
        print("\n--- Début de la validation ---")
        all_passed = True
        for rule in self.rules:
            if not rule.validate(df):
                all_passed = False
        
        print("--- Validation terminée ---")
        return all_passed


if __name__ == "__main__":
    # Données test
    df = pd.DataFrame({
        'age': [25, 30, None],
        'salary': [50000, 60000, 70000]
    })
    
    validator = DataValidator()
    # Règle 1: Pas de valeurs manquantes
    validator.add_rule(NoMissingValuesRule(columns=['age']))
    # Règle 2: Types corrects
    validator.add_rule(DataTypeRule({'salary': 'int64'}))
    
    is_valid = validator.validate(df)
    print(f"\nDataset valide ? {is_valid}")
