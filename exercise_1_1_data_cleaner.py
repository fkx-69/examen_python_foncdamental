"""
Exercice 1.1: Refactorisation du code procédural en classe DataCleaner
Transforme le script procédural de nettoyage de données en une classe réutilisable.
"""

import pandas as pd
import numpy as np
from typing import Optional, List


class DataCleaner:
    """
    Classe pour nettoyer et transformer des données de manière réutilisable.
    
    Cette classe encapsule toutes les opérations de nettoyage de données
    dans une interface orientée objet, permettant la réutilisation et la 
    configuration flexible.
    """
    
    def __init__(self, filepath: Optional[str] = None):
        """
        Initialise le nettoyeur de données.
        """
        self.filepath = filepath
        self.df = None
        
    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Charge les données depuis un fichier CSV.
        """
        if filepath:
            self.filepath = filepath
            
        if not self.filepath:
            raise ValueError("Aucun chemin de fichier fourni")
            
        self.df = pd.read_csv(self.filepath)
        print(f"✓ Données chargées: {len(self.df)} lignes, {len(self.df.columns)} colonnes")
        return self.df
    
    def remove_duplicates(self) -> 'DataCleaner':
        """
        Supprime les lignes dupliquées.
        """
        if self.df is None:
            raise ValueError("Aucune donnée chargée. Utilisez load_data() d'abord.")
        
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        duplicates_removed = initial_rows - len(self.df)
        
        print(f"✓ {duplicates_removed} doublons supprimés")
        return self
    
    def handle_missing_values(self, columns: Optional[List[str]] = None) -> 'DataCleaner':
        """
        Gère les valeurs manquantes.
        """
        if self.df is None:
            raise ValueError("Aucune donnée chargée. Utilisez load_data() d'abord.")
        
        cols_to_process = columns or self.df.columns
        missing_before = self.df.isnull().sum().sum()
        
        for col in cols_to_process:
            if col not in self.df.columns:
                continue
                
            if self.df[col].isnull().sum() == 0:
                continue
            
            # Médiane pour numérique, mode pour catégoriel
            if self.df[col].dtype in ['float64', 'int64']:
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                mode_value = self.df[col].mode()
                if len(mode_value) > 0:
                    self.df[col] = self.df[col].fillna(mode_value[0])
        
        missing_after = self.df.isnull().sum().sum()
        print(f"✓ {missing_before - missing_after} valeurs manquantes traitées")
        return self
    
    def remove_outliers_iqr(self, columns: Optional[List[str]] = None) -> 'DataCleaner':
        """
        Supprime les valeurs aberrantes en utilisant la méthode IQR.
        """
        if self.df is None:
            raise ValueError("Aucune donnée chargée. Utilisez load_data() d'abord.")
        
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
        print(f"✓ {outliers_removed} valeurs aberrantes supprimées (méthode IQR)")
        return self
    
    def clean(self) -> pd.DataFrame:
        """
        Exécute le pipeline de nettoyage complet.
        """
        print("\n=== Début du nettoyage des données ===\n")
        
        self.remove_duplicates()
        self.handle_missing_values()
        self.remove_outliers_iqr()
        
        print("\n=== Nettoyage terminé ===")
        print(f"Lignes finales: {len(self.df)}")
        
        return self.df
    
    def save_data(self, output_path: str, index: bool = False) -> None:
        """
        Sauvegarde les données nettoyées dans un fichier CSV.
        """
        if self.df is None:
            raise ValueError("Aucune donnée à sauvegarder")
        
        self.df.to_csv(output_path, index=index)
        print(f"✓ Données sauvegardées dans: {output_path}")
    
    def get_data(self) -> pd.DataFrame:
        """
        Retourne le DataFrame actuel.
        """
        if self.df is None:
            raise ValueError("Aucune donnée chargée")
        return self.df
    
    # ========================================================================
    # MÉTHODES ADDITIONNELLES pour l'exercice 1.2
    # ========================================================================
    
    def encode_categorical(self, column: str, method: str = 'label', prefix: Optional[str] = None) -> 'DataCleaner':
        """
        Encode une variable catégorielle
        """
        if self.df is None:
            raise ValueError("Chargez les données d'abord")
        
        if column not in self.df.columns:
            print(f"⚠ Colonne '{column}' non trouvée")
            return self
        
        if method == 'label':
            unique_values = self.df[column].dropna().unique()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            self.df[f'{column}_Encoded'] = self.df[column].map(mapping)
            print(f"✓ Variable '{column}' encodée (label encoding)")
        
        elif method == 'onehot':
            prefix = prefix or column
            dummies = pd.get_dummies(self.df[column], prefix=prefix)
            self.df = pd.concat([self.df, dummies], axis=1)
            print(f"✓ Variable '{column}' encodée (one-hot encoding)")
        
        return self
