"""
Exercice 3.3: Package Final
Intégration des concepts dans une structure unifiée.
Orchestration simple des composants créés précédemment.
"""

import pandas as pd
from typing import Dict, Any

# Import des composants des exercices précédents
from exercise_1_1_data_cleaner import DataCleaner
from exercise_2_1_ml_pipeline import DataLoader, DataSplitter, Scaler, ModelHandler, MLPipeline


class DataSciencePackage:
    """Façade unifiée pour le projet de Data Science."""
    
    def __init__(self, filepath: str, target_col: str):
        self.filepath = filepath
        self.target_col = target_col
        self.cleaner = DataCleaner(filepath)
        self.pipeline = None
        
    def run_full_workflow(self):
        print("=== Lancement du Workflow Data Science ===")
        
        # 1. Nettoyage
        print("\n1. Nettoyage des données...")
        try:
            self.cleaner.load_data()
            self.cleaner.clean()
            # Sauvegarde temporaire pour le loader du pipeline car DataLoader prend un path
            temp_path = "temp_cleaned_data.csv"
            self.cleaner.save_data(temp_path)
        except Exception as e:
            print(f"Erreur nettoyage: {e}")
            return

        # 2. Pipeline ML
        print("\n2. Exécution du Pipeline ML...")
        loader = DataLoader(temp_path, self.target_col)
        splitter = DataSplitter()
        scaler = Scaler()
        model = ModelHandler()
        
        self.pipeline = MLPipeline(loader, splitter, scaler, model)
        self.pipeline.run()
        
        print("\n=== Workflow Terminé ===")


if __name__ == "__main__":
    # Test d'intégration
    # (Supposons que customer_churn.csv existe déjà ou sera créé par le loader si besoin)
    # Pour le test, on va créer un fichier dummy si absent
    try:
        pd.read_csv('customer_churn.csv')
    except:
        import numpy as np
        df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=['A', 'B', 'C', 'Churn'])
        df['Churn'] = np.random.choice([0, 1], size=100)
        df.to_csv('customer_churn.csv', index=False)
        
    package = DataSciencePackage('customer_churn.csv', 'Churn')
    package.run_full_workflow()
