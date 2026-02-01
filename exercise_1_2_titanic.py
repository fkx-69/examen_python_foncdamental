"""
Exercice 1.2: Application au dataset Titanic
Démontre l'utilisation de la classe DataCleaner sur un cas réel.

Ce fichier utilise UNIQUEMENT les méthodes existantes de DataCleaner.
Nouvelles méthodes créées seulement si absolument nécessaire.
"""

import pandas as pd
import numpy as np
import re

# Import de la classe DataCleaner
from exercise_1_1_data_cleaner import DataCleaner


def create_sample_titanic_data() -> pd.DataFrame:
    """Crée un dataset d'exemple basé sur le Titanic."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'PassengerId': range(1, n_samples + 1),
        'Survived': np.random.choice([0, 1], n_samples),
        'Pclass': np.random.choice([1, 2, 3], n_samples),
        'Name': [f"Passenger {i}, {'Mr' if i % 2 == 0 else 'Mrs'}. Lastname" for i in range(n_samples)],
        'Sex': np.random.choice(['male', 'female'], n_samples),
        'Age': np.random.normal(30, 15, n_samples),
        'SibSp': np.random.choice([0, 1, 2, 3], n_samples),
        'Parch': np.random.choice([0, 1, 2], n_samples),
        'Ticket': [f"TICKET{i}" for i in range(n_samples)],
        'Fare': np.abs(np.random.normal(30, 20, n_samples)),
        'Cabin': [f"C{i}" if i % 3 == 0 else np.nan for i in range(n_samples)],
        'Embarked': np.random.choice(['C', 'Q', 'S', np.nan], n_samples, p=[0.3, 0.2, 0.45, 0.05])
    }
    
    df = pd.DataFrame(data)
    missing_age_indices = np.random.choice(df.index, 20, replace=False)
    df.loc[missing_age_indices, 'Age'] = np.nan
    
    return df


if __name__ == "__main__":
    print("="*80)
    print("EXERCICE 1.2: APPLICATION AU DATASET TITANIC")
    print("="*80)
    print("\nUtilisation des méthodes EXISTANTES de DataCleaner\n")
    print("-"*80)
    
    # =========================================================================
    # ÉTAPE 1: Charger les données
    # =========================================================================
    
    print("\n### ÉTAPE 1: Chargement des données ###\n")
    
    cleaner = DataCleaner(filepath='titanic.csv')
    
    try:
        cleaner.load_data()
    except FileNotFoundError:
        print("⚠ Fichier titanic.csv non trouvé")
        print("Création de données d'exemple...")
        sample_data = create_sample_titanic_data()
        sample_data.to_csv('titanic.csv', index=False)
        cleaner.load_data()
    
    print(f"\nDimensions: {cleaner.df.shape}")
    print(f"\nValeurs manquantes:")
    missing = cleaner.df.isnull().sum()
    for col, count in missing[missing > 0].items():
        print(f"  {col}: {count}")
    
    # =========================================================================
    # ÉTAPE 2map({'female': 0, 'male': 1})
    cleaner.df['Sex_Encoded'] = cleaner.df['Sex'].map({'female': 0, 'male': 1})
    print("Variable 'Sex' encodée manuellement")
    
    # Embarked - Utilise la méthode générique encode_categorical
    print("\n--- Encodage Embarked ---")
    cleaner.encode_categorical('Embarked', method='onehot')
    
    # =========================================================================
    # ÉTAPE 4: FEATURE ENGINEERING (nouvelles features)
    # =========================================================================
    
    print("\n### ÉTAPE 4: Feature Engineering ###\n")
    
    # Création de features dérivées (code manuel car pas de méthode existante)
    print("--- Création features famille ---")
    if 'SibSp' in cleaner.df.columns and 'Parch' in cleaner.df.columns:
        cleaner.df['FamilySize'] = cleaner.df['SibSp'] + cleaner.df['Parch'] + 1
        cleaner.df['IsAlone'] = (cleaner.df['FamilySize'] == 1).astype(int)
        print(f"FamilySize créée (min={cleaner.df['FamilySize'].min()}, max={cleaner.df['FamilySize'].max()})")
        print(f"IsAlone créée ({cleaner.df['IsAlone'].sum()} passagers seuls)")
    
    # Extraction du titre depuis le nom
    print("\n--- Extraction titres depuis noms ---")
    if 'Name' in cleaner.df.columns:
        cleaner.df['Title'] = cleaner.df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        # Regroupe les titres rares
        title_mapping = {'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master'}
        cleaner.df['Title'] = cleaner.df['Title'].apply(
            lambda x: title_mapping.get(x, 'Other') if pd.notna(x) else 'Other'
        )
        unique_titles = cleaner.df['Title'].unique()
        print(f"Titres extraits: {list(unique_titles)}")
        
        # Encode le titre
        cleaner.encode_categorical('Title', method='onehot')
    
    # Catégories d'âge
    print("\n--- Catégorisation Age ---")
    if 'Age' in cleaner.df.columns:
        bins = [0, 12, 18, 35, 60, 100]
        labels = ['Child', 'Teenager', 'Adult', 'MiddleAge', 'Senior']
        cleaner.df['AgeCategory'] = pd.cut(cleaner.df['Age'], bins=bins, labels=labels)
        print(f"AgeCategory créée")
        cleaner.encode_categorical('AgeCategory', method='onehot', prefix='Age')
    
    # Catégories de prix
    print("\n--- Catégorisation Fare ---")
    if 'Fare' in cleaner.df.columns:
        cleaner.df['FareCategory'] = pd.qcut(
            cleaner.df['Fare'], q=4, 
            labels=['Low', 'Medium', 'High', 'VeryHigh'],
            duplicates='drop'
        )
        print(f"FareCategory créée")
        cleaner.encode_categorical('FareCategory', method='onehot', prefix='Fare')
    
    # =========================================================================
    # ÉTAPE 5: Sauvegarder et Analyser
    # =========================================================================
    
    print("\n### ÉTAPE 5: Sauvegarde et Analyse ###\n")
    
    # Sauvegarde
    cleaner.save_data('titanic_processed.csv')
    
    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DU TRAITEMENT")
    print("="*80)
    
    print(f"Dimensions finales: {cleaner.df.shape}")
    print(f"Nombre de features: {len(cleaner.df.columns)}")
    
    # Valeurs manquantes finales
    missing_final = cleaner.df.isnull().sum().sum()
    print(f"Valeurs manquantes restantes: {missing_final}")
    
    # Features créées
    original_cols = {'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 
                    'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'}
    new_cols = set(cleaner.df.columns) - original_cols
    
    print(f"\nNouvelles features créées ({len(new_cols)}):")
    for col in sorted(new_cols):
        print(f"  - {col}")
    
    print("\n" + "="*80)
    print("EXERCICE 1.2 TERMINÉ!")
    print("="*80)
    
    print("\nCe qui a été démontré:")
    print("  Utilisation des méthodes EXISTANTES de DataCleaner")
    print("      - load_data()")
    print("      - remove_duplicates()")
    print("      - handle_missing_values()")
    print("      - encode_categorical()")
    print("      - save_data()")
    print("  Code manuel pour features spécifiques au Titanic")
    print("      - Extraction titres")
    print("      - Taille famille")
    print("      - Catégorisation variables")
    print("  RÉUTILISATION maximale du code de l'exercice 1.1")
    
    print("\nFichier généré: titanic_processed.csv")
