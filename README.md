# Boîte à outils de science des données orientée objet (ds_toolkit)

**Cours :** Master en Science des Données - Programmation Orientée Objet  
**Devoir :** Projet Final & Devoir à la Maison

## Vue d'ensemble du projet

Ce projet refactorise du code de science des données procédural en un package orienté objet robuste, modulaire et réutilisable (`ds_toolkit`). Il démontre l'application de principes avancés de génie logiciel aux flux de travail de science des données, y compris les modèles de conception, les tests unitaires et une structure de package solide.

## 📋 Fonctionnalités & Respect des Exigences

Cette soumission répond à toutes les exigences du devoir à la maison :

1.  **Structure complète du package** :
    - Code source organisé dans le répertoire `ds_toolkit/`.
    - `setup.py` inclus pour l'installation (`pip install -e .`).
    - Séparation claire des préoccupations (Nettoyage, Pipeline, Validation).

2.  **Modèles de conception implémentés** :
    - **Modèle Façade** (`ds_toolkit/facade.py`) : La classe `DataSciencePackage` fournit une interface simplifiée et unifiée pour l'ensemble du flux de travail (Nettoyage -> Modélisation), masquant la complexité à l'utilisateur.
    - **Modèle Stratégie** (`ds_toolkit/cross_validation.py`) : `CrossValidationStrategy` permet de changer dynamiquement d'algorithme de validation (par ex., `KFold`, `StratifiedKFold`) sans modifier le contexte.
    - **Modèle Décorateur** (`ds_toolkit/utils.py`) : `logging_decorator` et `timing_decorator` étendent le comportement des fonctions (journalisation, profilage) sans modifier le code source.
    - **Méthode Modèle** (Implicite dans `DataCleaner`) : La méthode `clean()` définit le squelette de l'opération de nettoyage, appelant des étapes spécifiques dans l'ordre.

3.  **Documentation complète** :
    - Ce README fournit des instructions d'installation, d'utilisation et des détails architecturaux.
    - Le code est documenté avec des docstrings.

4.  **Tests unitaires** :
    - Tests situés dans le répertoire `tests/`.
    - Couvre la logique de nettoyage des données et l'orchestration du pipeline.
    - Exécuter via `python -m unittest discover tests`.

## 📂 Structure du Projet

```
.
├── ds_toolkit/                # Package Python Main
│   ├── __init__.py            # Exporte les classes clés
│   ├── cleaning.py            # Module de Nettoyage de Données (DataCleaner)
│   ├── pipeline.py            # Module Pipeline ML (Loader, Splitter, Scaler, Model)
│   ├── cross_validation.py    # Stratégies de Validation Croisée
│   ├── validation.py          # Framework de Validation de Données
│   ├── facade.py              # Point d'Entrée Principal (Façade)
│   └── utils.py               # Utilitaires & Décorateurs
├── tests/                     # Suite de Tests Unitaires
│   ├── test_cleaning.py
│   └── test_pipeline.py
├── exercise_*.py              # Scripts d'exercices originaux (pour référence)
├── setup.py                   # Fichier d'installation du package
└── README.md                  # Documentation du Projet
```

## 🚀 Installation

Pour installer le package en mode éditable (recommandé pour le développement) :

```bash
pip install -e .
```

## 💻 Exemples d'Utilisation

### 1. Le "Bouton Facile" (Modèle Façade)

Le moyen le plus simple d'exécuter une analyse complète est d'utiliser la Façade :

```python
from ds_toolkit.facade import DataSciencePackage

# Initialiser
pkg = DataSciencePackage(filepath='customer_churn.csv', target_col='Churn')

# Tout exécuter : charger -> nettoyer -> entraîner -> évaluer
pkg.run_full_workflow()
```

### 2. Construction Personnalisée de Pipeline

Pour plus de contrôle, vous pouvez composer des composants individuels :

```python
from ds_toolkit.cleaning import DataCleaner
from ds_toolkit.pipeline import MLPipeline, DataLoader, DataSplitter, Scaler, ModelHandler

# 1. Nettoyer les Données
cleaner = DataCleaner('raw_data.csv')
cleaner.clean()
cleaner.save_data('clean_data.csv')

# 2. Construire le Pipeline
pipeline = MLPipeline(
    loader=DataLoader('clean_data.csv', target_column='target'),
    splitter=DataSplitter(test_size=0.2),
    scaler=Scaler(),
    model_handler=ModelHandler(n_estimators=200)
)

# 3. Exécuter
pipeline.run()
```

### 3. Utilisation des Décorateurs

```python
from ds_toolkit.utils import timing_decorator

@timing_decorator
def heavy_computation():
    # ... code ...
    pass
```

## 🧪 Exécution des Tests

Exécutez la suite de tests pour vous assurer que tout fonctionne :

```bash
python -m unittest discover tests
```

## 📊 Détails de Conception

### Nettoyage de Données (`cleaning.py`)

Encapsule toute la logique de nettoyage. Des méthodes comme `remove_duplicates` et `handle_missing_values` retournent `self` pour permettre le chaînage de méthodes (style Interface Fluide).

### Pipeline ML (`pipeline.py`)

Suit les principes **SOLID**. Le `MLPipeline` dépend d'abstractions (typage canard en Python) plutôt que d'implémentations concrètes, ce qui vous permet d'échanger facilement des éléments comme le modèle ou le scaler.

### Validation (`validation.py`)

Un framework extensible où vous pouvez ajouter de nouvelles classes `ValidationRule` (Principe Ouvert/Fermé) sans modifier le validateur principal.
