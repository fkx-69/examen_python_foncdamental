# Object-Oriented Data Science Toolkit (ds_toolkit)

**Course:** Master of Data Science - Object-Oriented Programming  
**Assignment:** Final Project & Take-Home Assignment

## Project Overview

This project refactors procedural data science code into a robust, modular, and reusable object-oriented package (`ds_toolkit`). It demonstrates the application of advanced software engineering principles to data science workflows, including design patterns, unit testing, and solid package structure.

## 📋 Features & Requirements Fulfillment

This submission addresses all requirements of the Take-Home Assignment:

1.  **Complete Package Structure**:
    - Source code organized in `ds_toolkit/` directory.
    - `setup.py` included for installation (`pip install -e .`).
    - Clear separation of concerns (Cleaning, Pipeline, Validation).

2.  **Design Patterns Implemented**:
    - **Facade Pattern** (`ds_toolkit/facade.py`): The `DataSciencePackage` class provides a simplified, unified interface for the entire workflow (Cleaning -> Modeling), hiding complexity from the user.
    - **Strategy Pattern** (`ds_toolkit/cross_validation.py`): `CrossValidationStrategy` allows dynamic switching between validation algorithms (e.g., `KFold`, `StratifiedKFold`) without modifying the context.
    - **Decorator Pattern** (`ds_toolkit/utils.py`): `logging_decorator` and `timing_decorator` extend function behavior (logging, profiling) without modifying the source code.
    - **Template Method** (Implicit in `DataCleaner`): The `clean()` method defines the skeleton of the cleaning operation, calling specific steps in order.

3.  **Comprehensive Documentation**:
    - This README provides installation, usage, and architectural details.
    - Code is documented with docstrings.

4.  **Unit Testing**:
    - Tests located in `tests/` directory.
    - Covers data cleaning logic and pipeline orchestration.
    - Run via `python -m unittest discover tests`.

## 📂 Project Structure

```
.
├── ds_toolkit/                # Main Python Package
│   ├── __init__.py            # Exports key classes
│   ├── cleaning.py            # Data Cleaning Module (DataCleaner)
│   ├── pipeline.py            # ML Pipeline Module (Loader, Splitter, Scaler, Model)
│   ├── cross_validation.py    # Cross-Validation Strategies
│   ├── validation.py          # Data Validation Framework
│   ├── facade.py              # Main Entry Point (Facade)
│   └── utils.py               # Utilities & Decorators
├── tests/                     # Unit Test Suite
│   ├── test_cleaning.py
│   └── test_pipeline.py
├── exercise_*.py              # Original exercise scripts (for reference)
├── setup.py                   # Package installation file
└── README.md                  # Project Documentation
```

## 🚀 Installation

To install the package in editable mode (recommended for development):

```bash
pip install -e .
```

## 💻 Usage Examples

### 1. The "Easy Button" (Facade Pattern)

The simplest way to run a full analysis is using the Facade:

```python
from ds_toolkit.facade import DataSciencePackage

# Initialize
pkg = DataSciencePackage(filepath='customer_churn.csv', target_col='Churn')

# Run everything: load -> clean -> train -> evaluate
pkg.run_full_workflow()
```

### 2. Custom Pipeline Construction

For more control, you can compose individual components:

```python
from ds_toolkit.cleaning import DataCleaner
from ds_toolkit.pipeline import MLPipeline, DataLoader, DataSplitter, Scaler, ModelHandler

# 1. Clean Data
cleaner = DataCleaner('raw_data.csv')
cleaner.clean()
cleaner.save_data('clean_data.csv')

# 2. Build Pipeline
pipeline = MLPipeline(
    loader=DataLoader('clean_data.csv', target_column='target'),
    splitter=DataSplitter(test_size=0.2),
    scaler=Scaler(),
    model_handler=ModelHandler(n_estimators=200)
)

# 3. Execute
pipeline.run()
```

### 3. Using Decorators

```python
from ds_toolkit.utils import timing_decorator

@timing_decorator
def heavy_computation():
    # ... code ...
    pass
```

## 🧪 Running Tests

Execute the test suite to ensure everything is working:

```bash
python -m unittest discover tests
```

## 📊 Design Details

### Data Cleaning (`cleaning.py`)

Encapsulates all cleaning logic. Methods like `remove_duplicates` and `handle_missing_values` return `self` to allow method chaining (Fluent Interface style).

### ML Pipeline (`pipeline.py`)

Follows the **SOLID** principles. The `MLPipeline` depends on abstractions (duck typing in Python) rather than concrete implementations, allowing you to swap out limits like the model or scaler easily.

### Validation (`validation.py`)

An extensible framework where you can add new `ValidationRule` classes (Open/Closed Principle) without modifying the main validator.
