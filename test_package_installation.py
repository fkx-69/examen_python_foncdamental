import sys
import pandas as pd
import os
from ds_toolkit.cleaning import DataCleaner
from ds_toolkit.pipeline import MLPipeline, DataLoader, DataSplitter, Scaler, ModelHandler
from ds_toolkit.utils import timing_decorator
from ds_toolkit.facade import DataSciencePackage

# Setup dummy data for testing if titanic.csv doesn't exist or just to be safe
if not os.path.exists('titanic.csv'):
    print("Creating dummy titanic.csv...")
    df = pd.DataFrame({
        'Survived': [0, 1, 1, 0, 1]*20,
        'Pclass': [3, 1, 3, 2, 1]*20,
        'Age': [22.0, 38.0, 26.0, 35.0, None]*20,
        'Fare': [7.25, 71.28, 7.92, 53.1, 8.05]*20
    })
    df.to_csv('titanic.csv', index=False)

print("--- Test 1: Decorators ---")
@timing_decorator
def sleep_test():
    pass
sleep_test()
print("Decorator test passed.\n")

print("--- Test 2: Data Cleaning ---")
try:
    cleaner = DataCleaner('titanic.csv')
    cleaner.load_data()
    print("Data initialized.")
    cleaner.clean()
    print("Data cleaned.")

    # Additional step for the test: Ensure only numeric columns are kept for the ML pipeline
    # because the generic MLPipeline might not handle automatic encoding of complex strings (Name, Ticket)
    # and Scikit-Learn models need numbers.
    numeric_df = cleaner.df.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
    # Ensure target is kept if it was categorical (though Survived is usually int)
    if 'Survived' not in numeric_df.columns and 'Survived' in cleaner.df.columns:
         numeric_df['Survived'] = cleaner.df['Survived']
    
    cleaner.df = numeric_df
    cleaner.save_data('clean_titanic_test.csv')
    print("Cleaned data saved.")
except Exception as e:
    print(f"Cleaning failed: {e}")
    sys.exit(1)

print("\n--- Test 3: ML Pipeline Manual Construction ---")
try:
    # Defining a simple pipeline
    loader = DataLoader('clean_titanic_test.csv', target_column='Survived')
    splitter = DataSplitter(test_size=0.2, random_state=42)
    scaler = Scaler()
    model = ModelHandler() # defaulted to RandomForest
    
    pipeline = MLPipeline(loader, splitter, scaler, model)
    pipeline.run()
    print("Manual pipeline execution successful.")
except Exception as e:
    print(f"Pipeline failed: {e}")
    sys.exit(1)

print("\n--- Test 4: Facade Pattern ---")
try:
    # Using the facade for a one-liner execution
    # We use the already cleaned numeric data because the simple Facade implementation
    # doesn't handle complex string encoding automatically.
    facade = DataSciencePackage(filepath='clean_titanic_test.csv', target_col='Survived')
    facade.run_full_workflow()
    print("Facade execution successful.")
except Exception as e:
    print(f"Facade failed: {e}")
    sys.exit(1)

print("\n--- ALL TESTS PASSED SUCCESSFULLY ---")
