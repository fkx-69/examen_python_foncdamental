"""
Facade for the Data Science Package.
Integrates all components into a unified interface (Facade Pattern).
"""

from .cleaning import DataCleaner
from .pipeline import MLPipeline, DataLoader, DataSplitter, Scaler, ModelHandler
from .utils import timing_decorator

class DataSciencePackage:
    """Unified Facade for the Data Science Project."""
    
    def __init__(self, filepath: str, target_col: str):
        self.filepath = filepath
        self.target_col = target_col
        self.cleaner = DataCleaner(filepath)
        self.pipeline = None
        
    @timing_decorator
    def run_full_workflow(self, saved_clean_path: str = "temp_cleaned_data.csv"):
        print("=== Launching Data Science Workflow ===")
        
        # 1. Cleaning
        print("\n1. Data Cleaning...")
        try:
            self.cleaner.load_data()
            self.cleaner.clean()
            # Save properly to communicate with the next step
            self.cleaner.save_data(saved_clean_path)
            print(f"Intermediate data saved to {saved_clean_path}")
        except Exception as e:
            print(f"Cleaning error: {e}")
            return

        # 2. ML Pipeline
        print("\n2. Executing ML Pipeline...")
        loader = DataLoader(saved_clean_path, self.target_col)
        splitter = DataSplitter()
        scaler = Scaler()
        model = ModelHandler()
        
        self.pipeline = MLPipeline(loader, splitter, scaler, model)
        self.pipeline.run()
        
        print("\n=== Workflow Completed ===")
