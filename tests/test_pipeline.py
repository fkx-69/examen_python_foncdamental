import unittest
import pandas as pd
import numpy as np
import os
from ds_toolkit.pipeline import MLPipeline, DataLoader, DataSplitter, Scaler, ModelHandler

class TestMLPipeline(unittest.TestCase):
    
    def setUp(self):
        self.filename = "test_pipeline_data.csv"
        # Create synthetic data for classification
        X = np.random.rand(20, 4)
        y = np.random.choice([0, 1], 20)
        df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])
        df['target'] = y
        df.to_csv(self.filename, index=False)

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_dataloader(self):
        loader = DataLoader(self.filename, 'target')
        X, y = loader.load()
        self.assertEqual(X.shape, (20, 4))
        self.assertEqual(y.shape, (20,))

    def test_datasplitter(self):
        X = pd.DataFrame(np.random.rand(10, 2))
        y = pd.Series(np.random.randint(0, 2, 10))
        splitter = DataSplitter(test_size=0.2)
        X_train, X_test, y_train, y_test = splitter.split(X, y)
        self.assertEqual(len(X_test), 2)
        self.assertEqual(len(X_train), 8)
        
    def test_scaler(self):
        X_train = pd.DataFrame([[10], [20], [30]])
        scaler = Scaler()
        X_scaled = scaler.fit_transform(X_train)
        self.assertAlmostEqual(X_scaled.mean(), 0, places=1)
        
    def test_model_handler_and_pipeline(self):
        loader = DataLoader(self.filename, 'target')
        splitter = DataSplitter(test_size=0.5)
        scaler = Scaler()
        model = ModelHandler(n_estimators=10)
        
        pipeline = MLPipeline(loader, splitter, scaler, model)
        # We just check if it runs without error and returns a string report
        report = pipeline.run()
        self.assertIsInstance(report, str)
        self.assertIn("accuracy", report)

if __name__ == '__main__':
    unittest.main()
