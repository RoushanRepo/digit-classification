import unittest
from sklearn.linear_model import LogisticRegression
import joblib
import os

class TestLogisticRegressionModel(unittest.TestCase):
    
    def setUp(self):
        
        
        self.solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

    def test_loaded_model_is_logistic_regression(self):
        for solver in self.solvers:
            model_name = f'm22aie243_lr_{solver}.joblib'
            model_path = model_name
            loaded_model = joblib.load(model_path)

            self.assertIsInstance(loaded_model, LogisticRegression)
        
    def test_solver_name_match(self):
        for solver in self.solvers:
            model_name = f'm22aie243_lr_{solver}.joblib'
            model_path = model_name

            loaded_model = joblib.load(model_path)

            self.assertEqual(solver, loaded_model.solver)

if __name__ == '__main__':
    unittest.main()
