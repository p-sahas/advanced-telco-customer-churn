import joblib, os
from typing import Any, Dict
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

class BaseModelBuilder(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.params = kwargs

    @abstractmethod
    def build_model(self):
        pass

    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("Model has not been built yet.")

        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")

        self.model = joblib.load(filepath)

class LogisticRegressionModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {
            'penalty': 'l2',
            'C': 0.1,  # Inverse regularization strength
            'solver': 'saga', # liblinear for L1, saga for elasticnet
            'max_iter': 100,
            'random_state': 42
        }
        default_params.update(kwargs)
        super().__init__('LogisticRegression', **default_params)

    def build_model(self):
        self.model = LogisticRegression(**self.params)
        return self.model
    
class DecisionTreeModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {
            'criterion': 'gini',
            'max_depth': None,
            'max_features': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'ccp_alpha': np.float64(0.00041608656377458515),
            'random_state': 42
        }
        default_params.update(kwargs)
        super().__init__('DecisionTree', **default_params)

    def build_model(self):
        self.model = DecisionTreeClassifier(**self.params)
        return self.model

class XGBoostModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {
            'n_estimators': 100,
            'max_depth': 9,
            'learning_rate': 0.05,
            'colsample_bytree': 0.6,
            'gamma': 0.2,
            'random_state': 42
        }  
        default_params.update(kwargs)
        super().__init__('XGBoost', **default_params)

    def build_model(self):
        self.model = XGBClassifier(**self.params)
        return self.model