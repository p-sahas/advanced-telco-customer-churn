import joblib, os
from typing import Any, Dict
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score

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

