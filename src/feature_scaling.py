import logging
import pandas as pd
import numpy as np
from enum import Enum
from typing import List
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')

class FeatureScalingStrategy(ABC):
    @abstractmethod
    def scale(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        pass

class StandardScalingStrategy(FeatureScalingStrategy):
    def __init__(self):
        self.scaler = StandardScaler()

    def scale(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        logging.info(f"Applying Standard Scaling to columns: {columns}")
        df[columns] = self.scaler.fit_transform(df[columns])
        return df
