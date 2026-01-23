import logging
import pandas as pd
import os
import json
from enum import Enum
from typing import Dict, List
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEncodingStrategy(ABC):
    @abstractmethod
    def encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        pass

class VariableType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


class OneHotEncodingStrategy(FeatureEncodingStrategy):
    def encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        logging.info(f"Applying One-Hot Encoding to the {column} column.")
        one_hot = pd.get_dummies(df[column], prefix=column, drop_first=False, dtype=int)
        df = pd.concat([df, one_hot], axis=1)
        df.drop(column, axis=1, inplace=True)
        return df


class LabelEncodingStrategy(FeatureEncodingStrategy):
    def encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        from sklearn.preprocessing import LabelEncoder
        logging.info(f"Applying Label Encoding to the {column} column.")
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        return df

