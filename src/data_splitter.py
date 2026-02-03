import logging
from enum import Enum
from abc import ABC, abstractmethod
from typing import Tuple
from pyspark.sql import DataFrame

logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')

class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df: DataFrame, target_column: str = None) -> Tuple[DataFrame, DataFrame]:
        """
        Split DataFrame into train and test sets.
        target_column is kept for API compatibility but not always needed for randomSplit.
        """
        pass


class SplitType(str, Enum):
    SIMPLE = 'simple' 
    STRATIFIED = 'stratified'

class SimpleTrainTestSplitStratergy(DataSplittingStrategy):
    def __init__(self, test_size: float = 0.2, seed: int = 42):
        self.test_size = test_size
        self.seed = seed

    def split_data(self, df: DataFrame, target_column: str = None) -> Tuple[DataFrame, DataFrame]:
        logging.info(f"Splitting data with test_size={self.test_size}, seed={self.seed}")
        train_df, test_df = df.randomSplit([1.0 - self.test_size, self.test_size], seed=self.seed)
        return train_df, test_df