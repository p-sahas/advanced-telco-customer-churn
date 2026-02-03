import logging
import pandas as pd
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType
from spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')

class OutlierDetectionStrategy(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        # Initialize with SparkSession.
        self.spark = spark or get_or_create_spark_session()
    @abstractmethod
    def detect_outliers(self, df: DataFrame, columns: List[str]) -> DataFrame:
        pass

class IQROutlierDetection(OutlierDetectionStrategy):
        
    def detect_outliers(self, df: DataFrame, selected_columns: List[str]) -> DataFrame:
        for col in columns:
            df[col] = df[col].astype(float)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            print(f"IQR for {col} : {IQR}")

            outliers[col] = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
            logging.info('Outliers detected using IQR Method.')
        return outliers
        
class OutlierDetector:
    def __init__(self, strategy):
        self._strategy = strategy

    def detect_outliers(self, df :DataFrame, selected_columns: List[str]) -> DataFrame:
        logger.info(f"Detecting outliers in {len(selected_columns)} columns")
        return self._strategy.detect_outliers(df, selected_columns)
    
    def handle_outliers(self, df: DataFrame, selected_columns: List[str], method='remove'):
        outliers = self.detect_outliers(df, selected_columns)

        outlier_count = outliers.sum(axis=1) # Getting for each row
        rows_to_remove = outlier_count >= 2
        return df[~rows_to_remove]
    


