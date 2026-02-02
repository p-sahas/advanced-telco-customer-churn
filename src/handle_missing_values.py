import logging
import pandas as pd
from enum import Enum
from typing import Optional
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.ml.feature import Imputer
from spark_session import get_or_create_spark_session


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class MissingValueHandlingStrategy(ABC):
    """Abstract base class for missing value handling strategies."""
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_or_create_spark_session()
    
    @abstractmethod
    def handle(self, df: DataFrame) -> DataFrame:
        """Handle missing values in the DataFrame."""
        pass

class TotalChargesMissingValueHandler(MissingValueHandlingStrategy):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Handling missing values in TotalCharges column.")
        # Convert TotalCharges to numeric, coercing errors to NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Fill missing TotalCharges with 0 (assuming new customers)
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        return df




