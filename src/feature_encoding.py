import logging
import os
import json
from enum import Enum
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, IndexToString
from pyspark.ml import Pipeline
from spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEncodingStrategy(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        # Initialize with SparkSession.
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def encode(self, df: DataFrame, column: str) -> DataFrame:
        pass

class VariableType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


class OneHotEncodingStrategy(FeatureEncodingStrategy):
    def encode(self, df: DataFrame, column: str) -> DataFrame:
        logging.info(f"Applying PySpark One-Hot Encoding to {column}")
        
        # 1. String Indexing
        indexer = StringIndexer(inputCol=column, outputCol=f"{column}_index", handleInvalid="keep", stringOrderType="alphabetAsc")
        
        # 2. One Hot Encoding
        encoder = OneHotEncoder(inputCols=[f"{column}_index"], outputCols=[f"{column}_encoded"])
        
        # 3. Pipeline
        pipeline = Pipeline(stages=[indexer, encoder])
        model = pipeline.fit(df)
        df_encoded = model.transform(df)
        
        # Return with original column dropped
        # and intermediate index column dropped.
        return df_encoded.drop(column, f"{column}_index")


class LabelEncodingStrategy(FeatureEncodingStrategy):
    def encode(self, df: DataFrame, column: str) -> DataFrame:
        logging.info(f"Applying PySpark Label Encoding to {column}")
        indexer = StringIndexer(inputCol=column, outputCol=f"{column}_index", handleInvalid="keep", stringOrderType="alphabetAsc")
        model = indexer.fit(df)
        df_encoded = model.transform(df)
        
        # Replace original column with indexed column
        # Rename new index column to original name after dropping original
        return df_encoded.drop(column).withColumnRenamed(f"{column}_index", column)
