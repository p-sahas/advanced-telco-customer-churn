import logging
from enum import Enum
from typing import List
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml import Pipeline

logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')

class FeatureScalingStrategy(ABC):
    @abstractmethod
    def scale(self, df: DataFrame, columns: List[str]) -> DataFrame:
        pass

class StandardScalingStrategy(FeatureScalingStrategy):
    def __init__(self, output_col: str = "scaled_features"):
        self.output_col = output_col

    def scale(self, df: DataFrame, columns: List[str]) -> DataFrame:
        logging.info(f"Applying Standard Scaling to columns: {columns}")
        
        # 1. Assemble features into a vector
        # We need a temporary column name for the unscaled vector
        assembler = VectorAssembler(inputCols=columns, outputCol=f"{self.output_col}_unscaled")
        
        # 2. Scale the vector
        scaler = StandardScaler(inputCol=f"{self.output_col}_unscaled", outputCol=self.output_col,
                                withMean=True, withStd=True)
        
        # 3. Pipeline
        pipeline = Pipeline(stages=[assembler, scaler])
        model = pipeline.fit(df)
        df_scaled = model.transform(df)
        
        # Drop the intermediate unscaled vector
        return df_scaled.drop(f"{self.output_col}_unscaled")
