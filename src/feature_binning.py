import logging
import pandas as pd
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Bucketizer
from spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureBinningStrategy(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        # Initialize with SparkSession.
        self.spark = spark or get_or_create_spark_session()
    
    @abstractmethod
    def bin_feature(self, df: DataFrame, column: str) -> DataFrame:
        pass

class TenureBinningStrategy(FeatureBinningStrategy):
    def __init__(self, bin_definitions: Dict[str, List[float]], spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.bin_definitions = bin_definitions
        logger.info(f"Initialized TenureBinningStrategy with bins: {self.bin_definitions}")
    
    def bin_feature(self, df: DataFrame, column: str) -> DataFrame:
        logger.info(f"Binning the {column} column into categories.")
        
        def tenure_group(t):
            if t <= 12:
                return 'New'
            elif t <= 48:
                return 'Established'
            else:
                return 'Loyal'
        
        df[f'{column}Category'] = df[column].apply(tenure_group)
        return df


