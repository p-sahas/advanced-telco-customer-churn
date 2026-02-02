import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Union
import pandas as pd  # Keep pandas import for educational purposes
from pyspark.sql import DataFrame, SparkSession
from spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataIngestor(ABC):
    # Abstract base class for data ingestion supporting both pandas and PySpark.
    
    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_or_create_spark_session()
    
    @abstractmethod
    def ingest(self, file_path_or_link: str) -> DataFrame:
        pass


class DataIngestorCSV(DataIngestor):
    """CSV data ingestion implementation."""
    
    def ingest(self, file_path_or_link: str, **options) -> DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA INGESTION - CSV (PySpark)")
        logger.info(f"{'='*60}")
        logger.info(f"Starting CSV data ingestion from: {file_path_or_link}")
        
        try:
            # Default CSV options
            csv_options = {
                        "header": "true",
                        "inferSchema": "true",
                        "ignoreLeadingWhiteSpace": "true",
                        "ignoreTrailingWhiteSpace": "true",
                        "nullValue": "",
                        "nanValue": "NaN",
                        "escape": '"',
                        "quote": '"'
                        }
            csv_options.update(options)
            
            # PANDAS CODE
            # df = pd.read_csv(file_path_or_link)
            
            df = self.spark.read.options(**csv_options).csv(file_path_or_link)
            
        except Exception as e:
            logger.error(f" Failed to load CSV data from {file_path_or_link}: {str(e)}")
            logger.info(f"{'='*60}\n")
            raise


class DataIngestorExcel(DataIngestor):
    # Excel data ingestion implementation.
    
    def ingest(self, file_path_or_link: str, sheet_name: Optional[str] = None, **options) -> DataFrame:
        
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA INGESTION - EXCEL (PySpark)")
        logger.info(f"{'='*60}")
        logger.info(f"Starting Excel data ingestion from: {file_path_or_link}")
        
        try:
            
            logger.info(" Note: Using pandas for Excel reading, then converting to PySpark")
            
            # PANDAS CODE
            # df = pd.read_excel(file_path_or_link)
            
            pandas_df = pd.read_excel(file_path_or_link)
            df = self.spark.createDataFrame(pandas_df)
            
        except Exception as e:
            logger.error(f" Failed to load Excel data from {file_path_or_link}: {str(e)}")
            logger.info(f"{'='*60}\n")
            raise


class DataIngestorParquet(DataIngestor):
    # PySpark Parquet data ingestion implementation 
    
    def ingest(self, file_path_or_link: str, **options) -> DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA INGESTION - PARQUET (PySpark)")
        logger.info(f"{'='*60}")
        logger.info(f"Starting Parquet data ingestion from: {file_path_or_link}")
        
        try:
            # Read Parquet file(s)
            df = self.spark.read.options(**csv_options).parquet(file_path_or_link)
            
        except Exception as e:
            logger.error(f" Failed to load Parquet data from {file_path_or_link}: {str(e)}")
            logger.info(f"{'='*60}\n")
            raise


class DataIngestorFactory:
    # Factory class to create appropriate data ingestor based on file type.
    
    @staticmethod
    def get_ingestor(file_path: str, spark: Optional[SparkSession] = None) -> DataIngestor:
       
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            return DataIngestorCSV(spark)
        elif file_extension in ['.xlsx', '.xls']:
            return DataIngestorExcel(spark)
        elif file_extension == '.parquet':
            return DataIngestorParquet(spark)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")