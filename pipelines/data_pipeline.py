import os
import sys
import logging
import pandas as pd
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_ingestion import DataIngestorCSV
from handle_missing_values import TotalChargesMissingValueHandler
from outlier_detection import OutlierDetector, IQROutlierDetection

def data_pipeline(file_path: str) -> pd.DataFrame:
    # Step 1: Data Ingestion
    logger.info("Starting data ingestion.")
    ingestor = DataIngestorCSV()
    df = ingestor.ingest(file_path)
    logger.info("Data ingestion completed.")

    # Step 2: Handle Missing Values
    logger.info("Handling missing values.")
    missing_value_handler = TotalChargesMissingValueHandler()
    df = missing_value_handler.handle(df)
    logger.info("Missing values handled.")

    # Step 3: Outlier Detection and Handling
    logger.info("Detecting and handling outliers.")
    outlier_detector = OutlierDetector(IQROutlierDetection())
    selected_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df = outlier_detector.handle_outliers(df, selected_columns, method='remove')
    logger.info("Outliers handled.")

    # logger.info("Data pipeline completed.")
    # return df