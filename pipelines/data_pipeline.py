import os
import sys
import logging
import pandas as pd
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_ingestion import DataIngestorCSV
from handle_missing_values import TotalChargesMissingValueHandler
from outlier_detection import OutlierDetector, IQROutlierDetection
from feature_binning import TenureBinningStrategy
from feature_encoding import OneHotEncodingStrategy, LabelEncodingStrategy
from feature_scaling import StandardScalingStrategy
from data_splitter import SimpleTrainTestSplitStratergy

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_data_paths, get_columns, get_missing_values_config, get_outlier_config, get_binning_config, get_encoding_config, get_scaling_config, get_splitting_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


scalling_config = get_scaling_config()
data_paths = get_data_paths()


def data_pipeline() -> pd.DataFrame:
    data_paths = get_data_paths()
    # Step 1: Data Ingestion
    logger.info("Starting data ingestion.")
    artifacts_dir = os.path.join(os.path.dirname(
        __file__), '..', data_paths['data_artifacts_dir'])
    x_train_path = os.path.join('artifacts', 'data', 'X_train.csv')
    x_test_path = os.path.join('artifacts', 'data', 'X_test.csv')
    y_train_path = os.path.join('artifacts', 'data', 'Y_train.csv')
    y_test_path = os.path.join('artifacts', 'data', 'Y_test.csv')

    if os.path.exists(x_train_path) and \
            os.path.exists(x_test_path) and \
            os.path.exists(y_train_path) and \
            os.path.exists(y_test_path):

        X_train = pd.read_csv(x_train_path)
        X_test = pd.read_csv(x_test_path)
        Y_train = pd.read_csv(y_train_path)
        Y_test = pd.read_csv(y_test_path)

    os.makedirs(data_paths['data_artifacts_dir'], exist_ok=True)
    if not os.path.exists('temp_imputed.csv'):
        ingestor = DataIngestorCSV()
        df = ingestor.ingest(data_paths['raw_data'])
        print(f"loaded data shape : {df.shape}")
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
    df = outlier_detector.handle_outliers(
        df, selected_columns, method='remove')
    logger.info("Outliers handled.")

    # Step 4: Feature Binning
    logger.info("Binning features.")
    tenure_binner = TenureBinningStrategy()
    df = tenure_binner.bin(df, 'tenure')
    logger.info("Feature binning completed.")

    # Step 5: Feature Scaling
    logger.info("Scaling features.")
    columns_config = get_columns()
    numeric_columns = columns_config['numeric_columns']
    standard_scaler = StandardScalingStrategy()
    df = standard_scaler.scale(df, numeric_columns)
    logger.info("Feature scaling completed.")

    # Step 6: Feature Encoding
    logger.info("Encoding features.")
    categorical_columns = columns_config['categorical_columns']
    for col in categorical_columns:
        if col == 'Churn':
            df = LabelEncodingStrategy().encode(df, col)
        else:
            df = OneHotEncodingStrategy().encode(df, col)
    logger.info("Feature encoding completed.")

    # Step 7: Post Processing
    drop_columns = columns_config['drop_columns']
    df = df.drop(drop_columns, axis=1)
    print(f'data after post processing : \n {df.head()}')

    # Step 8: Data Splitting
    logger.info("Splitting data into training and testing sets.")
    splitter = SimpleTrainTestSplitStratergy(test_size=0.2)
    X_train, X_test, Y_train, Y_test = splitter.split_data(df, 'Churn')
    logger.info("Data splitting completed.")

    # Create directories and save splits
    os.makedirs('artifacts/data', exist_ok=True)
    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    Y_train.to_csv(y_train_path, index=False)
    Y_test.to_csv(y_test_path, index=False)

    print(f'X train size : {X_train.shape}')
    print(f'X test size : {X_test.shape}')
    print(f'Y train size : {Y_train.shape}')
    print(f'Y test size : {Y_test.shape}')
    logger.info("Data pipeline completed.")
    # return df


data_pipeline()
