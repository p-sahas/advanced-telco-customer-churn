import os
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
from pathlib import Path

from utils.config import get_data_paths, get_columns, get_inference_config
from utils.mlflow_utils import MLflowTracker
from src.feature_encoding import OneHotEncodingStrategy
from src.feature_scaling import StandardScalerStrategy
from src.handle_missing_values import MeanImputationStrategy, ModeImputationStrategy

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelInference:
    """
    Class for loading trained models and making predictions on new data.
    Handles preprocessing of input data to match training data format.
    """

    def __init__(self, model_path: Optional[str] = None, use_mlflow: bool = True):
        """
        Initialize the ModelInference class.

        Args:
            model_path: Path to the saved model file. If None, loads from MLflow registry.
            use_mlflow: Whether to use MLflow for model loading and logging.
        """
        self.model_path = model_path
        self.use_mlflow = use_mlflow
        self.model = None
        self.preprocessors = {}
        self.feature_columns = None

        if use_mlflow:
            self.mlflow_tracker = MLflowTracker()

        self._load_model()
        self._load_preprocessors()

    def _load_model(self):
        """Load the trained model from file or MLflow registry."""
        try:
            if self.model_path and os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
            elif self.use_mlflow:
                self.model = self.mlflow_tracker.load_model_from_registry(
                    stage='Production')
                if self.model is None:
                    self.model = self.mlflow_tracker.load_model_from_registry(
                        stage='Staging')
                if self.model is None:
                    logger.warning(
                        "No model found in Production or Staging. Loading latest version.")
                    self.model = self.mlflow_tracker.load_model_from_registry()
            else:
                raise FileNotFoundError(
                    "Model path not provided and MLflow not enabled.")

            if self.model is None:
                raise ValueError("Failed to load model.")

            logger.info("Model loaded successfully.")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _load_preprocessors(self):
        """Load saved preprocessors for data transformation."""
        try:
            artifacts_dir = get_data_paths().get('artifacts_dir', 'artifacts')

            # Load encoders
            encoder_path = os.path.join(artifacts_dir, 'encoders.joblib')
            if os.path.exists(encoder_path):
                self.preprocessors['encoder'] = joblib.load(encoder_path)
                logger.info("Loaded feature encoder.")

            # Load scaler
            scaler_path = os.path.join(artifacts_dir, 'scaler.joblib')
            if os.path.exists(scaler_path):
                self.preprocessors['scaler'] = joblib.load(scaler_path)
                logger.info("Loaded feature scaler.")

            # Load imputers
            numeric_imputer_path = os.path.join(
                artifacts_dir, 'numeric_imputer.joblib')
            if os.path.exists(numeric_imputer_path):
                self.preprocessors['numeric_imputer'] = joblib.load(
                    numeric_imputer_path)

            categorical_imputer_path = os.path.join(
                artifacts_dir, 'categorical_imputer.joblib')
            if os.path.exists(categorical_imputer_path):
                self.preprocessors['categorical_imputer'] = joblib.load(
                    categorical_imputer_path)

            # Load feature columns order
            columns_path = os.path.join(
                artifacts_dir, 'feature_columns.joblib')
            if os.path.exists(columns_path):
                self.feature_columns = joblib.load(columns_path)
                logger.info(
                    f"Loaded feature columns order: {len(self.feature_columns)} features.")

        except Exception as e:
            logger.warning(
                f"Error loading preprocessors: {e}. Using default preprocessing.")

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data to match training data format.

        Args:
            data: Input DataFrame to preprocess.

        Returns:
            Preprocessed DataFrame.
        """
        try:
            # Make a copy to avoid modifying original data
            processed_data = data.copy()

            # Get column configurations
            columns_config = get_columns()
            drop_columns = columns_config.get('drop_columns', [])
            categorical_columns = columns_config.get('categorical_columns', [])
            numeric_columns = columns_config.get('numeric_columns', [])

            # Drop unnecessary columns
            for col in drop_columns:
                if col in processed_data.columns:
                    processed_data.drop(col, axis=1, inplace=True)

            # Handle missing values
            if 'numeric_imputer' in self.preprocessors:
                for col in numeric_columns:
                    if col in processed_data.columns:
                        processed_data[col] = self.preprocessors['numeric_imputer'].transform(
                            processed_data[col].values.reshape(-1, 1)
                        ).ravel()

            if 'categorical_imputer' in self.preprocessors:
                for col in categorical_columns:
                    if col in processed_data.columns:
                        processed_data[col] = self.preprocessors['categorical_imputer'].transform(
                            processed_data[col].values.reshape(-1, 1)
                        ).ravel()

            # Encode categorical features
            if 'encoder' in self.preprocessors:
                for col in categorical_columns:
                    if col in processed_data.columns:
                        processed_data = self.preprocessors['encoder'].encode(
                            processed_data, col)

            # Scale numeric features
            if 'scaler' in self.preprocessors:
                for col in numeric_columns:
                    if col in processed_data.columns:
                        processed_data[col] = self.preprocessors['scaler'].transform(
                            processed_data[col].values.reshape(-1, 1)
                        ).ravel()

            # Ensure correct column order
            if self.feature_columns:
                # Add missing columns with zeros
                for col in self.feature_columns:
                    if col not in processed_data.columns:
                        processed_data[col] = 0

                # Reorder columns
                processed_data = processed_data[self.feature_columns]

            logger.info(
                f"Data preprocessed successfully. Shape: {processed_data.shape}")
            return processed_data

        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise

    def predict(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> np.ndarray:
        """
        Make predictions on input data.

        Args:
            data: Input data as DataFrame, dict, or list of dicts.

        Returns:
            Prediction probabilities or classes.
        """
        try:
            # Convert input to DataFrame
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            elif isinstance(data, list):
                data = pd.DataFrame(data)

            # Preprocess data
            processed_data = self.preprocess_data(data)

            # Make predictions
            predictions = self.model.predict_proba(
                processed_data)[:, 1]  # Probability of churn

            # Log inference if using MLflow
            if self.use_mlflow:
                try:
                    self.mlflow_tracker.log_inference_metrics(
                        input_data=processed_data,
                        predictions=predictions
                    )
                except Exception as e:
                    logger.warning(f"Failed to log inference metrics: {e}")

            logger.info(f"Made predictions for {len(predictions)} samples.")
            return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

    def predict_classes(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> np.ndarray:
        """
        Make class predictions (0 or 1) on input data.

        Args:
            data: Input data as DataFrame, dict, or list of dicts.

        Returns:
            Binary predictions (0: No churn, 1: Churn).
        """
        probabilities = self.predict(data)
        return (probabilities >= 0.5).astype(int)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information.
        """
        return {
            'model_type': type(self.model).__name__,
            'feature_columns': self.feature_columns,
            'preprocessors': list(self.preprocessors.keys()),
            'model_path': self.model_path
        }


def load_inference_model(model_path: Optional[str] = None, use_mlflow: bool = True) -> ModelInference:
    """
    Convenience function to load a ModelInference instance.

    Args:
        model_path: Path to the saved model file.
        use_mlflow: Whether to use MLflow for model loading.

    Returns:
        ModelInference instance.
    """
    return ModelInference(model_path=model_path, use_mlflow=use_mlflow)


if __name__ == "__main__":
    # Example usage
    inference = ModelInference()

    # Example input data
    sample_data = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'No',
        'DeviceProtection': 'Yes',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 29.85,
        'TotalCharges': 358.15
    }

    # Make prediction
    prediction = inference.predict(sample_data)
    print(f"Churn probability: {prediction[0]:.4f}")

    # Get class prediction
    class_pred = inference.predict_classes(sample_data)
    print(f"Predicted class: {'Churn' if class_pred[0] == 1 else 'No Churn'}")
