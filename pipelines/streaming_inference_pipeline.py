
import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Generator, Union
from datetime import datetime
import json
from pathlib import Path
import queue
import threading

# Add src and utils to path BEFORE importing project modules

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_inference import ModelInference
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_logging_config, get_data_paths

# Try to import MLflow utilities, make it optional
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from mlflow_utils import MLflowTracker
    MLFLOW_AVAILABLE = True
except ImportError:
    print("Warning: MLflow not available. Running without MLflow integration.")
    MLFLOW_AVAILABLE = False
    MLflowTracker = None


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StreamingInferencePipeline:
    """
    Pipeline for handling streaming inference requests.
    Supports batch processing, real-time inference, and data source integration.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 batch_size: int = 32,
                 max_queue_size: int = 1000,
                 enable_mlflow_logging: bool = True):
        """
        Initialize the streaming inference pipeline.

        Args:
            model_path: Path to the saved model file.
            batch_size: Number of samples to process in each batch.
            max_queue_size: Maximum size of the input queue.
            enable_mlflow_logging: Whether to log inference metrics to MLflow.
        """
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.enable_mlflow_logging = enable_mlflow_logging

        # Initialize model inference
        self.inference_model = ModelInference(
            model_path=model_path, use_mlflow=enable_mlflow_logging)

        # Initialize queues and threads
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        self.processing_thread = None
        self.is_running = False

        # Statistics
        self.stats = {
            'total_processed': 0,
            'total_batches': 0,
            'avg_processing_time': 0.0,
            'start_time': None,
            'errors': 0
        }

        # Setup logging
        self._setup_logging()

        logger.info("Streaming inference pipeline initialized.")

    def _setup_logging(self):
        """Setup logging configuration."""
        logging_config = get_logging_config()
        log_level = getattr(
            logging, logging_config.get('level', 'INFO').upper())
        logging.basicConfig(
            level=log_level,
            format=logging_config.get(
                'format', '%(asctime)s - %(levelname)s - %(message)s'),
            filename=logging_config.get('file', 'streaming_inference.log')
        )

    def start(self):
        """Start the streaming inference pipeline."""
        if self.is_running:
            logger.warning("Pipeline is already running.")
            return

        self.is_running = True
        self.stats['start_time'] = datetime.now()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_batches)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        logger.info("Streaming inference pipeline started.")

    def stop(self):
        """Stop the streaming inference pipeline."""
        if not self.is_running:
            logger.warning("Pipeline is not running.")
            return

        self.is_running = False

        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)

        # Log final statistics
        self._log_statistics()

        logger.info("Streaming inference pipeline stopped.")

    def _process_batches(self):
        """Main processing loop for handling batches of data."""
        while self.is_running:
            try:
                # Collect batch of data
                batch_data = []
                batch_start_time = time.time()

                # Wait for first item with timeout
                try:
                    first_item = self.input_queue.get(timeout=1.0)
                    batch_data.append(first_item)
                except queue.Empty:
                    continue

                # Get remaining items for batch (non-blocking)
                while len(batch_data) < self.batch_size and not self.input_queue.empty():
                    try:
                        item = self.input_queue.get_nowait()
                        batch_data.append(item)
                    except queue.Empty:
                        break

                if not batch_data:
                    continue

                # Process batch
                try:
                    predictions = self._process_batch(batch_data)

                    # Put results in output queue
                    for i, pred in enumerate(predictions):
                        result = {
                            'input_data': batch_data[i],
                            'prediction': float(pred),
                            'prediction_class': int(pred >= 0.5),
                            'timestamp': datetime.now().isoformat(),
                            'batch_id': self.stats['total_batches']
                        }
                        self.output_queue.put(result, timeout=1.0)

                    # Update statistics
                    processing_time = time.time() - batch_start_time
                    self.stats['total_processed'] += len(batch_data)
                    self.stats['total_batches'] += 1
                    self.stats['avg_processing_time'] = (
                        (self.stats['avg_processing_time'] * (self.stats['total_batches'] - 1) +
                         processing_time) / self.stats['total_batches']
                    )

                    logger.info(
                        f"Processed batch of {len(batch_data)} samples in {processing_time:.3f}s")

                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    self.stats['errors'] += 1

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                self.stats['errors'] += 1
                time.sleep(0.1)  # Brief pause on error

    def _process_batch(self, batch_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Process a batch of input data.

        Args:
            batch_data: List of input data dictionaries.

        Returns:
            Array of predictions.
        """
        # Convert to DataFrame
        batch_df = pd.DataFrame(batch_data)

        # Make predictions
        predictions = self.inference_model.predict(batch_df)

        return predictions

    def submit_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], timeout: float = 1.0):
        """
        Submit data for inference.

        Args:
            data: Single data point or list of data points.
            timeout: Timeout for queue operations.
        """
        if isinstance(data, dict):
            data = [data]

        for item in data:
            try:
                self.input_queue.put(item, timeout=timeout)
            except queue.Full:
                logger.warning("Input queue is full. Dropping data point.")
                break

    def get_results(self, timeout: float = 1.0, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Get inference results.

        Args:
            timeout: Timeout for getting results.
            max_results: Maximum number of results to return.

        Returns:
            List of result dictionaries.
        """
        results = []
        for _ in range(max_results):
            try:
                result = self.output_queue.get(timeout=timeout)
                results.append(result)
            except queue.Empty:
                break

        return results

    def _log_statistics(self):
        """Log pipeline statistics."""
        if self.stats['start_time']:
            runtime = (datetime.now() -
                       self.stats['start_time']).total_seconds()
            throughput = self.stats['total_processed'] / \
                runtime if runtime > 0 else 0

            logger.info("Pipeline Statistics:")
            logger.info(f"  Total processed: {self.stats['total_processed']}")
            logger.info(f"  Total batches: {self.stats['total_batches']}")
            logger.info(
                f"  Average processing time: {self.stats['avg_processing_time']:.3f}s")
            logger.info(f"  Throughput: {throughput:.2f} samples/second")
            logger.info(f"  Errors: {self.stats['errors']}")
            logger.info(f"  Runtime: {runtime:.2f}s")

    def get_statistics(self) -> Dict[str, Any]:
        """Get current pipeline statistics."""
        stats = self.stats.copy()
        if stats['start_time']:
            stats['runtime_seconds'] = (
                datetime.now() - stats['start_time']).total_seconds()
            stats['throughput'] = stats['total_processed'] / \
                stats['runtime_seconds'] if stats['runtime_seconds'] > 0 else 0
        return stats


class DataStreamSimulator:
    """
    Simulator for generating streaming data for testing the inference pipeline.
    """

    def __init__(self, data_file: Optional[str] = None, delay: float = 0.1):
        """
        Initialize the data stream simulator.

        Args:
            data_file: Path to CSV file with sample data.
            delay: Delay between sending data points (seconds).
        """
        self.delay = delay
        self.data_file = data_file or get_data_paths().get('raw_data')
        self.data = None

        if os.path.exists(self.data_file):
            self.data = pd.read_csv(self.data_file)
            # Remove target column if present
            if 'Churn' in self.data.columns:
                self.data = self.data.drop('Churn', axis=1)
            logger.info(
                f"Loaded {len(self.data)} samples from {self.data_file}")
        else:
            logger.warning(
                f"Data file {self.data_file} not found. Using synthetic data.")

    def generate_stream(self) -> Generator[Dict[str, Any], None, None]:
        """
        Generate a stream of data points.

        Yields:
            Dictionary representing a data point.
        """
        if self.data is not None:
            # Use real data
            for _, row in self.data.iterrows():
                yield row.to_dict()
                time.sleep(self.delay)
        else:
            # Generate synthetic data
            while True:
                yield self._generate_synthetic_sample()
                time.sleep(self.delay)

    def _generate_synthetic_sample(self) -> Dict[str, Any]:
        """Generate a synthetic data sample."""
        return {
            'gender': np.random.choice(['Male', 'Female']),
            'SeniorCitizen': np.random.choice([0, 1]),
            'Partner': np.random.choice(['Yes', 'No']),
            'Dependents': np.random.choice(['Yes', 'No']),
            'tenure': np.random.randint(1, 72),
            'PhoneService': np.random.choice(['Yes', 'No']),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service']),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No']),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service']),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service']),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service']),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service']),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service']),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service']),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year']),
            'PaperlessBilling': np.random.choice(['Yes', 'No']),
            'PaymentMethod': np.random.choice([
                'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
            ]),
            'MonthlyCharges': np.random.uniform(18.0, 118.0),
            'TotalCharges': np.random.uniform(18.0, 8684.0)
        }


def run_streaming_inference_demo(duration_seconds: int = 30):
    """
    Run a demonstration of the streaming inference pipeline.

    Args:
        duration_seconds: How long to run the demo.
    """
    # Initialize pipeline
    pipeline = StreamingInferencePipeline(batch_size=16)

    # Start pipeline
    pipeline.start()

    # Initialize data simulator
    simulator = DataStreamSimulator(delay=0.05)  # 20 samples per second

    # Start streaming
    start_time = time.time()
    stream_generator = simulator.generate_stream()

    logger.info("Starting streaming inference demo...")

    try:
        while time.time() - start_time < duration_seconds:
            # Get next data point
            data_point = next(stream_generator)

            # Submit for inference
            pipeline.submit_data(data_point)

            # Get results (non-blocking)
            results = pipeline.get_results(timeout=0.01, max_results=5)
            if results:
                logger.info(f"Received {len(results)} results")

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user.")
    finally:
        # Stop pipeline
        pipeline.stop()

        # Final statistics
        stats = pipeline.get_statistics()
        # Convert datetime objects to strings for JSON serialization
        if 'start_time' in stats and stats['start_time']:
            stats['start_time'] = stats['start_time'].isoformat()
        logger.info("Demo completed. Final statistics:")
        logger.info(json.dumps(stats, indent=2))


if __name__ == "__main__":
    # Run demo
    run_streaming_inference_demo(duration_seconds=10)
