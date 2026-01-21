import logging
import pandas as pd
from enum import Enum
from typing import Optimal
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class TotalChargesMissingValueHandler(MissingValueHandlingStrategy):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Handling missing values in TotalCharges column.")
        # Convert TotalCharges to numeric, coercing errors to NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Fill missing TotalCharges with 0 (assuming new customers)
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        return df




