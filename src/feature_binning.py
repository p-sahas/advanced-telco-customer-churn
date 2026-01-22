import logging
import pandas as pd
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureBinningStrategy(ABC):
    @abstractmethod
    def bin(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        pass

class TenureBinningStrategy(FeatureBinningStrategy):
    def bin(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
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


