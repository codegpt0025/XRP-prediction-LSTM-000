import yfinance as yf
import pandas as pd
from typing import List, Dict, Any
import time
import logging
import pandas_market_calendars as mcal
from datetime import datetime

class DataIntelligence:
    """
    A unified data intelligence system that handles data collection, cleaning,
    and feature extraction in a single, intelligent pipeline.
    """

    def __init__(self, api_sources: List[str] = ['yfinance']):
        """
        Initializes the DataIntelligence system.

        Args:
            api_sources (List[str], optional): A list of API sources to use for data collection.
                                              Defaults to ['yfinance'].
        """
        self.sources = self._discover_sources(api_sources)
        self.quality_scores: Dict[str, float] = {source: 1.0 for source in self.sources}
        self.nyse = mcal.get_calendar('NYSE')

    def is_market_open(self) -> bool:
        """
        Checks if the NYSE market is currently open.
        """
        now = datetime.utcnow()
        schedule = self.nyse.schedule(start_date=now.date(), end_date=now.date())
        if not schedule.empty:
            market_open = schedule.iloc[0]['market_open'].tz_convert('UTC')
            market_close = schedule.iloc[0]['market_close'].tz_convert('UTC')
            return market_open <= now <= market_close
        return False

    def _discover_sources(self, api_sources: List[str]) -> List[str]:
        """
        Discovers and ranks available data sources.
        """
        # In a real implementation, this would involve testing and scoring various APIs.
        return api_sources

    def collect(self, asset: str, lookback: str = '60d', interval='1h', max_retries: int = 3, backoff_factor: int = 2) -> pd.DataFrame:
        """
        Collects and cleans data for a given asset with a retry mechanism.
        """
        if not self.is_market_open():
            logging.warning("Market is closed. Using historical data.")

        for i in range(max_retries):
            try:
                data = yf.download(asset, period=lookback, interval=interval, timeout=30)
                if not data.empty:
                    cleaned_data = self._clean_data(data)
                    return cleaned_data
            except Exception as e:
                logging.warning(f"Error collecting data on attempt {i+1}: {e}")
                if i < max_retries - 1:
                    sleep_time = backoff_factor ** i
                    logging.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)

        logging.error(f"Failed to collect data for {asset} after {max_retries} attempts.")
        return pd.DataFrame()

    def get_live_price(self, asset: str) -> float:
        """
        Fetches the most recent price for a given asset.
        """
        if not self.is_market_open():
            logging.warning("Market is closed. Live price may not be available.")

        try:
            data = yf.download(asset, period='1d', interval='1m', progress=False)
            if not data.empty:
                price = data['Close'].iloc[-1]
                if hasattr(price, 'item'):
                    return price.item()
                return price
        except Exception as e:
            logging.error(f"Could not fetch live price for {asset}: {e}")
        return 0.0

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs data cleaning and handles missing values.
        """
        df.dropna(inplace=True)
        return df

    def adapt_to_new_asset(self, asset: str):
        """
        Adapts the data collection pipeline to a new asset.
        """
        # This would involve transfer learning of data patterns from known assets.
        print(f"Adapting to new asset: {asset}")
        pass
