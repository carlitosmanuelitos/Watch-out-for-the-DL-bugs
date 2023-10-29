import pandas as pd
from cryptocmd import CmcScraper


# Other settings
from IPython.display import display, HTML
import os, warnings, logging
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.3f}'.format)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class CryptoData:
    """
    The CryptoData class is responsible for fetching and validating cryptocurrency data. 
    It provides methods to fetch raw data, validate its integrity, and format it for display.
    
    Attributes:
    - EXPECTED_COLUMNS: A set of expected columns in the fetched data.
    - crypto_symbols: A list of cryptocurrency symbols to fetch.
    - retries: The maximum number of data fetch retries.
    - backoff_factor: The exponential backoff factor for retries.
    """
    
    EXPECTED_COLUMNS = {'Date', 'Open', 'High', 'Low', 'Close', 'Market Cap', 'Volume'}

    def __init__(self, crypto_symbols: list[str], retries: int = 5, backoff_factor: float = 0.3):
        """Initializes the class with the given list of cryptocurrency symbols."""
        logger.info("Initializing CryptoData class.")
        self.crypto_symbols = crypto_symbols
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.DATA_DIR = "crypto_assets"
        if not os.path.exists(self.DATA_DIR):
            os.makedirs(self.DATA_DIR)
        logger.info("CryptoData class initialized.")

    def _fetch_cryptocmd_data(self, symbol: str) -> pd.DataFrame:
        """Fetches cryptocurrency data with retries and exponential backoff."""
        logger.info(f"Fetching data for {symbol}.")
        scraper = CmcScraper(symbol)
        df = scraper.get_dataframe()

        # Drop unwanted columns
        unwanted_columns = ['Time Open', 'Time High', 'Time Low', 'Time Close']
        df.drop(columns=unwanted_columns, inplace=True)
        
        # Sort data by Date in ascending order
        df.sort_values(by='Date', ascending=True, inplace=True)
        return df

    def _local_data_path(self, symbol: str) -> str:
        return os.path.join(self.DATA_DIR, f"data_c_{symbol}.csv")

    def get_cryptocmd_data(self, symbol: str, overwrite: bool = False) -> pd.DataFrame:
        """Fetches and returns the cryptocurrency data."""
        logger.info(f"Retrieving {symbol} data.")
        df = self._fetch_cryptocmd_data(symbol)
        
        # Save to local storage if needed
        file_path = self._local_data_path(symbol)
        if overwrite or not os.path.exists(file_path):
            df.to_csv(file_path, index=False)
        
        # Set 'Date' as the index
        df.set_index('Date', inplace=True)
        return df

    def get_all_data(self, overwrite: bool = False) -> dict[str, pd.DataFrame]:
        """Fetches data for all specified cryptocurrencies."""
        logger.info("Getting data for all specified cryptocurrencies.")
        data_dict = {}
        for symbol in self.crypto_symbols:
            data_dict[symbol] = self.get_cryptocmd_data(symbol, overwrite)
        logger.info("All data retrieved successfully.")
        return data_dict

    @staticmethod
    def _format_monetary_value(value: float) -> str:
        """Formats a monetary value to a string."""
        return "${:,.2f}".format(value)

    @staticmethod
    def _format_volume_value(value: float) -> str:
        """Formats a volume value to a string."""
        if value > 1e9:
            return "{:.2f}B".format(value/1e9)
        elif value > 1e6:
            return "{:.2f}M".format(value/1e6)
        else:
            return "{:,.2f}".format(value)

    def get_display_data(self, symbol: str) -> pd.DataFrame:
        """Formats the cryptocurrency data for display."""
        logger.info(f"Formatting display data for {symbol}.")
        
        # Load the data for the given symbol from local storage
        file_path = self._local_data_path(symbol)
        if not os.path.exists(file_path):
            raise ValueError(f"No data found for {symbol}. Please fetch the data first.")
        display_df = pd.read_csv(file_path, parse_dates=['Date']).set_index('Date')
        
        # Format the data
        monetary_columns = ['Open', 'High', 'Low', 'Close']
        display_df[monetary_columns] = display_df[monetary_columns].applymap(self._format_monetary_value)
        volume_like_columns = ['Volume', 'Market Cap']
        display_df[volume_like_columns] = display_df[volume_like_columns].applymap(self._format_volume_value)
        logger.info(f"Display data formatted successfully for {symbol}.")
        return display_df
    

crypto_data_obj = CryptoData(['BTC', 'ETH', 'ADA'])
all_data = crypto_data_obj.get_all_data(overwrite=True)
btc_data, eth_data, ada_data = all_data['BTC'], all_data['ETH'], all_data['ADA']
btc_display_data = crypto_data_obj.get_display_data('BTC')