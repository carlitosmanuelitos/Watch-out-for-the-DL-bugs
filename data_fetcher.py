import pandas as pd
from cryptocmd import CmcScraper
import time
import datetime
from requests.exceptions import RequestException

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
    unwanted_columns = ['Time Open', 'Time High', 'Time Low', 'Time Close']

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
        logger.info(f"Attempting to fetch data for {symbol} from the scraper.")
        attempt = 0
        while attempt < self.retries:
            try:
                scraper = CmcScraper(symbol)
                df = scraper.get_dataframe()
                df.drop(columns=self.unwanted_columns, inplace=True)
                df.sort_values(by='Date', ascending=True, inplace=True)
                logger.info(f"Data successfully fetched for {symbol}.")
                return df
            except RequestException as e:
                logger.warning(f"Attempt {attempt + 1} for {symbol} failed: {e}")
                time.sleep(self.backoff_factor * (2 ** attempt))
                attempt += 1

        logger.error(f"Failed to fetch data for {symbol} after {self.retries} attempts.")
        raise Exception(f"Failed to fetch data for {symbol} after {self.retries} attempts.")

    def _local_data_path(self, symbol: str) -> str:
        today = datetime.date.today().isoformat()
        return os.path.join(self.DATA_DIR, f"data_c_{symbol}_{today}.csv")

    def _is_data_file_present(self, file_path: str) -> bool:
        """Check if the data file for today's date exists."""
        if os.path.exists(file_path):
            logger.info(f"Data file found: {file_path}")
            return True
        logger.info(f"No data file for today's date. File checked: {file_path}")
        return False

    def get_cryptocmd_data(self, symbol: str, overwrite: bool = False) -> pd.DataFrame:
        logger.info(f"Retrieving data for {symbol}.")
        file_path = self._local_data_path(symbol)

        if not overwrite and self._is_data_file_present(file_path):
            logger.info(f"Loading data for {symbol} from local storage.")
            return pd.read_csv(file_path, parse_dates=['Date']).set_index('Date')

        logger.info(f"Fetching new data for {symbol} as local data is not up-to-date.")
        df = self._fetch_cryptocmd_data(symbol)

        if not self.EXPECTED_COLUMNS.issubset(df.columns):
            missing_columns = self.EXPECTED_COLUMNS - set(df.columns)
            logger.error(f"Data for {symbol} is missing columns: {missing_columns}")
            raise ValueError(f"Data for {symbol} is missing columns: {missing_columns}")

        df.to_csv(file_path, index=False)
        logger.info(f"New data for {symbol} saved to local storage.")
        return df.set_index('Date')

    def get_all_data(self, overwrite: bool = False) -> dict[str, pd.DataFrame]:
        logger.info("Fetching data for all specified cryptocurrencies.")
        data_dict = {}
        for symbol in self.crypto_symbols:
            logger.info(f"Processing data for {symbol}.")
            data_dict[symbol] = self.get_cryptocmd_data(symbol, overwrite)
        logger.info("All cryptocurrency data retrieved successfully.")
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
    
    def get_all_display_data(self) -> pd.DataFrame:
        """Fetches display data for all specified cryptocurrencies and concatenates them into a single DataFrame."""
        logger.info("Getting display data for all specified cryptocurrencies.")
        display_data_list = []
        
        for symbol in self.crypto_symbols:
            display_df = self.get_display_data(symbol)
            display_df['Ticker'] = symbol  # Add a column for the ticker symbol
            display_data_list.append(display_df)
        
        # Concatenate all the DataFrames along the index
        all_display_data = pd.concat(display_data_list, keys=self.crypto_symbols)
        
        logger.info("All display data retrieved successfully.")
        return all_display_data


    
# Part 1 - Data Fetching
def run_data_fetcher(run: bool, tickers: list, get_display_data=False, overwrite=False):
    if not run:
        return None, None  # Return None for all objects if not running
    
    # Fetch the data
    crypto_data_obj = CryptoData(tickers)
    all_data = crypto_data_obj.get_all_data(overwrite=overwrite)
    
    all_display_data = None
    if get_display_data:
        all_display_data = crypto_data_obj.get_all_display_data()
    
    # Return the objects
    return crypto_data_obj, all_data, all_display_data

tickers = ['BTC','ETH','ADA']
crypto_data_obj, all_data, all_display_data = run_data_fetcher(True, tickers, get_display_data=True, overwrite=False)
btc_data = all_data['BTC']
btc_display_data = all_display_data.loc['BTC']
