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

import pandas as pd
import numpy as np
from data_fetcher import run_data_fetcher
from data_fetcher import btc_data


# Other settings
from IPython.display import display, HTML
import os, warnings, logging
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.3f}'.format)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class CryptoDataAnalytics:
    """
    This class is responsible for performing enhanced analytics on cryptocurrency data.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the cryptocurrency data.
        output_dir (str): The directory where analytics files will be saved.
    """
    
    def __init__(self, crypto_data: pd.DataFrame):
        logger.info("Initializing CryptoDataAnalytics class.")
        self.df = crypto_data
        self.output_dir = 'analytics_csv'
        self._create_output_dir()
        logger.info("CryptoDataAnalytics class initialized successfully.")
        
    def _create_output_dir(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
            
    def calculate_historical_volatility(self, column: str = 'Close', window: int = 30) -> pd.DataFrame:
        """Calculates historical volatility."""
        logger.info("Initiating historical volatility calculation.")
        if len(self.df) < window:
            logger.error("Data length is less than the rolling window size. Cannot calculate volatility.")
            raise ValueError("Insufficient data for volatility calculation.")
        
        log_ret = np.log(self.df[column] / self.df[column].shift(1))
        volatility = log_ret.rolling(window=window).std()
        logger.info("Historical volatility calculation successful.")
        return pd.DataFrame(volatility, columns=['Historical Volatility'])
        
    def perform_time_analysis(self, freq: str):
        """Performs time-based analysis."""
        logger.info(f"Initiating {freq}-based time analysis.")
        data = self.df.resample(freq).agg({'Close': ['last', 'mean', 'max', 'min'], 'Open': 'first'})
        data.columns = data.columns.map('_'.join).str.strip('_')
        data = self.calculate_price_variation(data)
        
        # Reorder columns
        ordered_columns = ['Close_mean', 'Close_max', 'Close_min', 'Close_last', 'Open_first', 'variation_$_abs', 'variation_%_rel']
        data = data[ordered_columns]
        
        logger.info(f"{freq}-based time analysis successful.")
        return data

    def calculate_price_variation(self, data: pd.DataFrame):
        """Calculates price variation."""
        logger.info("Initiating price variation calculation.")
        data['variation_$_abs'] = data['Close_last'] - data['Open_first']
        data['variation_%_rel'] = ((data['Close_last'] - data['Open_first']) / data['Open_first']) * 100
        logger.info("Price variation calculation successful.")
        return data
    
    def retrieve_all_time_records(self):
        """Retrieves all-time price records."""
        logger.info("Initiating retrieval of all-time records.")
        all_time_high = self.df['Close'].max()
        all_time_low = self.df['Close'].min()
        all_time_high_date = self.df['Close'].idxmax().strftime('%Y-%m-%d')
        all_time_low_date = self.df['Close'].idxmin().strftime('%Y-%m-%d')
        logger.info("All-time records retrieval successful.")
        return all_time_high, all_time_low, all_time_high_date, all_time_low_date
    
    def perform_and_save_all_analyses(self):
        """Performs all analyses and saves them to Excel files."""
        logger.info("Initiating all analyses.")
        self.save_analysis_to_excel(self.perform_time_analysis('Y'), 'yearly_data.xlsx')
        self.save_analysis_to_excel(self.perform_time_analysis('M'), 'monthly_data.xlsx')
        self.save_analysis_to_excel(self.perform_time_analysis('W'), 'weekly_data.xlsx')
        logger.info("All analyses have been successfully performed and saved.")
        
    def save_analysis_to_excel(self, analysis: pd.DataFrame, filename: str):
        """Saves the given DataFrame to an Excel file in the output directory."""
        filepath = os.path.join(self.output_dir, filename)
        analysis.to_excel(filepath)
        logger.info(f"Analysis saved to {filepath}.")


def run_data_analytics(run: bool):
    if not run:
        return None, None, None, None, None

    analytics = CryptoDataAnalytics(btc_data)

    # Retrieve and display all-time records
    all_time_high, all_time_low, all_time_high_date, all_time_low_date = analytics.retrieve_all_time_records()
    print(f"All Time High: {all_time_high} on {all_time_high_date}")
    print(f"All Time Low: {all_time_low} on {all_time_low_date}")

    # Run all analyses and save them
    analytics.perform_and_save_all_analyses()
    yearly_data = analytics.perform_time_analysis('Y')
    monthly_data = analytics.perform_time_analysis('M')
    weekly_data = analytics.perform_time_analysis('W')

    return analytics, all_time_high, all_time_low, yearly_data, monthly_data, weekly_data

#analytics, all_time_high, all_time_low, yearly_data, monthly_data, weekly_data = run_data_analytics(True)

import pandas as pd
import numpy as np
from scipy.signal import detrend
from pandas.tseries.holiday import USFederalHolidayCalendar
from statsmodels.tsa.api import seasonal_decompose
from data_fetcher import btc_data


# Other settings
from IPython.display import display, HTML
import os, warnings, logging
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.3f}'.format)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class Feature_Eng_Tech:
    """
    The Feature_Eng_Tech class is responsible for applying various feature engineering techniques on time series data.
    
    Attributes:
        df (pd.DataFrame): Original time series data.
        target_column (str): Target column for which features are being generated.
        data_eng (pd.DataFrame): DataFrame with engineered features.
        logger (logging.Logger): Logger for tracking operations and debugging.
        
    Methods:
        reset_data: Resets the engineered data to its original state.
        handle_missing_values: Handles missing values in the DataFrame.
        add_date_features: Adds date-related features like year, month, day, and optionally day of the week.
        add_lag_features: Adds lag features based on a given window size.
        add_rolling_features: Adds rolling window features like mean and standard deviation.
        add_expanding_window_features: Adds expanding window features like mean, min, max, and sum.
        add_seasonal_decomposition: Adds seasonal decomposition features like trend, seasonality, and residuals.
        detrend_data: Detrends the time series data.
        add_holiday_features: Adds a feature to indicate holidays.
        add_fourier_features: Adds Fourier features based on a given period and order.
        handle_nan_values_post_engineering: Handles NaN values post feature engineering.
        feature_engineering: Applies multiple feature engineering methods based on a configuration dictionary.
        get_engineered_data: Returns the DataFrame with engineered features.
    """
    
    def __init__(self, df: pd.DataFrame, target_column: str):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input data should be a pandas DataFrame.")
        if target_column not in df.columns:
            raise ValueError(f"Target column {target_column} not found in DataFrame.")
        
        self.df = df.copy()
        self.target_column = target_column
        self.data_eng = self.df.copy()
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized Feature_Eng_Tech.")

    def reset_data(self):
        """Resets the engineered data to its original state."""
        self.data_eng = self.df.copy()
        self.logger.info("Reset data to the original state.")

    def handle_missing_values(self, method: str = 'ffill'):
        """
        Handles missing values in the DataFrame.
        
        Parameters:
            method (str): Method to handle missing values ('ffill', 'bfill', 'interpolate', 'drop'). Default is 'ffill'.
        """
        if method not in ['ffill', 'bfill', 'interpolate', 'drop']:
            raise ValueError("Invalid method for handling missing values. Choose 'ffill', 'bfill', 'interpolate', or 'drop'.")
        if self.df.isnull().sum().sum() > 0:
            self.df.fillna(method=method, inplace=True)
            self.logger.info(f"Handled missing values using {method} method.")
        else:
            self.logger.info("No missing values detected.")

    def add_date_features(self, include_day_of_week: bool = True):
        """
        Adds date-related features like year, month, day, and optionally day of the week.
        
        Parameters:
            include_day_of_week (bool): Whether to include the day of the week as a feature. Default is True.
        """
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)
        self.df['year'] = self.df.index.year
        self.df['month'] = self.df.index.month
        self.df['day'] = self.df.index.day
        if include_day_of_week:
            self.df['day_of_week'] = self.df.index.dayofweek
        self.logger.info("Date-related features added.")

    def add_lag_features(self, window: int = 3):
        """
        Adds lag features based on a given window size.
        
        Parameters:
            window (int): The window size for creating lag features. Default is 3.
        """
        if window > len(self.data_eng):
            raise ValueError("The window parameter should be less than the length of the time series data.")
        for i in range(1, window + 1):
            self.data_eng[f"lag_{i}"] = self.data_eng[self.target_column].shift(i)
        self.logger.info(f'Added lag features with window size {window}.')

    def add_rolling_features(self, window: int = 3, min_periods: int = 1):
        """
        Adds rolling window features like mean and standard deviation.
        
        Parameters:
            window (int): The window size for rolling features. Default is 3.
            min_periods (int): Minimum number of observations required to have a value. Default is 1.
        """
        self.data_eng[f"rolling_mean_{window}"] = self.data_eng[self.target_column].rolling(window=window, min_periods=min_periods).mean()
        self.data_eng[f"rolling_std_{window}"] = self.data_eng[self.target_column].rolling(window=window, min_periods=min_periods).std()
        self.logger.info(f'Added rolling window features with window size {window}.')

    def add_expanding_window_features(self, min_periods: int = 1):
        """
        Adds expanding window features like mean, min, max, and sum.
        
        Parameters:
            min_periods (int): Minimum number of observations required to have a value. Default is 1.
        """
        self.data_eng['expanding_mean'] = self.data_eng[self.target_column].expanding(min_periods=min_periods).mean()
        self.data_eng['expanding_min'] = self.data_eng[self.target_column].expanding(min_periods=min_periods).min()
        self.data_eng['expanding_max'] = self.data_eng[self.target_column].expanding(min_periods=min_periods).max()
        self.data_eng['expanding_sum'] = self.data_eng[self.target_column].expanding(min_periods=min_periods).sum()
        self.logger.info('Added expanding window features.')

    def add_seasonal_decomposition(self, period: int = 12, model: str = 'additive'):
        """
        Adds seasonal decomposition features like trend, seasonality, and residuals.
        
        Parameters:
            period (int): The period for seasonal decomposition. Default is 12.
            model (str): The model type for seasonal decomposition ('additive' or 'multiplicative'). Default is 'additive'.
        """
        result = seasonal_decompose(self.data_eng[self.target_column], period=period, model=model)
        self.data_eng['trend'] = result.trend
        self.data_eng['seasonal'] = result.seasonal
        self.data_eng['residual'] = result.resid
        self.logger.info(f'Added seasonal decomposition with period {period} and model {model}.')

    def detrend_data(self):
        """Detrends the time series data."""
        self.data_eng['detrended'] = detrend(self.data_eng[self.target_column])
        self.logger.info('Detrended the data.')

    def add_holiday_features(self):
        """Adds a feature to indicate holidays."""
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=self.data_eng.index.min(), end=self.data_eng.index.max())
        self.data_eng['is_holiday'] = self.data_eng.index.isin(holidays).astype(int)
        self.logger.info('Added holiday features.')

    def add_fourier_features(self, period: int, order: int):
        """
        Adds Fourier features based on a given period and order.
        
        Parameters:
            period (int): The period for Fourier features.
            order (int): The order for Fourier features.
        """
        for i in range(1, order + 1):
            self.data_eng[f'fourier_sin_{i}'] = np.sin(2 * i * np.pi * self.data_eng.index.dayofyear / period)
            self.data_eng[f'fourier_cos_{i}'] = np.cos(2 * i * np.pi * self.data_eng.index.dayofyear / period)
        self.logger.info(f'Added Fourier features with period {period} and order {order}.')

    def handle_nan_values_post_engineering(self, method: str = 'drop'):
        """
        Handles NaN values post feature engineering.
        
        Parameters:
            method (str): Method to handle missing values ('drop', 'ffill', 'bfill'). Default is 'drop'.
        """
        if method == 'drop':
            self.data_eng.dropna(inplace=True)
        elif method == 'ffill':
            self.data_eng.fillna(method='ffill', inplace=True)
        elif method == 'bfill':
            self.data_eng.fillna(method='bfill', inplace=True)
        else:
            raise ValueError("Invalid method. Choose 'drop', 'ffill', or 'bfill'.")
        self.logger.info(f"Handled NaN values using {method} method.")

    def feature_engineering(self, config: dict):
        """
        Applies multiple feature engineering methods based on a configuration dictionary.
        
        Parameters:
            config (dict): A dictionary with the configuration for feature engineering.
        """
        feature_methods = {
            "handle_missing_values": self.handle_missing_values,
            "add_date_features": self.add_date_features,
            "add_lag_features": self.add_lag_features,
            "add_rolling_features": self.add_rolling_features,
            "add_expanding_window_features": self.add_expanding_window_features,
            "add_seasonal_decomposition": self.add_seasonal_decomposition,
            "detrend_data": self.detrend_data,
            "add_holiday_features": self.add_holiday_features,
            "add_fourier_features": lambda: self.add_fourier_features(config.get("fourier_period", 365), config.get("fourier_order", 3))
        }

        for feature, method in feature_methods.items():
            if config.get(feature):
                method()
        self.handle_nan_values_post_engineering()
        self.logger.info('Feature engineering steps applied based on configuration.')

    def get_engineered_data(self) -> pd.DataFrame:
        """
        Returns the DataFrame with engineered features.
        
        Returns:
            pd.DataFrame: DataFrame containing the engineered features.
        """
        return self.data_eng.copy()
    

def run_feature_engineering(run: bool, config: dict):
    if not run:
        return None

    feature_eng = Feature_Eng_Tech(btc_data, target_column='Close')
    feature_eng.feature_engineering(config)
    data_eng = feature_eng.get_engineered_data()

    return data_eng

# This is how you can call it in the same script
config = {
    "handle_missing_values": True,
    "add_date_features": True,
    "add_lag_features": True,
    "add_rolling_features": True,
    "add_expanding_window_features": True,
    "add_seasonal_decomposition": True,
    "detrend_data": True,
    "add_holiday_features": True,
    "add_fourier_features": True,
}

#data_eng = run_feature_engineering(True, config)
#display(data_eng)


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import List, Optional
from typing import Optional, List, Tuple
from scipy.stats import boxcox

# Other settings
from IPython.display import display, HTML
import os, warnings, logging
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.3f}'.format)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedDataPreprocessor:
    """ 
    UnifiedDataPreprocessor is responsible for preprocessing time series data.
    It performs actions like data splitting, normalization, reshaping, and sequence generation.
    
    Attributes:
        data (pd.DataFrame): Original time series data.
        target_column (str): Target column for preprocessing.
        logger (logging.Logger): Logger for tracking operations and debugging.
        transformations (list): List of applied transformations.
    """
    
    def __init__(self, df, target_column, logger=None):
        self.data = df.copy()
        self.target_column = target_column
        self.scalers = {}
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.X_train_seq, self.X_test_seq, self.y_train_seq, self.y_test_seq = None, None, None, None
        self.logger = logger if logger else logging.getLogger(__name__)
        self.transformations = []
        self.lambda_val = None  
        self.scalers = {
            "MinMax": MinMaxScaler(),
            "Standard": StandardScaler(),
            "Robust": RobustScaler(),
            "Quantile": QuantileTransformer(output_distribution='normal'),
            "Power": PowerTransformer(method='yeo-johnson')
        }
        self.logger.info("Initializing DataPreprocessor...")        
    
    def get_scaler(self, scaler_type: str):
        self.logger.info(f"Getting scaler of type: {scaler_type}")
        try:
            return self.scalers[scaler_type]
        except KeyError:
            raise ValueError(f"Invalid scaler_type. Supported types are: {', '.join(self.scalers.keys())}")

    def split_and_plot_data(self, test_size: float = 0.2, split_date: Optional[str] = None, plot: bool = True):
        self.logger.info("Splitting data...")
        self.transformations.append('Data Splitting')
        features = self.data.drop(columns=[self.target_column])
        target = self.data[self.target_column]

        if split_date:
            train_mask = self.data.index < split_date
            self.X_train, self.X_test = features[train_mask], features[~train_mask]
            self.y_train, self.y_test = target[train_mask], target[~train_mask]
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                features, target, test_size=test_size, shuffle=False
            )

        self.logger.info(f"Data split completed. X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")

        if plot:
            plt.figure(figsize=(20, 7))
            plt.subplot(1, 2, 1)
            plt.title('Training Data - Target')
            plt.plot(self.y_train, label=self.target_column)
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.title('Test Data - Target')
            plt.plot(self.y_test, label=self.target_column)
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.show()

    def normalize_data(self, scaler_type: str = 'MinMax', plot: bool = True):
        self.logger.info("Normalizing feature data...")
        scaler = self.get_scaler(scaler_type)
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.scalers['features'] = scaler
        self.logger.info("Feature data normalization completed.")
        self.transformations.append(f"Feature normalization with {scaler_type} scaler")

        if plot:
            plt.figure(figsize=(20, 8))
            plt.subplot(1, 2, 1)
            plt.title('Normalized Training Features')
            for i in range(self.X_train.shape[1]):
                plt.plot(self.X_train[:, i], label=f'Feature {i}')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.title('Normalized Test Features')
            for i in range(self.X_test.shape[1]):
                plt.plot(self.X_test[:, i], label=f'Feature {i}')
            plt.legend()
            plt.show()

    def normalize_target(self, scaler_type: str = 'MinMax', plot: bool = True):
        self.logger.info("Normalizing target data...")
        scaler = self.get_scaler(scaler_type)
        self.y_train = scaler.fit_transform(self.y_train.values.reshape(-1, 1))
        self.y_test = scaler.transform(self.y_test.values.reshape(-1, 1))
        self.scalers['target'] = scaler
        self.logger.info("Target data normalization completed.")
        self.transformations.append(f"Target normalization with {scaler_type} scaler")

        if plot:
            plt.figure(figsize=(20, 7))
            plt.subplot(1, 2, 1)
            plt.title('Normalized Training Target')
            plt.plot(self.y_train, label='Normalized ' + self.target_column)
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.title('Normalized Test Target')
            plt.plot(self.y_test, label='Normalized ' + self.target_column)
            plt.legend()
            plt.show()

    def difference_and_plot_data(self, interval: int = 1, plot: bool = True):
        self.logger.info(f"Applying differencing with interval {interval}...")
        self.data = self.data.diff(periods=interval).dropna()
        self.transformations.append(f'Differencing with interval {interval}')
        self.logger.info("Differencing applied.")
        
        if plot:
            plt.figure(figsize=(20, 7))
            plt.title('Data after Differencing')
            plt.plot(self.data[self.target_column], label=self.target_column)
            plt.legend()
            plt.show()

    def box_cox_transform_and_plot(self, lambda_val: Optional[float] = None, plot: bool = True):
        if self.y_train is None or self.y_test is None:
            self.logger.warning("Data not split yet. Run split_data first.")
            return self  # Allow method chaining

        if np.any(self.y_train <= 0) or np.any(self.y_test <= 0):
            self.logger.warning("Data must be positive for Box-Cox transformation.")
            return self  # Allow method chaining

        self.logger.info("Applying Box-Cox transformation...")
        self.y_train = self.y_train.ravel()
        self.y_test = self.y_test.ravel()
        self.y_train, fitted_lambda = boxcox(self.y_train)
        self.lambda_val = fitted_lambda if lambda_val is None else lambda_val
        self.y_test = boxcox(self.y_test, lmbda=self.lambda_val)
        self.transformations.append(f"Box-Cox transformation with lambda {self.lambda_val}")
        self.logger.info(f"Box-Cox transformation applied with lambda {self.lambda_val}.")

        if plot:
            plt.figure(figsize=(20, 7))
            plt.subplot(1, 2, 1)
            plt.title('Box-Cox Transformed Training Target')
            plt.plot(self.y_train, label='Transformed ' + self.target_column)
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.title('Box-Cox Transformed Test Target')
            plt.plot(self.y_test, label='Transformed ' + self.target_column)
            plt.legend()
            plt.show()

    def inverse_box_cox_and_plot(self, plot: bool = True):
        if "Box-Cox transformation" not in "".join(self.transformations):
            self.logger.warning("No Box-Cox transformation found on the target column. Skipping inverse transformation.")
            return

        self.logger.info("Applying inverse Box-Cox transformation...")
        self.y_train = invboxcox(self.y_train, self.lambda_val)
        self.y_test = invboxcox(self.y_test, self.lambda_val)
        self.transformations.remove(f"Box-Cox transformation with lambda {self.lambda_val}")
        self.logger.info(f"Inverse Box-Cox transformation applied on column {self.target_column}.")
        
        if plot:
            plt.figure(figsize=(20, 7))
            plt.subplot(1, 2, 1)
            plt.title('Inverse Box-Cox Transformed Training Target')
            plt.plot(self.y_train, label='Inverse Transformed ' + self.target_column)
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.title('Inverse Box-Cox Transformed Test Target')
            plt.plot(self.y_test, label='Inverse Transformed ' + self.target_column)
            plt.legend()
            plt.show()

    def reshape_for_recurrent(self, data: np.array) -> np.array:
        self.logger.info("Reshaping data for recurrent models...")
        reshaped_data = data.reshape(data.shape)
        self.logger.info(f"Data reshaped to {reshaped_data.shape}.")
        self.transformations.append('Data Reshaped')
        return reshaped_data

    def generate_sequences(self, X_data: np.array, y_data: np.array, n_steps: int, seq_to_seq: bool = False) -> Tuple[np.array, np.array]:
        X, y = [], []
        for i in range(len(X_data) - n_steps):
            seq_x = X_data[i:i + n_steps, :]
            if seq_to_seq:
                seq_y = y_data[i:i + n_steps, :]
            else:
                seq_y = y_data[i + n_steps - 1]
            X.append(seq_x)
            y.append(seq_y)
        self.logger.info(f"Generated {len(X)} sequences of shape {X[0].shape}.")
        self.transformations.append('Sequences Generated')
        return np.array(X), np.array(y)
    
    def prepare_data_for_recurrent(self, n_steps: int, seq_to_seq: bool = False) -> Tuple[np.array, np.array, np.array, np.array]:
        self.logger.info(f"Preparing data for recurrent models with {n_steps} timesteps...")
        X_train_seq, y_train_seq = self.generate_sequences(self.X_train, self.y_train, n_steps, seq_to_seq)
        X_test_seq, y_test_seq = self.generate_sequences(self.X_test, self.y_test, n_steps, seq_to_seq)

        # Update instance variables here
        self.X_train_seq = self.reshape_for_recurrent(X_train_seq)
        self.X_test_seq = self.reshape_for_recurrent(X_test_seq)
        self.y_train_seq = y_train_seq  # Assuming y_train_seq and y_test_seq are already 2D
        self.y_test_seq = y_test_seq

        self.logger.info("Data preparation for recurrent models completed.")
        return self.X_train_seq, self.y_train_seq, self.X_test_seq, self.y_test_seq

    def prepare_for_prophet(self) -> pd.DataFrame:
        prophet_data = self.data[[self.target_column]].reset_index()
        prophet_data.columns = ['ds', 'y']
        return prophet_data

    def get_preprocessed_data(self) -> Tuple[np.array, np.array, np.array, np.array]:
        return self.X_train, self.y_train, self.X_test, self.y_test

    def __str__(self) -> str:
        return "Transformations applied: " + ", ".join(self.transformations)


from IPython.display import display, HTML
import json, joblib, hashlib, logging, os, warnings
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from joblib import dump, load
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_notebook, save
from bokeh.models import (HoverTool, ColumnDataSource, WheelZoomTool, Span, Range1d,
                          FreehandDrawTool, MultiLine, NumeralTickFormatter, Button, CustomJS)
from bokeh.layouts import column, row
from bokeh.io import curdoc, export_png
from bokeh.models.widgets import CheckboxGroup
from bokeh.themes import Theme
# Machine Learning Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, accuracy_score

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, LSTM, TimeDistributed, Conv1D, MaxPooling1D, Flatten,
                                    ConvLSTM2D, BatchNormalization, GRU, Bidirectional, Attention, Input,
                                    Reshape, GlobalAveragePooling1D, GlobalMaxPooling1D, Lambda, LayerNormalization, 
                                    SimpleRNN, Layer, Multiply, Add, Activation)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tcn import TCN
from kerasbeats import NBeatsModel
from statsmodels.tsa.stattools import acf, pacf


# Other settings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.3f}'.format)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


# LSTM Sequece-to-One
from data_fetcher import btc_data
from data_preprocessor import UnifiedDataPreprocessor
df = btc_data.copy()
data_preprocessor = UnifiedDataPreprocessor(df, target_column='Close')
data_preprocessor.split_and_plot_data(test_size=0.2, plot=False)
data_preprocessor.normalize_data(scaler_type='MinMax',plot=False)
data_preprocessor.normalize_target(scaler_type='MinMax',plot=False)
X_train_seq, y_train_seq, X_test_seq, y_test_seq = data_preprocessor.prepare_data_for_recurrent(n_steps=10, seq_to_seq=False)



class BaseModelLSTM():
    """
    A base class for LSTM-like machine learning models.
    This class handles data preprocessing, model training, predictions, and evaluations.
    """
    def __init__(self, model_type, data_preprocessor, config, cross_val=False):
        self._validate_input_sequence(data_preprocessor.X_train_seq, data_preprocessor.y_train_seq, data_preprocessor.X_test_seq, data_preprocessor.y_test_seq)
        self.X_train = data_preprocessor.X_train_seq
        self.y_train = data_preprocessor.y_train_seq
        self.X_test = data_preprocessor.X_test_seq
        self.y_test = data_preprocessor.y_test_seq
        self.feature_scaler = data_preprocessor.scalers['features']
        self.target_scaler = data_preprocessor.scalers['target']
        self.data = data_preprocessor.data
        self.config = config
        self.cross_val = cross_val
        self.model_type = model_type
        self.params = {'model_type': model_type}
        self.params.update(config)
        self._initialize_model()
        self.logging = logging.getLogger(__name__)

    def _initialize_model(self):
        logging.info(f"Initializing {self.model_type} model")
        self.model = Sequential()
        
        if self.model_type in ['LSTM', 'GRU']:
            for i, unit in enumerate(self.config['units']):
                return_sequences = True if i < len(self.config['units']) - 1 else False
                layer = LSTM(units=unit, return_sequences=return_sequences) if self.model_type == 'LSTM' else GRU(units=unit, return_sequences=return_sequences)
                self.model.add(layer)
                self.model.add(Dropout(self.config['dropout']))

        elif self.model_type == 'CNN-LSTM':
            self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=self.config['input_shape']))
            self.model.add(Dropout(self.config['dropout']))
            self.model.add(LSTM(units=self.config['units'][0]))

        self.model.add(Dense(units=self.config['dense_units']))
        self.model.compile(optimizer=self.config['optimizer'], loss='mean_squared_error')
        self.model.summary()
    
    def _validate_input_sequence(self, X_train, y_train, X_test, y_test):
        """Validate the shape and type of training and testing sequence data."""
        for arr, name in [(X_train, 'X_train_seq'), (y_train, 'y_train_seq'), (X_test, 'X_test_seq'), (y_test, 'y_test_seq')]:
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"{name} should be a numpy array.")

            if len(arr.shape) < 2:
                raise ValueError(f"{name} should have at least two dimensions.")

            # Special check for X_* arrays, which should be 3D for sequence models
            if 'X_' in name and len(arr.shape) != 3:
                raise ValueError(f"{name} should be a 3D numpy array for sequence models. Found shape {arr.shape}.")
     
    def train_model(self, epochs=100, batch_size=50, early_stopping=True):
        logging.info(f"Training {self.params['model_type']} model")
        callbacks = [EarlyStopping(monitor='val_loss', patience=10)] if early_stopping else None

        if self.cross_val:
            tscv = TimeSeriesSplit(n_splits=5)
            self.history = []
            fold_no = 1
            for train, val in tscv.split(self.X_train):
                logging.info(f"Training on fold {fold_no}")
                history = self.model.fit(self.X_train[train], self.y_train[train], epochs=epochs,
                                         batch_size=batch_size, validation_data=(self.X_train[val], self.y_train[val]),
                                         callbacks=callbacks, shuffle=False)
                self.history.append(history)
                logging.info(f"Done with fold {fold_no}")
                self.model.summary()
                fold_no += 1
        else:
            self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs,
                                          batch_size=batch_size, validation_split=0.2,
                                          callbacks=callbacks, shuffle=False)
        logging.info("Training completed")
        self.model.summary()

    def make_predictions(self):
        logging.info("Making predictions")

        self._make_raw_predictions()
        self._make_unscaled_predictions()
        self._create_comparison_dfs()

        logging.info("Predictions made")

    def _make_raw_predictions(self):
        self.train_predictions = self.model.predict(self.X_train)
        self.test_predictions = self.model.predict(self.X_test)
        logging.info(f"Raw predictions made with shapes train: {self.train_predictions.shape}, test: {self.test_predictions.shape}")

    def _make_unscaled_predictions(self):
        # Check if the shape of the predictions matches that of y_train and y_test
        if self.train_predictions.shape[:-1] != self.y_train.shape[:-1]:
            logging.error(f"Shape mismatch: train_predictions {self.train_predictions.shape} vs y_train {self.y_train.shape}")
            return

        if self.test_predictions.shape[:-1] != self.y_test.shape[:-1]:
            logging.error(f"Shape mismatch: test_predictions {self.test_predictions.shape} vs y_test {self.y_test.shape}")
            return

        # If predictions are 3D, reduce dimensionality by taking mean along last axis
        if self.train_predictions.ndim == 3:
            self.train_predictions = np.mean(self.train_predictions, axis=-1)

        if self.test_predictions.ndim == 3:
            self.test_predictions = np.mean(self.test_predictions, axis=-1)

        # Perform the inverse transformation to get unscaled values
        self.train_predictions = self.target_scaler.inverse_transform(self.train_predictions).flatten()
        self.test_predictions = self.target_scaler.inverse_transform(self.test_predictions).flatten()

        logging.info(f"Unscaled predictions made with shapes train: {self.train_predictions.shape}, test: {self.test_predictions.shape}")

    def _create_comparison_dfs(self):
        y_train_flat = self.target_scaler.inverse_transform(self.y_train).flatten()
        y_test_flat = self.target_scaler.inverse_transform(self.y_test).flatten()

        # Obtain date indices from original data
        train_date_index = self.data.index[:len(self.y_train)]
        test_date_index = self.data.index[-len(self.y_test):]

        if y_train_flat.shape != self.train_predictions.shape:
            logging.error(f"Shape mismatch between y_train {y_train_flat.shape} and train_predictions {self.train_predictions.shape}")
        else:
            self.train_comparison_df = pd.DataFrame({'Actual': y_train_flat, 'Predicted': self.train_predictions})
            # Set date index for train_comparison_df
            self.train_comparison_df.set_index(train_date_index, inplace=True)

        if y_test_flat.shape != self.test_predictions.shape:
            logging.error(f"Shape mismatch between y_test {y_test_flat.shape} and test_predictions {self.test_predictions.shape}")
        else:
            self.test_comparison_df = pd.DataFrame({'Actual': y_test_flat, 'Predicted': self.test_predictions})
            # Set date index for test_comparison_df
            self.test_comparison_df.set_index(test_date_index, inplace=True)

    def evaluate_model(self):
            logging.info("Evaluating LSTM model")
            metrics = {'RMSE': mean_squared_error, 'R2 Score': r2_score,
                       'MAE': mean_absolute_error, 'Explained Variance': explained_variance_score}

            evaluation = {}
            for name, metric in metrics.items():
                if name == 'RMSE':
                    train_evaluation = metric(self.train_comparison_df['Actual'],
                                              self.train_comparison_df['Predicted'],
                                              squared=False)
                    test_evaluation = metric(self.test_comparison_df['Actual'],
                                             self.test_comparison_df['Predicted'],
                                             squared=False)
                else:
                    train_evaluation = metric(self.train_comparison_df['Actual'],
                                              self.train_comparison_df['Predicted'])
                    test_evaluation = metric(self.test_comparison_df['Actual'],
                                             self.test_comparison_df['Predicted'])
                evaluation[name] = {'Train': train_evaluation, 'Test': test_evaluation}

            self.evaluation_df = pd.DataFrame(evaluation)
            logging.info("Evaluation completed")
            return self.evaluation_df
   
    def plot_history(self, plot=True):
        if not plot:
            return
        if not hasattr(self, 'history'):
            print("No training history is available. Train model first.")
            return
        # Extracting loss data from training history
        train_loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = list(range(1, len(train_loss) + 1))
        # Preparing data
        source = ColumnDataSource(data=dict(
            epochs=epochs,
            train_loss=train_loss,
            val_loss=val_loss
        ))

        p1 = figure(width=700, height=600, title="Training Loss over Epochs",x_axis_label='Epochs', y_axis_label='Loss')
        hover1 = HoverTool()
        hover1.tooltips = [("Epoch", "@epochs"), ("Loss", "@{train_loss}{0,0.0000}")]
        p1.add_tools(hover1)
        hover2 = HoverTool()
        hover2.tooltips = [("Epoch", "@epochs"), ("Validation Loss", "@{val_loss}{0,0.0000}")]
        p1.add_tools(hover2)
        p1.line(x='epochs', y='train_loss', legend_label="Training Loss", line_width=2, source=source, color="green")
        p1.line(x='epochs', y='val_loss', legend_label="Validation Loss", line_width=2, source=source, color="red")
        p1.legend.location = "top_right"
        p1.legend.click_policy = "hide"

        output_notebook()
        return(show(p1, notebook_handle=True))

    def plot_predictions(self, plot=True):
        if not plot:
            return        
        if not hasattr(self, 'train_comparison_df') or not hasattr(self, 'test_comparison_df'):
            print("No predictions are available. Generate predictions first.")
            return
        actual_train = self.train_comparison_df['Actual']
        predicted_train = self.train_comparison_df['Predicted']
        actual_test = self.test_comparison_df['Actual']
        predicted_test = self.test_comparison_df['Predicted']
        index_train = self.train_comparison_df.index
        index_test = self.test_comparison_df.index

        # Preparing data
        source_train = ColumnDataSource(data=dict(
            index=index_train,
            actual_train=actual_train,
            predicted_train=predicted_train
        ))

        source_test = ColumnDataSource(data=dict(
            index=index_test,
            actual_test=actual_test,
            predicted_test=predicted_test
        ))

        p2 = figure(width=700, height=600, title="Training Data: Actual vs Predicted", x_axis_label='Date', y_axis_label='Value', x_axis_type="datetime")
        p3 = figure(width=700, height=600, title="Testing Data: Actual vs Predicted",x_axis_label='Date', y_axis_label='Value', x_axis_type="datetime")
        p2.line(x='index', y='actual_train', legend_label="Actual", line_width=2, source=source_train, color="green")
        p2.line(x='index', y='predicted_train', legend_label="Predicted", line_width=2, source=source_train, color="red")
        p3.line(x='index', y='actual_test', legend_label="Actual", line_width=2, source=source_test, color="green")
        p3.line(x='index', y='predicted_test', legend_label="Predicted", line_width=2, source=source_test, color="red")
        p2.legend.location = "top_left" 
        p2.legend.click_policy = "hide"
        p3.legend.location = "top_left" 
        p3.legend.click_policy = "hide"
        hover_train = HoverTool()
        hover_train.tooltips = [
            ("Date", "@index{%F}"),
            ("Actual Value", "@{actual_train}{0,0.0000}"),
            ("Predicted Value", "@{predicted_train}{0,0.0000}")
        ]
        hover_train.formatters = {"@index": "datetime"}

        hover_test = HoverTool()
        hover_test.tooltips = [
            ("Date", "@index{%F}"),
            ("Actual Value", "@{actual_test}{0,0.0000}"),
            ("Predicted Value", "@{predicted_test}{0,0.0000}")
        ]
        hover_test.formatters = {"@index": "datetime"}

        p2.add_tools(hover_train)
        p3.add_tools(hover_test)
        output_notebook()
        return(show(row(p2, p3), notebook_handle=True))
    
    def update_config_mapping(self, folder_name="models_assets"):
        """
        Update the configuration mapping with model_id.
        
        Parameters:
            folder_name (str): The name of the folder where models are saved.
        """
        mapping_file_path = os.path.join(folder_name, 'config_mapping.json')
        if os.path.exists(mapping_file_path):
            with open(mapping_file_path, 'r') as f:
                existing_mappings = json.load(f)
        else:
            existing_mappings = {}

        model_id = self.generate_model_id()
        existing_mappings[model_id] = {
            'Model Class': self.__class__.__name__,
            'Config': self.config
        }

        # Save updated mappings
        with open(mapping_file_path, 'w') as f:
            json.dump(existing_mappings, f, indent=4)
        self.logging.info(f"Configuration mapping updated in {folder_name}")

    def save_model_to_folder(self, version, folder_name="models_assets"):
        """
        Save the model to a specified folder.
        
        Parameters:
            version (str): The version of the model.
            folder_name (str): The name of the folder where models are saved.
        """
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Update the config mapping
        self.update_config_mapping(folder_name)

        # Save the model
        model_id = self.generate_model_id()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{model_id}_V{version}_{self.__class__.__name__}.h5"
        full_path = os.path.join(folder_name, filename)
        self.model.save(full_path)
        self.logging.info(f"Model saved to {full_path}")

    def generate_model_id(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
        model_id = f"{self.model_type}_{config_hash}"
        self.logging.info(f"Generated model ID: {model_id}")
        return model_id

    def save_predictions(self, model_id, subfolder=None, overwrite=False):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        folder = 'model_predictions'
        if subfolder:
            folder = os.path.join(folder, subfolder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, 'DL_model_predictions.csv')
        
        df = self.test_comparison_df.reset_index()
        df['Model Class'] = self.__class__.__name__
        df['Model ID'] = model_id
        df['Config'] = json.dumps(self.config)
        df['Date Run'] = timestamp
        
        # Reorder the columns
        df = df[['Date Run', 'Model Class', 'Model ID', 'Config', 'Date', 'Actual', 'Predicted']]
        
        if overwrite or not os.path.exists(filepath):
            df.to_csv(filepath, index=False)
        else:
            df.to_csv(filepath, mode='a', header=False, index=False)
        self.logging.info(f"Predictions saved to {filepath}" if overwrite or not os.path.exists(filepath) else f"Predictions appended to {filepath}")

    def save_accuracy(self, model_id, subfolder=None, overwrite=False):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        folder = 'model_accuracy'
        if subfolder:
            folder = os.path.join(folder, subfolder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, 'DL_model_accuracy.csv')
        
        df = self.evaluation_df.reset_index()
        df['Model Class'] = self.__class__.__name__
        df['Model ID'] = model_id
        df['Config'] = json.dumps(self.config)
        df['Date Run'] = timestamp
        
        # Reorder the columns
        df = df[['Date Run', 'Model Class', 'Model ID', 'Config', 'index', 'RMSE', 'R2 Score', 'MAE', 'Explained Variance']]
        
        if overwrite or not os.path.exists(filepath):
            df.to_csv(filepath, index=False)
        else:
            df.to_csv(filepath, mode='a', header=False, index=False)
        self.logging.info(f"Accuracy metrics saved to {filepath}" if overwrite or not os.path.exists(filepath) else f"Accuracy metrics appended to {filepath}")


class LSTM_(BaseModelLSTM):
    def _initialize_model(self):
        self.model = Sequential()
        additional_params = {
            'input_shape': self.config['input_shape'],
            'num_lstm_layers': self.config['num_lstm_layers'],
            'lstm_units': self.config['lstm_units']
        }
        self.params.update(additional_params)
        
        for i in range(self.config['num_lstm_layers']):
            units = self.config['lstm_units'][i]
            return_sequences = True if i < self.config['num_lstm_layers'] - 1 else False
            self.model.add(LSTM(units, return_sequences=return_sequences))
            self.model.add(Dropout(self.config['dropout']))
        for units in self.config['dense_units']:
            self.model.add(Dense(units))
        self.model.compile(optimizer=self.config['optimizer'], loss='mean_squared_error')

class GRU_(BaseModelLSTM):
    def _initialize_model(self):
        self.model = Sequential()
        for i in range(self.config['num_gru_layers']):
            units = self.config['gru_units'][i]
            return_sequences = True if i < self.config['num_gru_layers'] - 1 else False
            self.model.add(GRU(units, return_sequences=return_sequences))
            self.model.add(Dropout(self.config['dropout']))
        for units in self.config['dense_units']:
            self.model.add(Dense(units))
        self.model.compile(optimizer=self.config['optimizer'], loss='mean_squared_error')

class Bi_LSTM(BaseModelLSTM):
    def _initialize_model(self):
        self.model = Sequential()
        for i in range(self.config['num_lstm_layers']):
            units = self.config['lstm_units'][i]
            return_sequences = True if i < self.config['num_lstm_layers'] - 1 else False
            self.model.add(Bidirectional(LSTM(units, return_sequences=return_sequences)))
            self.model.add(Dropout(self.config['dropout']))
        for units in self.config['dense_units']:
            self.model.add(Dense(units))
        self.model.compile(optimizer=self.config['optimizer'], loss='mean_squared_error')

class Bi_GRU(BaseModelLSTM):
    """
    This class is an implementation of a bi-directional GRU model for sequence prediction.
    It inherits from the BaseModelLSTM class and overrides the _initialize_model method.
    """
    def _initialize_model(self):
        self.model = Sequential()
        additional_params = {
            'input_shape': self.config['input_shape'],
            'num_gru_layers': self.config['num_gru_layers'],
            'gru_units': self.config['gru_units']
        }
        self.params.update(additional_params)

        for i in range(self.config['num_gru_layers']):
            units = self.config['gru_units'][i]
            return_sequences = True if i < self.config['num_gru_layers'] - 1 else False
            self.model.add(Bidirectional(GRU(units, return_sequences=return_sequences)))
            self.model.add(Dropout(self.config['dropout']))

        # If the last RNN layer returns sequences, you may need to flatten it
        if return_sequences:
            self.model.add(Flatten())
        
        for units in self.config['dense_units']:
            self.model.add(Dense(units))

        self.model.compile(optimizer=self.config['optimizer'], loss='mean_squared_error')

class Simple_RNN(BaseModelLSTM):
    """
    This class is an implementation of a Simple RNN model for sequence prediction.
    It inherits from the BaseModelLSTM class and overrides the _initialize_model method.
    """
    def _initialize_model(self):
        self.model = Sequential()
        additional_params = {
            'input_shape': self.config['input_shape'],
            'num_rnn_layers': self.config['num_rnn_layers'],
            'rnn_units': self.config['rnn_units']
        }
        self.params.update(additional_params)

        for i in range(self.config['num_rnn_layers']):
            units = self.config['rnn_units'][i]
            # Make sure to set return_sequences=False for the last layer
            return_sequences = True if i < self.config['num_rnn_layers'] - 1 else False
            self.model.add(SimpleRNN(units, return_sequences=return_sequences))
            self.model.add(Dropout(self.config['dropout']))

        # Add Dense layers
        for units in self.config['dense_units']:
            self.model.add(Dense(units))

        # Compile the model
        self.model.compile(optimizer=self.config['optimizer'], loss='mean_squared_error')

class Stacked_RNN(BaseModelLSTM):
    def _initialize_model(self):
        additional_params = {
            'input_shape': self.config['input_shape'],
            'lstm_units': self.config.get('lstm_units', []),
            'gru_units': self.config.get('gru_units', [])
        }
        self.params.update(additional_params)
        
        input_layer = Input(shape=self.config['input_shape'])
        x = input_layer

        # Adding LSTM layers
        for i, units in enumerate(self.config['lstm_units']):
            return_sequences = True if i < len(self.config['lstm_units']) - 1 or self.config['gru_units'] else False
            x = LSTM(units, return_sequences=return_sequences)(x)
            x = Dropout(self.config['dropout'])(x)

        # Adding GRU layers
        for i, units in enumerate(self.config['gru_units']):
            return_sequences = True if i < len(self.config['gru_units']) - 1 else False
            x = GRU(units, return_sequences=return_sequences)(x)
            x = Dropout(self.config['dropout'])(x)

        # Adding Dense layers
        for units in self.config['dense_units']:
            x = Dense(units)(x)
        
        self.model = Model(inputs=input_layer, outputs=x)
        self.model.compile(optimizer=self.config['optimizer'], loss='mean_squared_error')

class Attention_LSTM(BaseModelLSTM):
    """
    This class is an implementation of a LSTM model with Attention for sequence prediction.
    It inherits from the BaseModelLSTM class and overrides the _initialize_model method.
    """
    def _initialize_model(self):
        additional_params = {
            'input_shape': self.config['input_shape'],
            'num_lstm_layers': self.config['num_lstm_layers'],
            'lstm_units': self.config['lstm_units']
        }
        self.params.update(additional_params)
        input_layer = Input(shape=self.config['input_shape'])
        x = input_layer

        # Add LSTM layers
        for i in range(self.config['num_lstm_layers']):
            units = self.config['lstm_units'][i]
            return_sequences = True  # For Attention, the last LSTM layer should also return sequences
            x = LSTM(units, return_sequences=return_sequences)(x)
            x = Dropout(self.config['dropout'])(x)

        x = Attention(use_scale=True)([x, x])  # Self-attention
        x = GlobalAveragePooling1D()(x)
        for units in self.config['dense_units']:
            x = Dense(units)(x)
        
        self.model = Model(inputs=input_layer, outputs=x)
        self.model.compile(optimizer=self.config['optimizer'], loss='mean_squared_error')
    
class CNN_LSTM(BaseModelLSTM):
    def _initialize_model(self):
        self.model = Sequential()
        # Conv1D layers
        for i in range(self.config['num_conv_layers']):
            filters = self.config['conv_filters'][i]
            kernel_size = self.config['conv_kernel_size'][i]
            if i == 0:
                self.model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=self.config['input_shape']))
            else:
                self.model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
            self.model.add(MaxPooling1D(pool_size=2))
        
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Reshape((1, self.config['conv_filters'][-1])))    
        # LSTM layers
        for i in range(self.config['num_lstm_layers']):
            units = self.config['lstm_units'][i]
            return_sequences = True if i < self.config['num_lstm_layers'] - 1 else False
            self.model.add(LSTM(units, return_sequences=return_sequences))
            self.model.add(Dropout(self.config['dropout']))
        
        # Dense layers
        for units in self.config['dense_units']:
            self.model.add(Dense(units))
    
        self.model.compile(optimizer=self.config['optimizer'], loss='mean_squared_error')


models = {
    'DL_LSTM': {
        'class': LSTM_,  # Replace with your actual class
        'config': {
            'input_shape': (10, 5),
            'num_lstm_layers': 2,
            'lstm_units': [50, 30],
            'dropout': 0.2,
            'dense_units': [1],
            'optimizer': 'adam'
        },
        'skip': False
    },
    'DL_BiLSTM': {
        'class': Bi_LSTM,  # Replace with your actual class
        'config': {
            'num_lstm_layers': 2,
            'lstm_units': [50, 30],
            'dropout': 0.2,
            'dense_units': [1],
            'optimizer': 'adam'
        },
        'skip': False
    },
    'DL_GRU': {
        'class': GRU_,  # Replace with your actual class
        'config': {
            'num_gru_layers': 2,
            'gru_units': [50, 30],
            'dropout': 0.2,
            'dense_units': [1],
            'optimizer': 'adam'
        },
        'skip': False
    },
    'DL_BiGRU': {
        'class': Bi_GRU,  # Replace with your actual class
        'config': {
            'input_shape': (10, 30),
            'num_gru_layers': 2,
            'gru_units': [50, 30],
            'dense_units': [1],
            'dropout': 0.2,
            'optimizer': 'adam'
        },
        'skip': False
    },
    'DL_SimpleRNN': {  
        'class': Simple_RNN,  # Replace with your actual class
        'config': {
            'input_shape': (10, 30),
            'num_rnn_layers': 2,
            'rnn_units': [50, 30],
            'dense_units': [1],
            'dropout': 0.2,
            'optimizer': 'adam'
        },
        'skip': False
    },
    'DL_StackedRNN': {
        'class': Stacked_RNN,  # Replace with your actual class
        'config': {
            'input_shape': (10, 5),
            'lstm_units': [50, 30],
            'gru_units': [20],
            'dropout': 0.2,
            'dense_units': [1],
            'optimizer': 'adam'
        },
        'skip': False
    },
    'DL_AttentionLSTM': {
        'class': Attention_LSTM,  # Replace with your actual class
        'config': {
            'input_shape': (10, 5),
            'num_lstm_layers': 2,
            'lstm_units': [50, 30],
            'dropout': 0.2,
            'dense_units': [1],
            'optimizer': 'adam'
        },
        'skip': False
    },
    'DL_CNNLSTM': {
        'class': CNN_LSTM,
        'config': {
            'input_shape': (10, 5),
            'num_conv_layers': 2,  # Increased the number of convolution layers
            'conv_filters': [64, 32],  # Additional filter
            'conv_kernel_size': [3, 2],  # Additional kernel size
            'num_lstm_layers': 1,
            'lstm_units': [50],
            'dropout': 0.2,
            'dense_units': [1],
            'optimizer': 'adam'
        },
        'skip': False
    },
}



models1 = {
    'LSTM_Original': {
        'class': LSTM,
        'config': {
            'input_shape': (10, 5),
            'num_lstm_layers': 2,
            'lstm_units': [50, 30],
            'dropout': 0.2,
            'dense_units': [1],
            'optimizer': 'adam'
        },
        'skip': False
    },
    'LSTM_SingleLayer_NoDropout': {
        'class': LSTM,
        'config': {
            'input_shape': (10, 5),
            'num_lstm_layers': 1,
            'lstm_units': [64],
            'dropout': 0.0,
            'dense_units': [1],
            'optimizer': 'adam'
        },
        'skip': False
    },
    'LSTM_ThreeLayer_VariedUnits': {
        'class': LSTM,
        'config': {
            'input_shape': (10, 5),
            'num_lstm_layers': 3,
            'lstm_units': [64, 32, 16],
            'dropout': 0.3,
            'dense_units': [1],
            'optimizer': 'adam'
        },
        'skip': False
    },
    'LSTM_TwoLayer_ExtraDense': {
        'class': LSTM,
        'config': {
            'input_shape': (10, 5),
            'num_lstm_layers': 2,
            'lstm_units': [50, 30],
            'dropout': 0.2,
            'dense_units': [16, 1],
            'optimizer': 'adam'
        },
        'skip': False
    }
}


def run_models(models, run_only=None, skip=None):
    for name, model_info in models.items():
        if run_only and name not in run_only:
            continue
        if skip and name in skip:
            continue
        if model_info.get('skip'):
            continue

        model_class = model_info['class']
        config = model_info['config']
        
        model = model_class(data_preprocessor=data_preprocessor, config=config, model_type=name)
        model.train_model(epochs=100, batch_size=32)
        model.make_predictions()
        evaluation_df = model.evaluate_model()
        display(evaluation_df)
        model.plot_history(plot=False)
        model.plot_predictions(plot=True)

        # Generate a unique model_id for this run
        model_id = model.generate_model_id()
        model.save_predictions(model_id, subfolder='model_deep_learning', overwrite=False)
        model.save_accuracy(model_id, subfolder='model_deep_learning', overwrite=False)
        model.save_model_to_folder(version="1")




# Run all models
#run_models(models)
run_models(models, run_only=['DL_LSTM'])
#run_models(models, skip=['SimpleRNN'])



from IPython.display import display, HTML
import json, joblib, hashlib, logging, os, warnings
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from joblib import dump, load
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_notebook, save
from bokeh.models import (HoverTool, ColumnDataSource, WheelZoomTool, Span, Range1d,
                          FreehandDrawTool, MultiLine, NumeralTickFormatter, Button, CustomJS)
from bokeh.layouts import column, row
from bokeh.io import curdoc, export_png
from bokeh.models.widgets import CheckboxGroup
from bokeh.themes import Theme
# Machine Learning Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
import xgboost as xgb



# Other settings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.3f}'.format)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class BaseModel_ML:
    """
    A base class for machine learning models.
    This class handles data preprocessing, model training, predictions, and evaluations.
    """
    def __init__(self, model_type, data_preprocessor, config):
        self.model_type = model_type
        self._validate_input(data_preprocessor.X_train, data_preprocessor.y_train, data_preprocessor.X_test, data_preprocessor.y_test)
        self.X_train = data_preprocessor.X_train
        self.y_train = data_preprocessor.y_train
        self.X_test = data_preprocessor.X_test
        self.y_test = data_preprocessor.y_test
        self.feature_scaler = data_preprocessor.scalers['features']
        self.target_scaler = data_preprocessor.scalers['target']
        self.data = data_preprocessor.data
        self.config = config
        self.params = {'model_type': model_type}
        self.params.update(config)  # Add other config parameters to the params dictionary
        self._initialize_model()
        self.logging = logging.getLogger(f"{self.model_type}_model")
    
    def _validate_input(self, X_train, y_train, X_test, y_test):
        """Validate the shape and type of training and testing data."""
        for arr, name in [(X_train, 'X_train'), (y_train, 'y_train'), (X_test, 'X_test'), (y_test, 'y_test')]:
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"{name} should be a numpy array.")
            
            if len(arr.shape) != 2:
                raise ValueError(f"{name} should be a 2D numpy array for ML models. Found shape {arr.shape}.")

    def train_model(self, cross_val=False, n_splits=5):
        logging.info(f"Training {self.model_type} model")

        if cross_val:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            fold_no = 1
            for train_index, val_index in tscv.split(self.X_train):
                logging.info(f"Training on fold {fold_no}")
                # Split the data into training and validation sets for this fold
                X_train_fold, X_val_fold = self.X_train[train_index], self.X_train[val_index]
                y_train_fold, y_val_fold = self.y_train[train_index], self.y_train[val_index]

                self.model.fit(X_train_fold, y_train_fold)
                val_score = self.model.score(X_val_fold, y_val_fold)
                logging.info(f"Validation score for fold {fold_no}: {val_score}")

                fold_no += 1
        else:
            self.model.fit(self.X_train, self.y_train)

        logging.info("Training completed")

    def make_predictions(self):
        logging.info("Making predictions")

        self._make_raw_predictions()
        self._make_unscaled_predictions()
        self._create_comparison_dfs()

        logging.info("Predictions made")

    def _make_raw_predictions(self):
        self.train_predictions = self.model.predict(self.X_train)
        self.test_predictions = self.model.predict(self.X_test)
        logging.info(f"Raw predictions made with shapes train: {self.train_predictions.shape}, test: {self.test_predictions.shape}")

    def _make_unscaled_predictions(self):
        # Check if the shape of the predictions matches that of y_train and y_test
        if self.train_predictions.ndim > 1 and self.train_predictions.shape[1] != 1:
            logging.error(f"Unexpected number of columns in train_predictions: {self.train_predictions.shape[1]}")
            return

        if self.test_predictions.ndim > 1 and self.test_predictions.shape[1] != 1:
            logging.error(f"Unexpected number of columns in test_predictions: {self.test_predictions.shape[1]}")
            return

        # If predictions are 2D, flatten to 1D
        if self.train_predictions.ndim == 2:
            self.train_predictions = self.train_predictions.flatten()

        if self.test_predictions.ndim == 2:
            self.test_predictions = self.test_predictions.flatten()

        # Perform the inverse transformation to get unscaled values if required
        if self.target_scaler:
            self.train_predictions = self.target_scaler.inverse_transform(self.train_predictions.reshape(-1, 1)).flatten()
            self.test_predictions = self.target_scaler.inverse_transform(self.test_predictions.reshape(-1, 1)).flatten()

        logging.info(f"Unscaled predictions made with shapes train: {self.train_predictions.shape}, test: {self.test_predictions.shape}")

    def _create_comparison_dfs(self):
        # Check if the target scaler was used and inverse transform if necessary
        if self.target_scaler:
            y_train_flat = self.target_scaler.inverse_transform(self.y_train.reshape(-1, 1)).flatten()
            y_test_flat = self.target_scaler.inverse_transform(self.y_test.reshape(-1, 1)).flatten()
        else:
            y_train_flat = self.y_train
            y_test_flat = self.y_test

        # Obtain date indices from original data
        train_date_index = self.data.index[:len(y_train_flat)]
        test_date_index = self.data.index[-len(y_test_flat):]

        if y_train_flat.shape != self.train_predictions.shape:
            logging.error(f"Shape mismatch between y_train {y_train_flat.shape} and train_predictions {self.train_predictions.shape}")
        else:
            self.train_comparison_df = pd.DataFrame({'Actual': y_train_flat, 'Predicted': self.train_predictions})
            # Set date index for train_comparison_df
            self.train_comparison_df.set_index(train_date_index, inplace=True)

        if y_test_flat.shape != self.test_predictions.shape:
            logging.error(f"Shape mismatch between y_test {y_test_flat.shape} and test_predictions {self.test_predictions.shape}")
        else:
            self.test_comparison_df = pd.DataFrame({'Actual': y_test_flat, 'Predicted': self.test_predictions})
            # Set date index for test_comparison_df
            self.test_comparison_df.set_index(test_date_index, inplace=True)

    def evaluate_model(self):
            logging.info("Evaluating LSTM model")
            metrics = {'RMSE': mean_squared_error, 'R2 Score': r2_score,
                       'MAE': mean_absolute_error, 'Explained Variance': explained_variance_score}

            evaluation = {}
            for name, metric in metrics.items():
                if name == 'RMSE':
                    train_evaluation = metric(self.train_comparison_df['Actual'],
                                              self.train_comparison_df['Predicted'],
                                              squared=False)
                    test_evaluation = metric(self.test_comparison_df['Actual'],
                                             self.test_comparison_df['Predicted'],
                                             squared=False)
                else:
                    train_evaluation = metric(self.train_comparison_df['Actual'],
                                              self.train_comparison_df['Predicted'])
                    test_evaluation = metric(self.test_comparison_df['Actual'],
                                             self.test_comparison_df['Predicted'])
                evaluation[name] = {'Train': train_evaluation, 'Test': test_evaluation}

            self.evaluation_df = pd.DataFrame(evaluation)
            logging.info("Evaluation completed")
            return self.evaluation_df
    
    def plot_predictions(self, plot=True):
        if not plot:
            return        
        if not hasattr(self, 'train_comparison_df') or not hasattr(self, 'test_comparison_df'):
            print("No predictions are available. Generate predictions first.")
            return
        actual_train = self.train_comparison_df['Actual']
        predicted_train = self.train_comparison_df['Predicted']
        actual_test = self.test_comparison_df['Actual']
        predicted_test = self.test_comparison_df['Predicted']
        index_train = self.train_comparison_df.index
        index_test = self.test_comparison_df.index

        # Preparing data
        source_train = ColumnDataSource(data=dict(
            index=index_train,
            actual_train=actual_train,
            predicted_train=predicted_train
        ))

        source_test = ColumnDataSource(data=dict(
            index=index_test,
            actual_test=actual_test,
            predicted_test=predicted_test
        ))

        p2 = figure(width=700, height=600, title="Training Data: Actual vs Predicted", x_axis_label='Date', y_axis_label='Value', x_axis_type="datetime")
        p3 = figure(width=700, height=600, title="Testing Data: Actual vs Predicted",x_axis_label='Date', y_axis_label='Value', x_axis_type="datetime")
        p2.line(x='index', y='actual_train', legend_label="Actual", line_width=2, source=source_train, color="green")
        p2.line(x='index', y='predicted_train', legend_label="Predicted", line_width=2, source=source_train, color="red")
        p3.line(x='index', y='actual_test', legend_label="Actual", line_width=2, source=source_test, color="green")
        p3.line(x='index', y='predicted_test', legend_label="Predicted", line_width=2, source=source_test, color="red")
        p2.legend.location = "top_left" 
        p2.legend.click_policy = "hide"
        p3.legend.location = "top_left" 
        p3.legend.click_policy = "hide"
        hover_train = HoverTool()
        hover_train.tooltips = [
            ("Date", "@index{%F}"),
            ("Actual Value", "@{actual_train}{0,0.0000}"),
            ("Predicted Value", "@{predicted_train}{0,0.0000}")
        ]
        hover_train.formatters = {"@index": "datetime"}

        hover_test = HoverTool()
        hover_test.tooltips = [
            ("Date", "@index{%F}"),
            ("Actual Value", "@{actual_test}{0,0.0000}"),
            ("Predicted Value", "@{predicted_test}{0,0.0000}")
        ]
        hover_test.formatters = {"@index": "datetime"}

        p2.add_tools(hover_train)
        p3.add_tools(hover_test)
        output_notebook()
        return(show(row(p2, p3), notebook_handle=True))
    
    def update_config_mapping(self, folder_name="models_assets"):
        """
        Update the configuration mapping with model_id.
        
        Parameters:
            folder_name (str): The name of the folder where models are saved.
        """
        mapping_file_path = os.path.join(folder_name, 'config_mapping.json')
        if os.path.exists(mapping_file_path):
            with open(mapping_file_path, 'r') as f:
                existing_mappings = json.load(f)
        else:
            existing_mappings = {}

        model_id = self.generate_model_id()
        existing_mappings[model_id] = {
            'Model Class': self.__class__.__name__,
            'Config': self.config
        }

        # Save updated mappings
        with open(mapping_file_path, 'w') as f:
            json.dump(existing_mappings, f, indent=4)
        self.logging.info(f"Configuration mapping updated in {folder_name}")

    def save_model_to_folder(self, version, folder_name="models_assets"):
        """
        Save the model to a specified folder.
        
        Parameters:
            version (str): The version of the model.
            folder_name (str): The name of the folder where models are saved.
        """
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Update the config mapping
        self.update_config_mapping(folder_name)

        # Generate a filename
        model_id = self.generate_model_id()
        filename = f"{model_id}_V{version}_{self.__class__.__name__}.joblib"
        full_path = os.path.join(folder_name, filename)

        # Serialize and save the model
        joblib.dump(self.model, full_path)
        self.logging.info(f"Model saved to {full_path}")

    def generate_model_id(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
        model_id = f"{self.model_type}_{config_hash}"
        self.logging.info(f"Generated model ID: {model_id}")
        return model_id

    def save_predictions(self, model_id, subfolder=None, overwrite=False):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        folder = 'model_predictions'
        if subfolder:
            folder = os.path.join(folder, subfolder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, 'ML_model_predictions.csv')
        
        df = self.test_comparison_df.reset_index()
        df['Model Class'] = self.__class__.__name__
        df['Model ID'] = model_id
        df['Config'] = json.dumps(self.config)
        df['Date Run'] = timestamp
        
        # Reorder the columns
        df = df[['Date Run', 'Model Class', 'Model ID', 'Config', 'Date', 'Actual', 'Predicted']]
        
        if overwrite or not os.path.exists(filepath):
            df.to_csv(filepath, index=False)
        else:
            df.to_csv(filepath, mode='a', header=False, index=False)
        self.logging.info(f"Predictions saved to {filepath}" if overwrite or not os.path.exists(filepath) else f"Predictions appended to {filepath}")

    def save_accuracy(self, model_id, subfolder=None, overwrite=False):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        folder = 'model_accuracy'
        if subfolder:
            folder = os.path.join(folder, subfolder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, 'ML_model_accuracy.csv')
        
        df = self.evaluation_df.reset_index()
        df['Model Class'] = self.__class__.__name__
        df['Model ID'] = model_id
        df['Config'] = json.dumps(self.config)
        df['Date Run'] = timestamp
        
        # Reorder the columns
        df = df[['Date Run', 'Model Class', 'Model ID', 'Config', 'index', 'RMSE', 'R2 Score', 'MAE', 'Explained Variance']]
        
        if overwrite or not os.path.exists(filepath):
            df.to_csv(filepath, index=False)
        else:
            df.to_csv(filepath, mode='a', header=False, index=False)
        self.logging.info(f"Accuracy metrics saved to {filepath}" if overwrite or not os.path.exists(filepath) else f"Accuracy metrics appended to {filepath}")




class Linear_Regression(BaseModel_ML):
    """
    Enhanced Linear Regression model supporting Ridge and Lasso regularization.
    Inherits from BaseModel_ML.
    """
    def _initialize_model(self):
        # Set up the regression model based on the configuration
        regularization = self.config.get('regularization', 'none').lower()
        alpha = self.config.get('alpha', 1.0)
        if regularization == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif regularization == 'lasso':
            self.model = Lasso(alpha=alpha)
        else:
            self.model = LinearRegression()
        
        # Log the initialization with model type and regularization type
        logging.info(f"{self.model_type} model initialized with {regularization} regularization and alpha={alpha}")

        # Update params with specific parameters of the regression model
        self.params.update({'regularization': regularization, 'alpha': alpha})
    
class XGBoost(BaseModel_ML):
    """
    Enhanced XGBoost model that inherits from BaseModel_ML.
    """
    def _initialize_model(self):
        """
        Initialize the XGBoost model with parameters from the config.
        """
        # Set up the XGBoost model based on the configuration
        self.model = xgb.XGBRegressor(**self.config)
        
        logging.info(f"{self.model_type} model initialized with configuration: {self.config}")
        self.params.update(self.config)

class LightGBM(BaseModel_ML):
    """
    Enhanced LightGBM model that inherits from BaseModel_ML.
    """
    def _initialize_model(self):
        # Initialize LightGBM model with config parameters
        self.model = LGBMRegressor(**self.config)
        logging.info(f"{self.model_type} model initialized with configuration: {self.config}")
        self.params.update(self.config)

class SVM(BaseModel_ML):
    """
    Enhanced SVM model that inherits from BaseModel_ML.
    """
    def _initialize_model(self):
        # Initialize SVM model with config parameters
        self.model = SVR(**self.config)
        logging.info(f"{self.model_type} model initialized with configuration: {self.config}")
        self.params.update(self.config)

class SVRegressor(BaseModel_ML):
    """
    Enhanced SVR model that inherits from BaseModel_ML.
    """
    def _initialize_model(self):
        # Initialize SVR model with config parameters
        self.model = SVR(**self.config)
        logging.info(f"{self.model_type} model initialized with configuration: {self.config}")
        self.params.update(self.config)

class KNN(BaseModel_ML):
    """
    Enhanced KNN model that inherits from BaseModel_ML.
    """
    def _initialize_model(self):
        # Initialize KNN model with config parameters
        self.model = KNeighborsRegressor(**self.config)
        logging.info(f"{self.model_type} model initialized with configuration: {self.config}")
        self.params.update(self.config)

class RandomForest(BaseModel_ML):
    """
    Enhanced Random Forest model that inherits from BaseModel_ML.
    """
    def _initialize_model(self):
        # Initialize Random Forest model with config parameters
        self.model = RandomForestRegressor(**self.config)
        logging.info(f"{self.model_type} model initialized with configuration: {self.config}")
        self.params.update(self.config)

    # The feature_importance method can be kept as is if it provides additional functionality specific to Random Forest.
    def feature_importance(self):
        """
        Extract feature importance scores.
        """
        try:
            importance_scores = self.model.feature_importances_
            logging.info("Feature importance scores extracted.")
            return importance_scores
        except Exception as e:
            logging.error(f"Error occurred while extracting feature importance: {str(e)}")

class ExtraTrees(BaseModel_ML):
    """
    Enhanced Extra Trees model that inherits from BaseModel_ML.
    """
    def _initialize_model(self):
        # Initialize Extra Trees model with config parameters
        self.model = ExtraTreesRegressor(**self.config)
        logging.info(f"{self.model_type} model initialized with configuration: {self.config}")
        self.params.update(self.config)


from data_fetcher import btc_data
from data_preprocessor import UnifiedDataPreprocessor
df = btc_data.copy()

data_preprocessor = UnifiedDataPreprocessor(df, target_column='Close')
data_preprocessor.split_and_plot_data(test_size=0.2, plot=False)
data_preprocessor.normalize_data(scaler_type='MinMax', plot=False)
data_preprocessor.normalize_target(scaler_type='MinMax', plot=False)


models = {
    'ML_LR': {
        'class': Linear_Regression,
        'config': {
            'regularization': 'ridge',
            'alpha': 1.0
        },
        'skip': False
    },
    'ML_XGBoost': {
        'class': XGBoost,
        'config': {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 5
        },
        'skip': False
    },
    'ML_LightGBM': {
        'class': LightGBM,
        'config': {
            'objective': 'regression',
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 5
        },
        'skip': False
    },
    'ML_SVM': {
        'class': SVM,
        'config': {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1
        },
        'skip': False
    },
    'ML_SVRegressor': {
        'class': SVRegressor,
        'config': {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1
        },
        'skip': False
    },
    'ML_KNN': {
        'class': KNN,
        'config': {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto'
        },
        'skip': False
    },
    'ML_RandomForest': {
        'class': RandomForest,
        'config': {
            'n_estimators': 100,
            'criterion': 'poisson',
            'max_depth': None
        },
        'skip': False
    },
    'ML_ExtraTrees': {
        'class': ExtraTrees,
        'config': {
            'n_estimators': 100,
            'criterion': 'squared_error',
            'max_depth': None
        },
        'skip': False
    }
}

def run_models(models, run_only=None, skip=None):
    for name, model_info in models.items():
        if run_only and name not in run_only:
            continue
        if skip and name in skip:
            continue
        if model_info.get('skip'):
            continue

        model_class = model_info['class']
        config = model_info['config']
        
        model = model_class(data_preprocessor=data_preprocessor, config=config, model_type=name)
        model.train_model()
        model.make_predictions()
        evaluation_df = model.evaluate_model()
        display(evaluation_df)
        model.plot_predictions(plot=True)
                
        # Generate a unique model_id for this run
        model_id = model.generate_model_id()
        model.save_predictions(model_id, subfolder='model_machine_learning', overwrite=False)
        model.save_accuracy(model_id, subfolder='model_machine_learning', overwrite=False)
        model.save_model_to_folder(version="1")


# Run all models
run_models(models)

# Run only specific models
#run_models(models, run_only=['Baby_Cow', 'Baby_coooow'])

# Skip specific models
#run_models(models, data_preprocessor, skip=['Linear_Regression'])









import numpy as np
import pandas as pd
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from joblib import dump, load
from tensorflow.keras.models import save_model
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_notebook, save
from bokeh.models import (HoverTool, ColumnDataSource, WheelZoomTool, Span, Range1d,
                          FreehandDrawTool, MultiLine, NumeralTickFormatter, Button, CustomJS)
from bokeh.layouts import column, row
from bokeh.io import curdoc, export_png
from bokeh.models.widgets import CheckboxGroup
from bokeh.themes import Theme
# Machine Learning Libraries
import sklearn
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, accuracy_score

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, LSTM, TimeDistributed, Conv1D, MaxPooling1D, Flatten,
                                    ConvLSTM2D, BatchNormalization, GRU, Bidirectional, Attention, Input,
                                    Reshape, GlobalAveragePooling1D, GlobalMaxPooling1D, Lambda, LayerNormalization, 
                                    SimpleRNN, Layer, Multiply, Add, Activation)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1, l2, l1_l2
from keras_tuner import HyperModel, RandomSearch, BayesianOptimization
from tcn import TCN
from kerasbeats import NBeatsModel
from typing import List, Optional
from typing import Optional, List, Tuple
from statsmodels.tsa.stattools import acf, pacf


# Other settings
from IPython.display import display, HTML
import os, warnings, logging
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.3f}'.format)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


# LSTM Sequece-to-One
from data_preprocessor import UnifiedDataPreprocessor
from data_fetcher import btc_data
df = btc_data.copy()
data_preprocessor = UnifiedDataPreprocessor(df, target_column='Close')
data_preprocessor.split_and_plot_data(test_size=0.2, plot=False)
data_preprocessor.normalize_data(scaler_type='MinMax',plot=False)
data_preprocessor.normalize_target(scaler_type='MinMax',plot=False)
n_steps = 10 
X_train_seq, y_train_seq, X_test_seq, y_test_seq = data_preprocessor.prepare_data_for_recurrent(n_steps, seq_to_seq=False)



class BaseModel_DL_SOA():
    """
    A base class for LSTM-like machine learning models.
    This class handles data preprocessing, model training, predictions, and evaluations.
    """
    def __init__(self, model_type, data_preprocessor, config, cross_val=False):
        self._validate_input_sequence(data_preprocessor.X_train_seq, data_preprocessor.y_train_seq, data_preprocessor.X_test_seq, data_preprocessor.y_test_seq)
        self.X_train = data_preprocessor.X_train_seq
        self.y_train = data_preprocessor.y_train_seq
        self.X_test = data_preprocessor.X_test_seq
        self.y_test = data_preprocessor.y_test_seq
        self.feature_scaler = data_preprocessor.scalers['features']
        self.target_scaler = data_preprocessor.scalers['target']
        self.data = data_preprocessor.data
        self.config = config
        self.cross_val = cross_val
        self.model_type = model_type
        self.params = {'model_type': model_type}
        self.params.update(config)
        self.logger = logging.getLogger(__name__)
    
    def _validate_input_sequence(self, X_train, y_train, X_test, y_test):
        """Validate the shape and type of training and testing sequence data."""
        for arr, name in [(X_train, 'X_train_seq'), (y_train, 'y_train_seq'), (X_test, 'X_test_seq'), (y_test, 'y_test_seq')]:
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"{name} should be a numpy array.")
            if len(arr.shape) < 2:
                raise ValueError(f"{name} should have at least two dimensions.")
            if 'X_' in name and len(arr.shape) != 3:
                raise ValueError(f"{name} should be a 3D numpy array for sequence models. Found shape {arr.shape}.")

    def make_predictions(self):
        logging.info("Making predictions")

        self._make_raw_predictions()
        self._make_unscaled_predictions()
        self._create_comparison_dfs()

        logging.info("Predictions made")

    def _make_raw_predictions(self):
        self.train_predictions = self.model.predict(self.X_train)
        self.test_predictions = self.model.predict(self.X_test)
        logging.info(f"Raw predictions made with shapes train: {self.train_predictions.shape}, test: {self.test_predictions.shape}")

    def _make_unscaled_predictions(self):
        # Check if the shape of the predictions matches that of y_train and y_test
        if self.train_predictions.shape[:-1] != self.y_train.shape[:-1]:
            logging.error(f"Shape mismatch: train_predictions {self.train_predictions.shape} vs y_train {self.y_train.shape}")
            return

        if self.test_predictions.shape[:-1] != self.y_test.shape[:-1]:
            logging.error(f"Shape mismatch: test_predictions {self.test_predictions.shape} vs y_test {self.y_test.shape}")
            return

        # If predictions are 3D, reduce dimensionality by taking mean along last axis
        if self.train_predictions.ndim == 3:
            self.train_predictions = np.mean(self.train_predictions, axis=-1)

        if self.test_predictions.ndim == 3:
            self.test_predictions = np.mean(self.test_predictions, axis=-1)

        # Perform the inverse transformation to get unscaled values
        self.train_predictions = self.target_scaler.inverse_transform(self.train_predictions).flatten()
        self.test_predictions = self.target_scaler.inverse_transform(self.test_predictions).flatten()

        logging.info(f"Unscaled predictions made with shapes train: {self.train_predictions.shape}, test: {self.test_predictions.shape}")

    def _create_comparison_dfs(self):
        y_train_flat = self.target_scaler.inverse_transform(self.y_train).flatten()
        y_test_flat = self.target_scaler.inverse_transform(self.y_test).flatten()

        # Obtain date indices from original data
        train_date_index = self.data.index[:len(self.y_train)]
        test_date_index = self.data.index[-len(self.y_test):]

        if y_train_flat.shape != self.train_predictions.shape:
            logging.error(f"Shape mismatch between y_train {y_train_flat.shape} and train_predictions {self.train_predictions.shape}")
        else:
            self.train_comparison_df = pd.DataFrame({'Actual': y_train_flat, 'Predicted': self.train_predictions})
            # Set date index for train_comparison_df
            self.train_comparison_df.set_index(train_date_index, inplace=True)

        if y_test_flat.shape != self.test_predictions.shape:
            logging.error(f"Shape mismatch between y_test {y_test_flat.shape} and test_predictions {self.test_predictions.shape}")
        else:
            self.test_comparison_df = pd.DataFrame({'Actual': y_test_flat, 'Predicted': self.test_predictions})
            # Set date index for test_comparison_df
            self.test_comparison_df.set_index(test_date_index, inplace=True)
            
    def evaluate_model(self):
            logging.info("Evaluating SOA models")
            metrics = {'RMSE': mean_squared_error, 'R2 Score': r2_score,
                       'MAE': mean_absolute_error, 'Explained Variance': explained_variance_score}

            evaluation = {}
            for name, metric in metrics.items():
                if name == 'RMSE':
                    train_evaluation = metric(self.train_comparison_df['Actual'],
                                              self.train_comparison_df['Predicted'],
                                              squared=False)
                    test_evaluation = metric(self.test_comparison_df['Actual'],
                                             self.test_comparison_df['Predicted'],
                                             squared=False)
                else:
                    train_evaluation = metric(self.train_comparison_df['Actual'],
                                              self.train_comparison_df['Predicted'])
                    test_evaluation = metric(self.test_comparison_df['Actual'],
                                             self.test_comparison_df['Predicted'])
                evaluation[name] = {'Train': train_evaluation, 'Test': test_evaluation}

            self.evaluation_df = pd.DataFrame(evaluation)
            logging.info("Evaluation completed")
            return self.evaluation_df

    def plot_history(self, plot=True):
        if not plot:
            return
        if not hasattr(self, 'history'):
            print("No training history is available. Train model first.")
            return
        # Extracting loss data from training history
        train_loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = list(range(1, len(train_loss) + 1))
        # Preparing data
        source = ColumnDataSource(data=dict(
            epochs=epochs,
            train_loss=train_loss,
            val_loss=val_loss
        ))

        p1 = figure(width=700, height=600, title="Training Loss over Epochs",x_axis_label='Epochs', y_axis_label='Loss')
        hover1 = HoverTool()
        hover1.tooltips = [("Epoch", "@epochs"), ("Loss", "@{train_loss}{0,0.0000}")]
        p1.add_tools(hover1)
        hover2 = HoverTool()
        hover2.tooltips = [("Epoch", "@epochs"), ("Validation Loss", "@{val_loss}{0,0.0000}")]
        p1.add_tools(hover2)
        p1.line(x='epochs', y='train_loss', legend_label="Training Loss", line_width=2, source=source, color="green")
        p1.line(x='epochs', y='val_loss', legend_label="Validation Loss", line_width=2, source=source, color="red")
        p1.legend.location = "top_right"
        p1.legend.click_policy = "hide"

        output_notebook()
        return(show(p1, notebook_handle=True))

    def plot_predictions(self, plot=True):
        if not plot:
            return        
        if not hasattr(self, 'train_comparison_df') or not hasattr(self, 'test_comparison_df'):
            print("No predictions are available. Generate predictions first.")
            return
        actual_train = self.train_comparison_df['Actual']
        predicted_train = self.train_comparison_df['Predicted']
        actual_test = self.test_comparison_df['Actual']
        predicted_test = self.test_comparison_df['Predicted']
        index_train = self.train_comparison_df.index
        index_test = self.test_comparison_df.index

        # Preparing data
        source_train = ColumnDataSource(data=dict(
            index=index_train,
            actual_train=actual_train,
            predicted_train=predicted_train
        ))

        source_test = ColumnDataSource(data=dict(
            index=index_test,
            actual_test=actual_test,
            predicted_test=predicted_test
        ))

        p2 = figure(width=700, height=600, title="Training Data: Actual vs Predicted", x_axis_label='Date', y_axis_label='Value', x_axis_type="datetime")
        p3 = figure(width=700, height=600, title="Testing Data: Actual vs Predicted",x_axis_label='Date', y_axis_label='Value', x_axis_type="datetime")
        p2.line(x='index', y='actual_train', legend_label="Actual", line_width=2, source=source_train, color="green")
        p2.line(x='index', y='predicted_train', legend_label="Predicted", line_width=2, source=source_train, color="red")
        p3.line(x='index', y='actual_test', legend_label="Actual", line_width=2, source=source_test, color="green")
        p3.line(x='index', y='predicted_test', legend_label="Predicted", line_width=2, source=source_test, color="red")
        p2.legend.location = "top_left" 
        p2.legend.click_policy = "hide"
        p3.legend.location = "top_left" 
        p3.legend.click_policy = "hide"
        hover_train = HoverTool()
        hover_train.tooltips = [
            ("Date", "@index{%F}"),
            ("Actual Value", "@{actual_train}{0,0.0000}"),
            ("Predicted Value", "@{predicted_train}{0,0.0000}")
        ]
        hover_train.formatters = {"@index": "datetime"}

        hover_test = HoverTool()
        hover_test.tooltips = [
            ("Date", "@index{%F}"),
            ("Actual Value", "@{actual_test}{0,0.0000}"),
            ("Predicted Value", "@{predicted_test}{0,0.0000}")
        ]
        hover_test.formatters = {"@index": "datetime"}

        p2.add_tools(hover_train)
        p3.add_tools(hover_test)
        output_notebook()
        return(show(row(p2, p3), notebook_handle=True))
    
    def update_config_mapping(self, folder_name="models_assets"):
        """
        Update the configuration mapping with model_id.
        
        Parameters:
            folder_name (str): The name of the folder where models are saved.
        """
        mapping_file_path = os.path.join(folder_name, 'config_mapping.json')
        if os.path.exists(mapping_file_path):
            with open(mapping_file_path, 'r') as f:
                existing_mappings = json.load(f)
        else:
            existing_mappings = {}

        model_id = self.generate_model_id()
        existing_mappings[model_id] = {
            'Model Class': self.__class__.__name__,
            'Config': self.config
        }

        # Save updated mappings
        with open(mapping_file_path, 'w') as f:
            json.dump(existing_mappings, f, indent=4)
        self.logger.info(f"Configuration mapping updated in {folder_name}")

    def save_model_to_folder(self, version, folder_name="models_assets"):
        """
        Save the model to a specified folder.
        
        Parameters:
            version (str): The version of the model.
            folder_name (str): The name of the folder where models are saved.
        """
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Update the config mapping
        self.update_config_mapping(folder_name)

        # Save the model
        model_id = self.generate_model_id()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{model_id}_V{version}_{self.__class__.__name__}.h5"
        full_path = os.path.join(folder_name, filename)
        self.save(full_path)
        self.logger.info(f"Model saved to {full_path}")

    def save(self, path):
        try:
            # Try saving normally for Keras models
            save_model(self.model, path)
            logger.info(f"Model saved to {path}")
        except AttributeError:
            try:
                # Special case for N-BEATS
                save_model(self.model.model, path)
                logger.info(f"Internal Keras model saved to {path}")
            except Exception as e:
                logger.error(f"Failed to save the model. Error: {e}")

    def generate_model_id(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
        model_id = f"{self.model_type}_{config_hash}"
        self.logger.info(f"Generated model ID: {model_id}")
        return model_id

    def save_predictions(self, model_id, subfolder=None, overwrite=False):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        folder = 'model_predictions'
        if subfolder:
            folder = os.path.join(folder, subfolder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, 'SOA_model_predictions.csv')
        
        df = self.test_comparison_df.reset_index()
        df['Model Class'] = self.__class__.__name__
        df['Model ID'] = model_id
        df['Config'] = json.dumps(self.config)
        df['Date Run'] = timestamp
        
        # Reorder the columns
        df = df[['Date Run', 'Model Class', 'Model ID', 'Config', 'Date', 'Actual', 'Predicted']]
        
        if overwrite or not os.path.exists(filepath):
            df.to_csv(filepath, index=False)
        else:
            df.to_csv(filepath, mode='a', header=False, index=False)
        self.logger.info(f"Predictions saved to {filepath}" if overwrite or not os.path.exists(filepath) else f"Predictions appended to {filepath}")

    def save_accuracy(self, model_id, subfolder=None, overwrite=False):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        folder = 'model_accuracy'
        if subfolder:
            folder = os.path.join(folder, subfolder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, 'SOA_model_accuracy.csv')
        
        df = self.evaluation_df.reset_index()
        df['Model Class'] = self.__class__.__name__
        df['Model ID'] = model_id
        df['Config'] = json.dumps(self.config)
        df['Date Run'] = timestamp
        
        # Reorder the columns
        df = df[['Date Run', 'Model Class', 'Model ID', 'Config', 'index', 'RMSE', 'R2 Score', 'MAE', 'Explained Variance']]
        
        if overwrite or not os.path.exists(filepath):
            df.to_csv(filepath, index=False)
        else:
            df.to_csv(filepath, mode='a', header=False, index=False)
        self.logger.info(f"Accuracy metrics saved to {filepath}" if overwrite or not os.path.exists(filepath) else f"Accuracy metrics appended to {filepath}")



class SOA_TCN(BaseModel_DL_SOA):
    def __init__(self, model_type, data_preprocessor, config):
        super().__init__(model_type, data_preprocessor, config)
        self._initialize_model()  # Call this here
    
    def _initialize_model(self):
        logger.info("Initializing the TCN model...")
        self.model = Sequential()

        # Add the TCN layer
        self.model.add(TCN(
            nb_filters=self.config.get('nb_filters', 64),
            kernel_size=self.config.get('kernel_size', 2),
            nb_stacks=self.config.get('nb_stacks', 1),
            dilations=self.config.get('dilations', [1, 2, 4, 8, 16, 32]),
            padding=self.config.get('padding', 'causal'),
            use_skip_connections=self.config.get('use_skip_connections', True),
            dropout_rate=self.config.get('dropout_rate', 0.2),
            return_sequences=self.config.get('return_sequences', False),
            activation=self.config.get('activation', 'relu'),
            kernel_initializer=self.config.get('kernel_initializer', 'he_normal'),
            use_batch_norm=self.config.get('use_batch_norm', False),
            input_shape=self.config['input_shape']
        ))

        self.model.add(Dense(1))  # Assuming regression task, adjust if needed
        self.model.compile(optimizer=self.config.get('optimizer', 'adam'), loss='mean_squared_error')
        
        logger.info("TCN model compiled successfully.")
        self.model.summary()

    def train_model(self, epochs=100, batch_size=50, early_stopping=True):
        logger.info(f"Training the TCN model for {epochs} epochs with batch size of {batch_size}...")
        callbacks = [EarlyStopping(monitor='val_loss', patience=10)] if early_stopping else None
        self.history = self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=epochs,
            batch_size=batch_size, 
            validation_split=0.2,
            callbacks=callbacks, 
            shuffle=False
        )
        logger.info("Training completed.")

class SOA_NBEATS(BaseModel_DL_SOA):
    def __init__(self, model_type, data_preprocessor, config):
        super(SOA_NBEATS, self).__init__(model_type, data_preprocessor, config)
        # Assign the sequences
        self.X_train = data_preprocessor.y_train_seq
        self.y_train = data_preprocessor.y_train_seq
        self.X_test = data_preprocessor.y_test_seq
        self.y_test = data_preprocessor.y_test_seq
        self._initialize_model()

    
    def _initialize_model(self):
        logger.info("Initializing the N-BEATS model...")
        
        # Initialize the N-BEATS model
        self.model = NBeatsModel(
            lookback=self.config.get('lookback', 7),
            horizon=self.config.get('horizon', 1),
            num_generic_neurons=self.config.get('num_generic_neurons', 512),
            num_generic_stacks=self.config.get('num_generic_stacks', 30),
            num_generic_layers=self.config.get('num_generic_layers', 4),
            num_trend_neurons=self.config.get('num_trend_neurons', 256),
            num_trend_stacks=self.config.get('num_trend_stacks', 3),
            num_trend_layers=self.config.get('num_trend_layers', 4),
            num_seasonal_neurons=self.config.get('num_seasonal_neurons', 2048),
            num_seasonal_stacks=self.config.get('num_seasonal_stacks', 3),
            num_seasonal_layers=self.config.get('num_seasonal_layers', 4),
            num_harmonics=self.config.get('num_harmonics', 1),
            polynomial_term=self.config.get('polynomial_term', 3),
            loss=self.config.get('loss', 'mae'),
            learning_rate=self.config.get('learning_rate', 0.001),
            batch_size=self.config.get('batch_size', 1024)
        )
        
        self.model.build_layer()
        self.model.build_model()

        # Compile the actual Keras model inside the NBeatsModel
        optimizer = self.config.get('optimizer', 'adam')
        loss = self.config.get('loss', 'mae')
        self.model.model.compile(optimizer=optimizer, loss=loss)

        logger.info("N-BEATS model initialized and compiled successfully.")
        self.model.model.summary()

    def train_model(self, epochs=100, batch_size=32, early_stopping=True, patience=10, val_split=0.2):
        logger.info(f"Training the N-BEATS model for {epochs} epochs with batch size of {batch_size}...")

        callbacks = []
        if early_stopping:
            es_callback = EarlyStopping(monitor='val_loss', patience=patience)
            callbacks.append(es_callback)

        # Use the actual Keras model inside the NBeatsModel for training
        self.history = self.model.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=callbacks
        )
        logger.info("Training completed.")

class SOA_LSTNET(BaseModel_DL_SOA):
    def __init__(self, model_type, data_preprocessor, config):
        super(SOA_LSTNET, self).__init__(model_type, data_preprocessor, config)
        self._initialize_model()  # Call this here

    def _initialize_model(self):
        logger.info("Initializing the LSTNet model...")
        
        # Build the LSTNet model structure
        input_ = Input(shape=self.config['input_shape'])
        
        # CNN Layer
        x = Conv1D(filters=self.config.get('cnn_filters', 64), 
                   kernel_size=self.config.get('kernel_size', 3), 
                   activation='relu')(input_)
        
        # GRU Layer
        x = GRU(units=self.config.get('gru_units', 64), 
                return_sequences=True)(x)
        
        # Attention Layer
        query_value_attention_seq = Attention()([x, x])
        x = Add()([x, query_value_attention_seq])
        
        # Fully Connected Layer for Output
        x = Flatten()(x)
        output = Dense(1)(x)
        
        self.model = Model(inputs=input_, outputs=output)
        self.model.compile(optimizer=self.config.get('optimizer', 'adam'), loss=self.config.get('loss', 'mae'))
        
        logger.info("LSTNet model initialized and compiled successfully.")
        self.model.summary()

    def train_model(self, epochs=100, batch_size=32, early_stopping=True, patience=10, val_split=0.2):
        logger.info(f"Training the LSTNet model for {epochs} epochs with batch size of {batch_size}...")
        
        callbacks = []
        if early_stopping:
            es_callback = EarlyStopping(monitor='val_loss', patience=patience)
            callbacks.append(es_callback)
        
        self.history = self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=callbacks
        )
        logger.info("Training completed.")

class SOA_WAVENET(BaseModel_DL_SOA):
    
    def __init__(self, model_type, data_preprocessor, config):
        super(SOA_WAVENET, self).__init__(model_type, data_preprocessor, config)
        self.X_train = data_preprocessor.X_train_seq
        self.y_train = data_preprocessor.y_train_seq
        self.X_test = data_preprocessor.X_test_seq
        self.y_test = data_preprocessor.y_test_seq
        self._initialize_model()

    def wavenet_block(self, filters, kernel_size, dilation_rate):
        def f(input_):
            tanh_out = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal', activation='tanh')(input_)
            sigmoid_out = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal', activation='sigmoid')(input_)
            merged = Multiply()([tanh_out, sigmoid_out])
            out = Conv1D(filters=filters, kernel_size=1, activation='relu')(merged)
            skip = Conv1D(filters=filters, kernel_size=1)(out)
            
            # Adjust the residual connection's number of filters to match the input's filters
            input_residual = Conv1D(filters=filters, kernel_size=1)(input_)
            residual = Add()([skip, input_residual])
            
            return residual, skip
        return f

    def build_wavenet_model(self, input_shape, num_blocks, filters, kernel_size):
        input_ = Input(shape=input_shape)
        x = input_
        skip_connections = []
        for dilation_rate in [2**i for i in range(num_blocks)]:
            x, skip = self.wavenet_block(filters, kernel_size, dilation_rate)(x)
            skip_connections.append(skip)
        x = Add()(skip_connections)
        x = Activation('relu')(x)
        x = Conv1D(filters=filters, kernel_size=1, activation='relu')(x)
        x = Conv1D(filters=filters, kernel_size=1)(x)
        x = Flatten()(x)
        output = Dense(1)(x)  # For regression; adjust if different task

        model = Model(input_, output)
        return model

    def _initialize_model(self):
        logger.info("Initializing the WaveNet model...")
        
        self.model = self.build_wavenet_model(
            input_shape=self.config['input_shape'],
            num_blocks=self.config.get('num_blocks', 4),
            filters=self.config.get('filters', 32),
            kernel_size=self.config.get('kernel_size', 2)
        )
        
        optimizer = self.config.get('optimizer', 'adam')
        loss = self.config.get('loss', 'mae')
        self.model.compile(optimizer=optimizer, loss=loss)
        
        logger.info("WaveNet model compiled successfully.")
        self.model.summary()

    def train_model(self, epochs=100, batch_size=32, early_stopping=True, patience=10, val_split=0.2):
        logger.info(f"Training the WaveNet model for {epochs} epochs with batch size of {batch_size}...")
        
        callbacks = []
        if early_stopping:
            es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
            callbacks.append(es_callback)
        
        self.history = self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=callbacks
        )
        logger.info("Training completed.")

class SOA_TRANSFORMER(BaseModel_DL_SOA):
    class TransformerBlock(Layer):
        def __init__(self, embed_size, heads, dropout, forward_expansion):
            super().__init__()
            logger.info("Initializing TransformerBlock with embed_size %d, heads %d, dropout %f, forward_expansion %d.", embed_size, heads, dropout, forward_expansion)
            
            self.attention = SOA_TRANSFORMER.MultiHeadSelfAttention(embed_size, heads)
            self.norm1 = LayerNormalization(epsilon=1e-6)
            self.norm2 = LayerNormalization(epsilon=1e-6)
            
            self.feed_forward = tf.keras.Sequential([
                Dense(forward_expansion * embed_size, activation="relu"),
                Dense(embed_size),
            ])
            
            self.dropout = Dropout(dropout)

        def call(self, value, key, query, mask):
            logger.info("Calling TransformerBlock with value shape %s, key shape %s, query shape %s.", str(value.shape), str(key.shape), str(query.shape))
            
            attention = self.attention(value, key, query, mask)
            x = self.norm1(attention + query)
            x = self.dropout(x)
            
            forward = self.feed_forward(x)
            out = self.norm2(forward + x)
            out = self.dropout(out)
            
            return out

    class MultiHeadSelfAttention(Layer):
        def __init__(self, embed_size, heads):
            super().__init__()
            self.embed_size = embed_size
            self.heads = heads
            self.head_dim = embed_size // heads

            assert (
                self.head_dim * heads == embed_size
            ), "Embedding size needs to be divisible by heads"

            logger.info("Initializing MultiHeadSelfAttention with embed_size %d and heads %d.", embed_size, heads)
            
            self.values = Dense(self.head_dim, activation="linear")
            self.keys = Dense(self.head_dim, activation="linear")
            self.queries = Dense(self.head_dim, activation="linear")
            self.fc_out = Dense(embed_size, activation="linear")

        def call(self, values, keys, queries, mask):
            logger.info("Calling MultiHeadSelfAttention with values shape %s, keys shape %s, queries shape %s", str(values.shape), str(keys.shape), str(queries.shape))
            
            N = tf.shape(queries)[0]
            value_len, key_len, query_len = tf.shape(values)[1], tf.shape(keys)[1], tf.shape(queries)[1]

            values = tf.reshape(values, (N, value_len, self.heads, self.head_dim))
            keys = tf.reshape(keys, (N, key_len, self.heads, self.head_dim))
            queries = tf.reshape(queries, (N, query_len, self.heads, self.head_dim))

            values = self.values(values)
            keys = self.keys(keys)
            queries = self.queries(queries)

            score = tf.einsum("nqhd,nkhd->nhqk", queries, keys)
            if mask is not None:
                score *= mask

            attention_weights = tf.nn.softmax(score / (self.embed_size ** (1 / 2)), axis=3)
            out = tf.einsum("nhql,nlhd->nqhd", attention_weights, values)
            out = tf.reshape(out, (N, query_len, self.heads * self.head_dim))
            out = self.fc_out(out)
            return out

    def __init__(self, model_type, data_preprocessor, config):
        super(SOA_TRANSFORMER, self).__init__(model_type, data_preprocessor, config)
        self._initialize_model()
    
    def create_positional_encoding(self, max_seq_len, d_model):
        logger.info("Generating positional encoding for max_seq_len %d and d_model %d.", max_seq_len, d_model)

        pos = tf.range(max_seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
        sinusoidal_input = tf.matmul(pos, div_term[tf.newaxis, :])
        sines = tf.sin(sinusoidal_input)
        cosines = tf.cos(sinusoidal_input)
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        logger.info("Generated positional encoding with shape %s.", str(pos_encoding.shape))

        return pos_encoding

    def _initialize_model(self):
        inputs = Input(shape=self.config['input_shape'])

        embed_size = self.config['embed_size']
        num_layers = self.config['num_layers']
        heads = self.config['heads']
        dropout = self.config['dropout']
        forward_expansion = self.config['forward_expansion']

        # Logging the initial configuration
        logger.info("Initializing Transformer with the following configuration:")
        logger.info("Embed size: %d", embed_size)
        logger.info("Number of layers: %d", num_layers)
        logger.info("Number of heads: %d", heads)
        logger.info("Dropout rate: %f", dropout)
        logger.info("Forward expansion: %d", forward_expansion)

        # Add an embedding layer
        x = Dense(embed_size)(inputs)
        logger.info("Added embedding layer with shape: %s", str(x.shape))

        positional_encoding = self.create_positional_encoding(self.config['input_shape'][0], embed_size)
        x += positional_encoding
        logger.info("Added positional encoding to the model")

        for i in range(num_layers):
            x = SOA_TRANSFORMER.TransformerBlock(embed_size, heads, dropout, forward_expansion)(x, x, x, None)
            logger.info("Added Transformer block %d/%d", i+1, num_layers)

        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1, activation="linear")(x)
        logger.info("Added global average pooling and output layers")

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.config['optimizer'], loss=self.config['loss'])
        self.model.summary()
        logger.info("Model compiled with optimizer: %s and loss: %s", self.config['optimizer'], self.config['loss'])

    def train_model(self, epochs=100, batch_size=32, early_stopping=True, patience=10, val_split=0.2):
        logger.info("Training the Transformer model for %d epochs with batch size of %d...", epochs, batch_size)
        
        callbacks = []
        if early_stopping:
            es_callback = EarlyStopping(monitor='val_loss', patience=patience)
            callbacks.append(es_callback)
        
        self.history = self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=callbacks
        )
        logger.info("Training completed.")


models = {
    'SOA_TCN': {
        'class': SOA_TCN,
        'config': {
            'input_shape': (data_preprocessor.X_train_seq.shape[1], data_preprocessor.X_train_seq.shape[2]),
            'sequence_length': data_preprocessor.X_train_seq.shape[1],
            'num_features': data_preprocessor.X_train_seq.shape[2],
            'nb_filters': 64,
            'kernel_size': 2,
            'nb_stacks': 1,
            'dilations': [1, 2, 4, 8, 16, 32],
            'padding': 'causal',
            'use_skip_connections': True,
            'dropout_rate': 0.2,
            'return_sequences': False,
            'activation': 'relu',
            'kernel_initializer': 'he_normal',
            'use_batch_norm': False,
            'optimizer': 'adam'
        },
        'skip': False
    },
    'SOA_NBEATS': {
        'class': SOA_NBEATS,
        'config': {
            'lookback': 1,  # This should be 10
            'horizon': 1,
            'num_generic_neurons': 512,
            'num_generic_stacks': 30,
            'num_generic_layers': 4,
            'num_trend_neurons': 256,
            'num_trend_stacks': 3,
            'num_trend_layers': 4,
            'num_seasonal_neurons': 2048,
            'num_seasonal_stacks': 3,
            'num_seasonal_layers': 4,
            'num_harmonics': 1,
            'polynomial_term': 3,
            'loss': 'mae',
            'learning_rate': 0.001,
            'batch_size': 1024
        },
        'skip': False
    },
    'SOA_WAVENET': {
        'class': SOA_WAVENET,
        'config': {
            'input_shape': (data_preprocessor.X_train_seq.shape[1], data_preprocessor.X_train_seq.shape[2]),
            'sequence_length': data_preprocessor.X_train_seq.shape[1],
            'num_features': data_preprocessor.X_train_seq.shape[2],
            'num_blocks': 4,
            'filters': 32,
            'kernel_size': 2,
            'optimizer': 'adam',
            'loss': 'mae'
        },
        'skip': False
    },
    'SOA_LSTNET': {
        'class': SOA_LSTNET,
        'config': {
            'input_shape': (data_preprocessor.X_train_seq.shape[1], data_preprocessor.X_train_seq.shape[2]),
            'sequence_length': data_preprocessor.X_train_seq.shape[1],
            'num_features': data_preprocessor.X_train_seq.shape[2],
            'cnn_filters': 64,
            'gru_units': 64,
            'kernel_size': 3,
            'optimizer': 'adam',
            'loss': 'mae'
        },
        'skip': False
    },
    'SOA_TRANSFORMER': {
        'class': SOA_TRANSFORMER,
        'config': {
            'input_shape': (data_preprocessor.X_train_seq.shape[1], data_preprocessor.X_train_seq.shape[2]),
            'sequence_length': data_preprocessor.X_train_seq.shape[1],
            'num_features': data_preprocessor.X_train_seq.shape[2],
            'num_layers': 2,
            'embed_size': 64,
            'heads': 4,
            'dropout': 0.2,
            'forward_expansion': 2,
            'optimizer': 'adam',
            'loss': 'mae'
        },
        'skip': False
    }
}


def run_models(models, data_preprocessor, run_only=None, skip=None):
    for name, model_info in models.items():
        if run_only and name not in run_only:
            continue
        if skip and name in skip:
            continue
        if model_info.get('skip'):
            continue

        model_class = model_info['class']
        config = model_info['config']
        
        # Check if sequence_length and num_features exist in config
        sequence_length = config.get('sequence_length', None)  # Defaults to None if not found
        num_features = config.get('num_features', None)  # Defaults to None if not found

        # If they do exist, update the input shape
        if sequence_length is not None and num_features is not None:
            config['input_shape'] = (sequence_length, num_features)
        
        model = model_class(model_type=name, data_preprocessor=data_preprocessor, config=config)
        model.train_model(epochs=100, batch_size=32)
        model.make_predictions()
        evaluation_df = model.evaluate_model()
        display(evaluation_df)
        model.plot_history(plot=False)
        model.plot_predictions(plot=True)

        model_id = model.generate_model_id()
        model.save_predictions(model_id, subfolder='model_state_of_art', overwrite=False)
        model.save_accuracy(model_id, subfolder='model_state_of_art', overwrite=False)
        model.save_model_to_folder(version="1")



# Run all models
run_models(models, data_preprocessor)

# Run only specific models
#run_models(models, data_preprocessor, run_only=['SOA_NBEATS'])

# Skip specific models
#run_models(models, data_preprocessor, skip=['NBEATS'])
#print hello





import logging
import numpy as np
import pandas as pd
import json
import pickle
import hashlib
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging
import hashlib
from datetime import datetime, timedelta
from joblib import dump, load
import sklearn
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, accuracy_score
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_notebook, save
from bokeh.models import (HoverTool, ColumnDataSource, WheelZoomTool, Span, Range1d,
                          FreehandDrawTool, MultiLine, NumeralTickFormatter, Button, CustomJS)
from bokeh.layouts import column, row
from bokeh.io import curdoc, export_png
from bokeh.models.widgets import CheckboxGroup
from bokeh.themes import Theme

# Other settings
from IPython.display import display, HTML
import os, warnings, logging
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.3f}'.format)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


class BaseModel_TS:
    """
    A base class for traditional time series models.
    This class handles data preprocessing, model training, predictions, and evaluations.
    
    - AR model
    - ARIMA
    - SARIMA
    - ARIMAX
    - SARIMAX
    """
    def __init__(self, data_preprocessor, config, plot=True):
        self._validate_input(data_preprocessor.y_train, data_preprocessor.y_test)
        self.y_train = data_preprocessor.y_train
        self.y_test = data_preprocessor.y_test
        self.data = data_preprocessor.data
        self.config = config
        self.plot = plot
        self.logger = logging.getLogger(__name__)
        
    def _validate_input(self, y_train, y_test):
        """Validate the shape and type of training and testing data."""
        for arr, name in [(y_train, 'y_train'), (y_test, 'y_test')]:
            if not isinstance(arr, (np.ndarray, pd.Series)) or len(arr.shape) != 1:
                raise ValueError(f"{name} should be a 1D numpy array or pandas Series.")
                
    def inverse_scale_predictions(self):
        """ Inverse and unscale the predicstion back to their original shape"""
        try:
            self.train_predictions = self.target_scaler.inverse_transform(self.train_predictions.reshape(-1, 1)).flatten()
            self.test_predictions = self.target_scaler.inverse_transform(self.test_predictions.reshape(-1, 1)).flatten()
            self.logger.info("Predictions inverse transformed to original scale")
        except Exception as e:
            self.logger.error(f"Error occurred while inverse transforming predictions: {str(e)}")
            
    def compare_predictions(self):
        """Create dataframes comparing the original and predicted values for both training and test sets."""
        try:
            train_indices = self.data['Close'].iloc[:len(self.y_train)].values
            test_indices = self.data['Close'].iloc[-len(self.y_test):].values

            train_comparison_df = pd.DataFrame({'Original': train_indices, 'Predicted': self.train_predictions.ravel()})
            test_comparison_df = pd.DataFrame({'Original': test_indices, 'Predicted': self.test_predictions.ravel()})

            train_date_index = self.data.index[:len(self.y_train)]
            test_date_index = self.data.index[-len(self.y_test):]

            train_comparison_df.set_index(train_date_index, inplace=True)
            test_comparison_df.set_index(test_date_index, inplace=True)
            self.logger.info("Comparison dataframes generated")
            return train_comparison_df, test_comparison_df
        except Exception as e:
            self.logger.error(f"Error occurred while creating comparison dataframes: {str(e)}")

    def evaluate_model(self):
        """Evaluate the model using various metrics for both training and test sets."""
        try:
            train_comparison_df, test_comparison_df = self.compare_predictions()
            metrics = {
                'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                'R2 Score': r2_score,
                'MAE': mean_absolute_error,
                'Explained Variance': explained_variance_score
            }

            results = []
            for dataset, comparison_df in [('Train', train_comparison_df), ('Test', test_comparison_df)]:
                dataset_results = {metric_name: metric_func(comparison_df['Original'], comparison_df['Predicted']) for metric_name, metric_func in metrics.items()}
                results.append(dataset_results)

            results_df = pd.DataFrame(results, index=['Train', 'Test'])
            return results_df
        except Exception as e:
            self.logger.error(f"Error occurred while evaluating the model: {str(e)}")
        
    @staticmethod
    def update_config_hash_mapping(config_hash, config, folder_name="models_assets"):
        mapping_file_path = os.path.join(folder_name, 'config_hash_mapping.json')
        if os.path.exists(mapping_file_path):
            with open(mapping_file_path, 'r') as f:
                existing_mappings = json.load(f)
        else:
            existing_mappings = {}

        existing_mappings[config_hash] = config

        # Save updated mappings
        with open(mapping_file_path, 'w') as f:
            json.dump(existing_mappings, f, indent=4)

    def save_model_to_folder(self, version, folder_name="models_assets"):
        model_name = self.__class__.__name__[9:]  # Remove 'Enhanced_' from the class name
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        BaseModel_TS.update_config_hash_mapping(config_hash, self.config, folder_name)

        # Save the model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{model_name}_V{version}_{config_hash}_{timestamp}.joblib"
        full_path = os.path.join(folder_name, filename)
        dump(self.model, full_path)
        self.logger.info(f"Model saved to {full_path}")
        
    def plot_predictions(self):
        """Plot the original vs predicted values for both training and testing data."""
        if not self.plot:
            return

        train_comparison_df, test_comparison_df = self.compare_predictions()
        train_comparison_df.index = pd.to_datetime(train_comparison_df.index)
        test_comparison_df.index = pd.to_datetime(test_comparison_df.index)

        source_train = ColumnDataSource(data=dict(
            date=train_comparison_df.index,
            original=train_comparison_df['Original'],
            predicted=train_comparison_df['Predicted']
        ))

        source_test = ColumnDataSource(data=dict(
            date=test_comparison_df.index,
            original=test_comparison_df['Original'],
            predicted=test_comparison_df['Predicted']
        ))

        p1 = figure(width=700, height=600, x_axis_type="datetime", title="Training Data: Actual vs Predicted")
        p1.line('date', 'original', legend_label="Actual", line_alpha=0.6, source=source_train)
        p1.line('date', 'predicted', legend_label="Predicted", line_color="red", line_dash="dashed", source=source_train)
        p1.legend.location = "top_left"

        p2 = figure(width=700, height=600, x_axis_type="datetime", title="Testing Data: Actual vs Predicted")
        p2.line('date', 'original', legend_label="Actual", line_alpha=0.6, source=source_test)
        p2.line('date', 'predicted', legend_label="Predicted", line_color="red", line_dash="dashed", source=source_test)
        p2.legend.location = "top_left"

        hover1 = HoverTool()
        hover1.tooltips = [
            ("Date", "@date{%F}"),
            ("Actual Value", "@original{0,0.0000}"),
            ("Predicted Value", "@predicted{0,0.0000}")
        ]
        hover1.formatters = {"@date": "datetime"}
        p1.add_tools(hover1)

        hover2 = HoverTool()
        hover2.tooltips = [
            ("Date", "@date{%F}"),
            ("Actual Value", "@original{0,0.0000}"),
            ("Predicted Value", "@predicted{0,0.0000}")
        ]
        hover2.formatters = {"@date": "datetime"}
        p2.add_tools(hover2)

        # Show plots
        show(row(p1, p2))



class Enhanced_AR(BaseModel_TS):
    """
    Initialize the AR model.
    """
    def __init__(self, data_preprocessor, config, plot=True):
        super().__init__(data_preprocessor, config, plot)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize AR model."""
        self.model = AutoReg(self.y_train, lags=self.config['lags'])
        self.logger.info("AR model initialized.")

    def train_model(self):
        """Train the AR model."""
        try:
            self.model_result = self.model.fit()
            self.logger.info("AR model trained successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while training the model: {str(e)}")

    def make_predictions(self):
        """Make predictions using the trained model for training and test sets."""
        try:
            start = len(self.y_train)
            end = start + len(self.y_test) - 1
            self.train_predictions = self.model_result.predict(start=0, end=start-1)
            self.test_predictions = self.model_result.predict(start=start, end=end)
            self.logger.info("Predictions made successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while making predictions: {str(e)}")

class Enhanced_ARIMA(BaseModel_TS):
    """
    Initialize the ARIMA model.
    """
    def __init__(self, data_preprocessor, config, plot=True):
        super().__init__(data_preprocessor, config, plot)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize ARIMA model."""
        self.model = ARIMA(self.y_train, order=self.config['order'])
        self.logger.info("ARIMA model initialized.")

    def train_model(self):
        """Train the ARIMA model."""
        try:
            self.model_result = self.model.fit()
            self.logger.info("ARIMA model trained successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while training the model: {str(e)}")

    def make_predictions(self):
        """Make predictions using the trained model for training and test sets."""
        try:
            start = len(self.y_train)
            end = start + len(self.y_test) - 1
            self.train_predictions = self.model_result.predict(start=0, end=start-1, typ='levels')
            self.test_predictions = self.model_result.predict(start=start, end=end, typ='levels')
            self.logger.info("Predictions made successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while making predictions: {str(e)}")

class Enhanced_SARIMA(BaseModel_TS):
    """
    Initialize the SARIMA model.
    """
    def __init__(self, data_preprocessor, config, plot=True):
        super().__init__(data_preprocessor, config, plot)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize SARIMA model."""
        self.model = SARIMAX(self.y_train, order=self.config['order'], seasonal_order=self.config['seasonal_order'])
        self.logger.info("SARIMA model initialized.")

    def train_model(self):
        """Train the SARIMA model."""
        try:
            self.model_result = self.model.fit(disp=False)
            self.logger.info("SARIMA model trained successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while training the model: {str(e)}")

    def make_predictions(self):
        """Make predictions using the trained model for training and test sets."""
        try:
            start = len(self.y_train)
            end = start + len(self.y_test) - 1
            self.train_predictions = self.model_result.predict(start=0, end=start-1, typ='levels')
            self.test_predictions = self.model_result.predict(start=start, end=end, typ='levels')
            self.logger.info("Predictions made successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while making predictions: {str(e)}")

class Enhanced_ARIMAX(BaseModel_TS):
    """
    Initialize the ARIMAX model.
    """
    def __init__(self, data_preprocessor, config, plot=True):
        super().__init__(data_preprocessor, config, plot)
        self.X_train = data_preprocessor.X_train
        self.X_test = data_preprocessor.X_test
        self._initialize_model()

    def _initialize_model(self):
        """Initialize ARIMAX model."""
        self.model = ARIMA(self.y_train, exog=self.X_train, order=self.config['order'])
        self.logger.info("ARIMAX model initialized.")

    def train_model(self):
        """Train the ARIMAX model."""
        try:
            self.model_result = self.model.fit()
            self.logger.info("ARIMAX model trained successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while training the model: {str(e)}")

    def make_predictions(self):
        """Make predictions using the trained model for training and test sets."""
        try:
            start = len(self.y_train)
            end = start + len(self.y_test) - 1
            self.train_predictions = self.model_result.predict(start=0, end=start-1, exog=self.X_train, typ='levels')
            self.test_predictions = self.model_result.predict(start=start, end=end, exog=self.X_test, typ='levels')
            self.logger.info("Predictions made successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while making predictions: {str(e)}")

class Enhanced_SARIMAX(BaseModel_TS):
    """
    Initialize the SARIMAX model.
    """
    def __init__(self, data_preprocessor, config, plot=True):
        super().__init__(data_preprocessor, config, plot)
        self.X_train = data_preprocessor.X_train
        self.X_test = data_preprocessor.X_test
        self._initialize_model()

    def _initialize_model(self):
        """Initialize SARIMAX model."""
        self.model = SARIMAX(self.y_train, exog=self.X_train, order=self.config['order'], seasonal_order=self.config['seasonal_order'])
        self.logger.info("SARIMAX model initialized.")

    def train_model(self):
        """Train the SARIMAX model."""
        try:
            self.model_result = self.model.fit(disp=False)
            self.logger.info("SARIMAX model trained successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while training the model: {str(e)}")

    def make_predictions(self):
        """Make predictions using the trained model for training and test sets."""
        try:
            start = len(self.y_train)
            end = start + len(self.y_test) - 1
            self.train_predictions = self.model_result.predict(start=0, end=start-1, exog=self.X_train, typ='levels')
            self.test_predictions = self.model_result.predict(start=start, end=end, exog=self.X_test, typ='levels')
            self.logger.info("Predictions made successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while making predictions: {str(e)}")



models_config = {
    'AR': {
        'class': Enhanced_AR,
        'config': {
            'lags': 5
        },
        'skip': False
    },
    'ARIMA': {
        'class': Enhanced_ARIMA,
        'config': {
            'order': (5, 1, 0)
        },
        'skip': False
    },
    'SARIMA': {
        'class': Enhanced_SARIMA,
        'config': {
            'order': (1, 1, 1),
            'seasonal_order': (1, 1, 1, 12)
        },
        'skip': False
    },
    'ARIMAX': {
        'class': Enhanced_ARIMAX,
        'config': {
            'order': (5, 1, 0)
            # Add exogenous variables if needed
        },
        'skip': False
    },
    'SARIMAX': {
        'class': Enhanced_SARIMAX,
        'config': {
            'order': (1, 1, 1),
            'seasonal_order': (1, 1, 1, 12)
            # Add exogenous variables if needed
        },
        'skip': False
    }
}


def run_models(models, data_preprocessor, run_only=None, skip=None):
    for name, model_info in models.items():
        if run_only and name not in run_only:
            continue
        if skip and name in skip:
            continue
        if model_info.get('skip'):
            continue

        model_class = model_info['class']
        config = model_info['config']
        
        model = model_class(data_preprocessor=data_preprocessor, config=config, plot=True)
        model.train_model()
        model.make_predictions()
        evaluation_df = model.evaluate_model()
        print(f"{name} Model Evaluation:\n", evaluation_df)
        model.plot_predictions()

# Usage
data_preprocessor = ...  # Assume this is already initialized and prepared
run_models(models_config, data_preprocessor)

# Run only specific models
run_models(models_config, data_preprocessor, run_only=['AR', 'ARIMA'])

# Skip specific models
run_models(models_config, data_preprocessor, skip=['SARIMA'])


