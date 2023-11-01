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