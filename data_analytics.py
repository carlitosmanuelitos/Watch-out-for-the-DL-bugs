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