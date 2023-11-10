# Watch-out-for-the-DL-bugs


## `data_fetcher.py` - Cryptocurrency Data Retrieval and Formatting

### Description

The `data_fetcher.py` module is responsible for the acquisition, processing, and local storage of cryptocurrency price data. It utilizes the `cryptocmd` library to fetch historical price data from CoinMarketCap. The module is designed to handle multiple cryptocurrencies and includes robust error handling, retries with exponential backoff, and data integrity checks.

Key features of this module include:
- Data retrieval for a predefined list of cryptocurrency symbols.
- Data validation against expected column formats.
- Local caching of data to reduce API calls and speed up subsequent data retrieval.
- Data formatting for display purposes, with monetary values and volumes presented in a human-readable form.
- Logging of operations to track the data fetching process.

### Method Overview

#### `CryptoData` Class
- `__init__(self, crypto_symbols, retries, backoff_factor)`: Initializes the `CryptoData` object with cryptocurrency symbols to fetch, retry logic, and the backoff factor for retries.
- `_fetch_cryptocmd_data(self, symbol)`: Private method to fetch data for a single cryptocurrency symbol with retries and exponential backoff.
- `_local_data_path(self, symbol)`: Private method to generate the path for local storage of the cryptocurrency data.
- `get_cryptocmd_data(self, symbol, overwrite)`: Public method to fetch and store cryptocurrency data, with an option to overwrite existing data.
- `get_all_data(self, overwrite)`: Public method to fetch data for all specified cryptocurrencies.
- `get_display_data(self, symbol)`: Public method to format and return data for a single cryptocurrency for display purposes.
- `get_all_display_data(self)`: Public method to fetch and format data for all specified cryptocurrencies for display purposes.

#### Helper Functions
- `_format_monetary_value(value)`: Static method to format monetary values to a string.
- `_format_volume_value(value)`: Static method to format volume values to a string.

#### Script Entry Point
- `run_data_fetcher(run, tickers, get_display_data, overwrite)`: Function to fetch all data and optionally format it for display.

### Usage Example
```python
tickers = ['BTC', 'ETH', 'ADA']
crypto_data_obj, all_data, all_display_data = run_data_fetcher(True, tickers, get_display_data=True, overwrite=True)
btc_data = all_data['BTC']
btc_display_data = all_display_data.loc['BTC']
python```


## `data_analytics.py` - Advanced Cryptocurrency Data Analysis

### Description

The `data_analytics.py` module is dedicated to providing advanced analytical capabilities for cryptocurrency data. It builds upon the data fetched by `data_fetcher.py` to perform historical volatility calculations, time-based aggregation and analysis, price variation computations, and all-time record retrieval. This module is structured to save the results of these analyses in an organized manner for easy access and further use.

Key features of this module include:
- Calculation of historical volatility using a specified rolling window.
- Time-based analysis with frequency aggregation for open, high, low, and close prices.
- Price variation calculation both in absolute dollar terms and relative percentage.
- Retrieval of all-time high and low records for each cryptocurrency.
- Persistence of analysis results to Excel files for reporting and archival purposes.
- Comprehensive logging throughout the analysis process.

### Method Overview

#### `CryptoDataAnalytics` Class
- `__init__(self, crypto_data)`: Initializes the `CryptoDataAnalytics` object with the cryptocurrency DataFrame.
- `_create_output_dir(self)`: Private method to create a directory for saving analysis files.
- `calculate_historical_volatility(self, column, window)`: Calculates and returns the historical volatility of the specified price column over a rolling window.
- `perform_time_analysis(self, freq)`: Aggregates data based on a time frequency and performs analysis on price data.
- `calculate_price_variation(self, data)`: Calculates the absolute and relative price variation within the provided DataFrame.
- `retrieve_all_time_records(self)`: Retrieves all-time high and low prices and their corresponding dates.
- `perform_and_save_all_analyses(self)`: Orchestrates the execution of all analytical methods and saves their results to Excel files.
- `save_analysis_to_excel(self, analysis, filename)`: Saves a DataFrame to an Excel file within the output directory.

#### Execution Function
- `run_data_analytics(run)`: Entry point function to run all data analytics if the `run` flag is set to `True`.

### Usage Example
```python
# Uncomment the line below to run the analytics and retrieve all-time records.
# analytics, all_time_high, all_time_low, yearly_data, monthly_data, weekly_data = run_data_analytics(True)

# After running, you can access the results as follows:
# print(f"All Time High: {all_time_high} on {all_time_high_date}")
# print(f"All Time Low: {all_time_low} on {all_time_low_date}")






