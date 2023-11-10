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






