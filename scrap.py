from IPython.display import display, HTML
from data_fetcher import run_data_fetcher
from data_fetcher import btc_data
from data_analytics import run_data_analytics
from data_visuals import run_data_visuals  # Replace with the actual name of your data visuals script
from data_eng import run_feature_engineering  # Replace with the actual name of your feature engineering script


# PART 1 - Fetch the data only once here
display(btc_data)


# PART 2 - Run data analytics by passing the fetched data
analytics, all_time_high, all_time_low, yearly_data, monthly_data, weekly_data = run_data_analytics(True)
display(weekly_data)


# PART 3 -  Run the data visuals and unpack the returned objects
crypto_analytics, candle, trend, bollinger_bands, macd, rsi, fibonacci_retracement, volume = run_data_visuals(True)
display(candle)


# PART 4 - Data Engeneering and retrieve dataframe
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

data_eng = run_feature_engineering(True, config)
display(data_eng)

