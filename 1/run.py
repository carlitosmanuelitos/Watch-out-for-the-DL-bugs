from IPython.display import display, HTML
from data_fetcher import CryptoData 
from data_analytics import CryptoDataAnalytics
from data_visuals import CryptoAnalyticsVisual
from data_eng import Feature_Eng_Tech

# Part 1 - Data Fetching
crypto_data_obj = CryptoData(['BTC', 'ETH', 'ADA'])
all_data = crypto_data_obj.get_all_data(overwrite=True)
btc_data, eth_data, ada_data = all_data['BTC'], all_data['ETH'], all_data['ADA']
btc_display_data = crypto_data_obj.get_display_data('BTC')

# Part 2 - Data Analytics
analytics = CryptoDataAnalytics(btc_data)
all_time_high, all_time_low, all_time_high_date, all_time_low_date = analytics.retrieve_all_time_records()
print(f"All Time High: {all_time_high} on {all_time_high_date}"), print(f"All Time Low: {all_time_low} on {all_time_low_date}")

analytics.perform_and_save_all_analyses()
yearly_data = analytics.perform_time_analysis('Y')
monthly_data = analytics.perform_time_analysis('M')
weekly_data = analytics.perform_time_analysis('W')
display(yearly_data), display(monthly_data), display(weekly_data)

# Part 3 - Data Visualization
    
crypto_analytics = CryptoAnalyticsVisual(btc_data)
candle = crypto_analytics.create_candlestick_chart(time_period='last_6_months', ma_period=20)
trend = crypto_analytics.plot_trend_bokeh()
bollinger_bands = crypto_analytics.plot_bollinger_bands_bokeh()
macd = crypto_analytics.plot_macd_bokeh()
rsi = crypto_analytics.plot_rsi_bokeh()
fibonacci_retracement = crypto_analytics.plot_fibonacci_retracement_bokeh()
volume = crypto_analytics.plot_volume_analysis_bokeh()

crypto_analytics.save_plot_to_file(candle, 'candle.html')
crypto_analytics.save_plot_to_file(trend, 'trend.html')
crypto_analytics.save_plot_to_file(bollinger_bands, 'bollinger_bands.html')
crypto_analytics.save_plot_to_file(macd, 'macd.html')
crypto_analytics.save_plot_to_file(rsi, 'rsi.html')
crypto_analytics.save_plot_to_file(fibonacci_retracement, 'fibonacci_retracement.html')
crypto_analytics.save_plot_to_file(volume, 'rsi.html')

# Part 4 - Data Eng

feature_eng = Feature_Eng_Tech(btc_data, target_column='Close')
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
feature_eng.feature_engineering(config)
data_eng = feature_eng.get_engineered_data()
display(data_eng)