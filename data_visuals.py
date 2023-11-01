import numpy as np 
from cryptocmd import CmcScraper
from math import pi
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_notebook, save
from bokeh.models import (HoverTool, ColumnDataSource, WheelZoomTool, Span, Range1d,
                          FreehandDrawTool, MultiLine, NumeralTickFormatter, Button, CustomJS)
from bokeh.layouts import column, row
from bokeh.io import curdoc, export_png
from bokeh.models.widgets import CheckboxGroup
from bokeh.themes import Theme
from data_fetcher import btc_data


# Other settings
from IPython.display import display, HTML
import os, warnings, logging
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.3f}'.format)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


class CryptoAnalyticsVisual:
    """
    The CryptoAnalyticsVisual class provides tools for cryptocurrency market analysis and visualization.
    
    Attributes:
        data (pd.DataFrame): Raw crypto data with 'Open', 'Close', 'High', 'Low', 'Volume'.
    
    Methods:
        _create_visualizations_directory: Creates directory for visualizations.
        save_plot_to_file: Saves Bokeh plot to file.
        calculate_macd, plot_macd_bokeh: Handles MACD calculation and plotting.
        calculate_rsi, plot_rsi_bokeh: Handles RSI calculation and plotting.
        calculate_bollinger_bands, plot_bollinger_bands_bokeh: Handles Bollinger Bands.
        calculate_fibonacci_retracement, plot_fibonacci_retracement_bokeh: Handles Fibonacci retracement.
        volume_analysis, plot_volume_analysis_bokeh: Handles volume analysis.
        create_candlestick_chart, plot_trend_bokeh: Plots candlestick and trend data.

    Example:
        >>> df = pd.read_csv('crypto_data.csv')
        >>> analytics = CryptoAnalyticsVisual(df)
        >>> analytics.plot_macd_bokeh()
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data
        output_notebook()
        curdoc().theme = 'dark_minimal'
        self._create_visualizations_directory()
        logger.info('CryptoAnalyticsVisual instance created and initialized.')

    def _create_visualizations_directory(self):
        """Creates a directory for storing visualization assets."""
        if not os.path.exists('visualizations_assets'):
            os.makedirs('visualizations_assets')
            logger.info("Created directory: visualizations_assets")

    def save_plot_to_file(self, plot, filename: str, format: str = 'html'):
        """Saves plot to a file."""
        full_path = os.path.join('visualizations_assets', filename)
        if format == 'html':
            save(plot, filename=full_path)
            logger.info(f'Plot saved to file: {full_path}')
        else:
            logger.error('Unsupported file format: {}'.format(format))

    def calculate_macd(self, short_window=12, long_window=26, signal_window=9):
        """Calculates MACD and signal line."""
        short_ema = self.data['Close'].ewm(span=short_window, adjust=False).mean()
        long_ema = self.data['Close'].ewm(span=long_window, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        logger.info(f'MACD calculated with short_window={short_window}, long_window={long_window}, and signal_window={signal_window}')
        return macd_line, signal_line

    def plot_macd_bokeh(self, display=True):
        """Plots MACD and signal line."""
        macd_line, signal_line = self.calculate_macd()
        source = ColumnDataSource(data=dict(x=self.data.index, y1=macd_line, y2=signal_line))
        p = figure(width=1400, height=600, title="MACD Analysis", x_axis_type="datetime")
        p.line(x='x', y='y1', source=source, legend_label="MACD Line", color="blue", alpha=0.8)
        p.line(x='x', y='y2', source=source, legend_label="Signal Line", color="red", alpha=0.8)
        hover = HoverTool(tooltips=[("Date", "@x{%F}"), ("MACD", "@y1"), ("Signal", "@y2")], formatters={"@x": "datetime"})
        p.add_tools(hover)
        logger.info('MACD plot displayed.')
        if display:
            show(p)
        return p

    def calculate_rsi(self, window=14):
        """Calculates the Relative Strength Index (RSI)."""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        logger.info(f'RSI calculated with window={window}')
        return rsi

    def plot_rsi_bokeh(self, display=True):
        """Plots the Relative Strength Index (RSI)."""
        rsi = self.calculate_rsi()
        source = ColumnDataSource(data=dict(x=self.data.index, y=rsi))
        p = figure(width=1400, height=600, title="RSI Analysis", x_axis_type="datetime")
        p.line(x='x', y='y', source=source, legend_label="RSI", color="green", alpha=0.8)
        hover = HoverTool(tooltips=[("Date", "@x{%F}"), ("RSI", "@y")], formatters={"@x": "datetime"})
        p.add_tools(hover)
        p.add_layout(Span(location=70, dimension='width', line_color='red', line_width=1, line_dash='dashed'))
        p.add_layout(Span(location=30, dimension='width', line_color='red', line_width=1, line_dash='dashed'))
        logger.info('RSI plot displayed.')
        if display:
            show(p)
        return p

    def calculate_bollinger_bands(self, window=20, num_std=2):
        """Calculates upper and lower Bollinger Bands."""
        rolling_mean = self.data['Close'].rolling(window=window).mean()
        rolling_std = self.data['Close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        logger.info(f'Bollinger Bands calculated with window={window} and num_std={num_std}')
        return upper_band, lower_band

    def plot_bollinger_bands_bokeh(self, display=True):
        """Plots Bollinger Bands and Close Price."""
        upper_band, lower_band = self.calculate_bollinger_bands()
        source = ColumnDataSource(data=dict(x=self.data.index, close=self.data['Close'], upper=upper_band, lower=lower_band))
        p = figure(width=1400, height=600, title="Bollinger Bands Analysis", x_axis_type="datetime")
        p.line(x='x', y='close', source=source, legend_label="Close Price", color="blue", alpha=0.8)
        p.line(x='x', y='upper', source=source, legend_label="Upper Band", color="red", alpha=0.5)
        p.line(x='x', y='lower', source=source, legend_label="Lower Band", color="green", alpha=0.5)
        hover = HoverTool(tooltips=[("Date", "@x{%F}"), ("Close", "@close{$0,0.00} K"), ("Upper Band", "@upper{$0,0.00} K"), ("Lower Band", "@lower{$0,0.00} K")], formatters={"@x": "datetime"})
        p.yaxis.formatter = NumeralTickFormatter(format="$0,0.00")
        p.add_tools(hover)
        logger.info('Bollinger Bands plot displayed.')
        if display:
            show(p)
        return p

    def calculate_fibonacci_retracement(self):
        """Calculates Fibonacci retracement levels."""
        max_price = self.data['High'].max()
        min_price = self.data['Low'].min()
        diff = max_price - min_price
        levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        retracement_levels = {level: (max_price - level * diff) for level in levels}
        logger.info('Fibonacci retracement levels calculated.')
        return retracement_levels

    def plot_fibonacci_retracement_bokeh(self, display=True):
        """Plots Fibonacci retracement levels."""
        retracement_levels = self.calculate_fibonacci_retracement()
        source = ColumnDataSource(data=dict(x=self.data.index, close=self.data['Close']))
        p = figure(width=1400, height=600, title="Fibonacci Retracement Levels", x_axis_type="datetime")
        p.line(x='x', y='close', source=source, legend_label="Close Price", color="blue", alpha=0.8)
        for level, price in retracement_levels.items():
            p.add_layout(Span(location=price, dimension='width', line_color='red', line_width=1, line_dash='dashed'))
            p.line([], [], line_color="red", legend_label=f'Level: {level}', line_dash='dashed')
        hover = HoverTool(tooltips=[("Date", "@x{%F}"), ("Close Price", "@close{$0,0.00} K")], formatters={"@x": "datetime"})
        p.yaxis.formatter = NumeralTickFormatter(format="$0,0.00")
        p.add_tools(hover)
        logger.info('Fibonacci retracement plot displayed.')
        if display:
            show(p)
        return p

    def volume_analysis(self):
        """Analyzes volume data and computes average volume over 30 days."""
        volume = self.data['Volume'] / 1_000  # Convert to Thousands
        avg_volume = volume.rolling(window=30).mean()
        logger.info('Volume analysis completed.')
        return volume, avg_volume

    def plot_volume_analysis_bokeh(self, display=True):
        """Plots volume and 30-day average volume."""
        volume, avg_volume = self.volume_analysis()
        source = ColumnDataSource(data=dict(x=self.data.index, volume=volume, avg_volume=avg_volume))
        p = figure(width=1400, height=600, title="Volume Analysis (in Thousands)", x_axis_type="datetime")
        p.vbar(x='x', top='volume', source=source, width=0.9, legend_label="Volume", alpha=0.6, color="blue")
        p.line(x='x', y='avg_volume', source=source, legend_label="30-Day Avg Volume", color="red", line_width=2)
        hover = HoverTool(tooltips=[("Date", "@x{%F}"), ("Volume", "@volume{$0,0} K"), ("30-Day Avg Volume", "@avg_volume{$0,0} K")], formatters={"@x": "datetime"})
        p.yaxis.formatter = NumeralTickFormatter(format="$0,0")
        p.add_tools(hover)
        logger.info('Volume analysis plot displayed.')
        if display:
            show(p) 
        return p
    
    def create_candlestick_chart(self, time_period='last_month', ma_period=20, display=True):
        """
        Creates a candlestick chart for the selected time period and moving average period.
        """
        logger.info("Creating candlestick chart.")
        # Assuming _select_data is a method that filters the data based on the time_period
        df = self._select_data(time_period)

        df['index_col'] = df.index  
        df['MA'] = df['Close'].rolling(window=ma_period).mean()

        inc = df.Close > df.Open
        dec = df.Open > df.Close

        source_inc = ColumnDataSource(df[inc])
        source_dec = ColumnDataSource(df[dec])
        source_hover = ColumnDataSource(df)
        w = 12 * 60 * 60 * 1000 

        TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
        p = figure(x_axis_type="datetime", tools=TOOLS, width=1400, title="Crypto Candlestick with MA")
        p.xaxis.major_label_orientation = pi / 4
        p.grid.grid_line_alpha = 0.3

        p.segment('index_col', 'High', 'index_col', 'Low', color="black", source=source_inc)
        p.vbar('index_col', w, 'Open', 'Close', fill_color="#39B86B", line_color="black", source=source_inc)
        p.segment('index_col', 'High', 'index_col', 'Low', color="black", source=source_dec)
        p.vbar('index_col', w, 'Open', 'Close', fill_color="#F2583E", line_color="black", source=source_dec)

        hover = HoverTool(
            tooltips=[
                ("Date", "@index_col{%F}"),
                ("Open", "@{Open}{($ 0,0.00)}"),
                ("Close", "@{Close}{($ 0,0.00)}"),
                ("High", "@{High}{($ 0,0.00)}"),
                ("Low", "@{Low}{($ 0,0.00)}"),
                ("MA", "@{MA}{($ 0,0.00)}")
            ],
            formatters={
                '@index_col': 'datetime',
                '@Open': 'numeral',
                '@Close': 'numeral',
                '@High': 'numeral',
                '@Low': 'numeral',
                '@MA': 'numeral'
            },
            mode='vline'
        )
        p.add_tools(hover)
        p.line('index_col', 'MA', color='blue', legend_label='Moving Average', source=source_hover)

        if display:
            show(p)

        logger.info('Candlestick chart displayed.')
        return p

    def plot_trend_bokeh(self, display=True):
        """
        Plots trend data using various moving averages for analysis.
        """
        logger.info("Creating trend analysis plot.")
        # Assuming _identify_trend is a method that identifies the trend in the data
        trend_data = self._identify_trend()
        source = ColumnDataSource(data={**{'x': self.data.index, 'price': trend_data['Price']}, **{f"mavg{period}": trend_data[f"{period}_day_mavg"] for period in [3, 7, 15, 40, 90, 120]}})

        p = figure(width=1400, height=600, title="Trend Analysis using Moving Averages", x_axis_type="datetime")
        p.line(x='x', y='price', source=source, legend_label="Close Price", alpha=0.8)

        colors = {"3": "orange", "7": "yellow", "15": "cyan", "40": "red", "90": "purple", "120": "green"}
        for period, color in colors.items():
            p.line(x='x', y=f'mavg{period}', source=source, legend_label=f"{period}-day MA", color=color, line_dash="dashed")

        hover = HoverTool(
            tooltips=[
                ("Date", "@x{%F}"),
                ("Price", "@price{$0,0.00} K"),
                ("3-day MA", "@mavg3{$0,0.00} K"),
                ("7-day MA", "@mavg7{$0,0.00} K"),
                ("15-day MA", "@mavg15{$0,0.00} K"),
                ("40-day MA", "@mavg40{$0,0.00} K"),
                ("90-day MA", "@mavg90{$0,0.00} K"),
                ("120-day MA", "@mavg120{$0,0.00} K")
            ],
            formatters={"@x": "datetime"}
        )
        p.yaxis.formatter = NumeralTickFormatter(format="$0,0.00")
        p.add_tools(hover)
        if display:
            show(p)

        logger.info('Trend plot displayed.')
        return p
    
    def _identify_trend(self, column: str = 'Close'):
        signals = pd.DataFrame(index=self.data.index)
        signals['Price'] = self.data[column]

        # Moving Averages
        ma_periods = [3, 7, 15, 40, 90, 120]
        for period in ma_periods:
            signals[f'{period}_day_mavg'] = self.data[column].rolling(window=period, min_periods=1, center=False).mean()

        # Signal based on 40-day and 120-day moving averages (since there's no 100-day moving average in the new setup)
        signals['signal'] = 0.0
        signals['signal'][40:] = np.where(signals['40_day_mavg'][40:] > signals['120_day_mavg'][40:], 1.0, 0.0)
        return signals

    def _select_data(self, time_period):
        logger.info("Selecting data for time period: %s", time_period)
        if time_period == 'last_month':
            last_month = self.data.index.max() - pd.DateOffset(months=1)
            df = self.data[self.data.index >= last_month]
        elif time_period == 'last_3_months':
            last_3_months = self.data.index.max() - pd.DateOffset(months=3)
            df = self.data[self.data.index >= last_3_months]
        elif time_period == 'last_6_months':
            last_6_months = self.data.index.max() - pd.DateOffset(months=6)
            df = self.data[self.data.index >= last_6_months]
        elif time_period == 'last_1_year':
            last_1_year = self.data.index.max() - pd.DateOffset(years=1)
            df = self.data[self.data.index >= last_1_year]
        elif time_period == 'last_3_years':
            last_3_years = self.data.index.max() - pd.DateOffset(years=3)
            df = self.data[self.data.index >= last_3_years]
        else:
            df = self.data

        return df


def run_data_visuals(run: bool):
    if not run:
        return None

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
    crypto_analytics.save_plot_to_file(volume, 'volume.html')

    return crypto_analytics, candle, trend, bollinger_bands, macd, rsi, fibonacci_retracement, volume

#crypto_analytics, candle, trend, bollinger_bands, macd, rsi, fibonacci_retracement, volume = run_data_visuals(True)
