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
```

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
```

## `data_visuals.py` - Visualization of Cryptocurrency Analytics

### Description

The `data_visuals.py` module offers a suite of visualization tools for analyzing cryptocurrency market trends and indicators such as Moving Average Convergence Divergence (MACD), Relative Strength Index (RSI), Bollinger Bands, Fibonacci Retracement levels, and volume analysis. It leverages Bokeh for creating interactive charts and Matplotlib for static plots, ensuring that the visualizations are both informative and engaging.

Key features of this module include:
- Generation of standard financial charts and custom analytics visualizations.
- Interactive Bokeh plots with hover tools for detailed data inspection.
- Automated saving of visualizations in various formats for reporting and sharing.
- Comprehensive logging for tracking visualization processes.

### Method Overview

#### `CryptoAnalyticsVisual` Class
- `__init__(self, data)`: Constructor for initializing the class with raw cryptocurrency data.
- `_create_visualizations_directory(self)`: Private method for creating a directory to store visualization assets.
- `save_plot_to_file(self, plot, filename, format)`: Saves a Bokeh plot to a file in the specified format.
- `calculate_macd(self, short_window, long_window, signal_window)`: Computes the MACD and signal line.
- `plot_macd_bokeh(self, display)`: Generates an interactive Bokeh plot for the MACD indicator.
- `calculate_rsi(self, window)`: Calculates the RSI of the cryptocurrency data.
- `plot_rsi_bokeh(self, display)`: Generates an interactive Bokeh plot for the RSI indicator.
- `calculate_bollinger_bands(self, window, num_std)`: Calculates the Bollinger Bands.
- `plot_bollinger_bands_bokeh(self, display)`: Generates an interactive Bokeh plot for Bollinger Bands.
- `calculate_fibonacci_retracement(self)`: Calculates Fibonacci retracement levels based on historical data.
- `plot_fibonacci_retracement_bokeh(self, display)`: Generates an interactive Bokeh plot for Fibonacci retracement levels.
- `volume_analysis(self)`: Analyzes trading volume data and calculates its average over a period.
- `plot_volume_analysis_bokeh(self, display)`: Generates an interactive Bokeh plot for volume analysis.
- `create_candlestick_chart(self, time_period, ma_period, display)`: Creates a candlestick chart for the given time period and moving average period.
- `plot_trend_bokeh(self, display)`: Plots trend data using various moving averages for analysis.

### Usage Example
```python
# Uncomment the line below to run the visualizations.
# crypto_analytics, candle, trend, bollinger_bands, macd, rsi, fibonacci_retracement, volume = run_data_visuals(True)

# To view a specific visualization in a Jupyter notebook, use the corresponding plotting method, e.g.:
# macd_plot = crypto_analytics.plot_macd_bokeh(display=False)  # Set display to False to prevent auto-showing the plot
```

## `data_eng.py` - Feature Engineering for Time Series Data

### Description

The `data_eng.py` module focuses on advanced feature engineering techniques for time series data. It serves to enhance the original dataset with additional features that can improve the performance of machine learning models. The class `Feature_Eng_Tech` within this module provides a comprehensive suite of methods for handling missing values, adding date-time features, generating lag and rolling window features, and more complex transformations like Fourier features, seasonal decomposition, and detrending.

Key features of this module include:
- Comprehensive handling of missing values in time series data.
- Generation of date-related features and lagged values for autoregressive features.
- Calculation of rolling and expanding window statistics.
- Seasonal decomposition to identify underlying trends and patterns.
- Fourier transformations for capturing cyclical behaviors.
- Holiday feature integration for recognizing potential anomalies on holidays.
- A systematic approach to feature engineering using a configuration dictionary.

### Method Overview

#### `Feature_Eng_Tech` Class
- `__init__(self, df, target_column)`: Constructor for the class requiring a DataFrame and a target column for feature generation.
- `reset_data(self)`: Resets the features to the original dataset state.
- `handle_missing_values(self, method)`: Handles missing values using the specified method.
- `add_date_features(self, include_day_of_week)`: Adds date-related features to the DataFrame.
- `add_lag_features(self, window)`: Generates lag features up to the specified window size.
- `add_rolling_features(self, window, min_periods)`: Calculates rolling window statistics.
- `add_expanding_window_features(self, min_periods)`: Generates expanding window statistics.
- `add_seasonal_decomposition(self, period, model)`: Decomposes the series into trend, seasonal, and residual components.
- `detrend_data(self)`: Removes linear trends from the series.
- `add_holiday_features(self)`: Marks federal US holidays in the dataset.
- `add_fourier_features(self, period, order)`: Adds sinusoidal features based on Fourier series expansion.
- `handle_nan_values_post_engineering(self, method)`: Cleans up NaN values after feature engineering.
- `feature_engineering(self, config)`: Applies a series of feature engineering steps based on a configuration.
- `get_engineered_data(self)`: Returns the DataFrame with the engineered features.

### Usage Example
```python
# Configuration for the feature engineering methods to apply.
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

# Uncomment the line below to execute the feature engineering process.
# data_eng = run_feature_engineering(True, config)
# After running, you can display the engineered DataFrame using:
# display(data_eng)
```

## `data_preprocessor.py` - Time Series Data Preprocessing

### Description

The `data_preprocessor.py` module encapsulates a comprehensive suite of data preprocessing functions for time series analysis. The `UnifiedDataPreprocessor` class within this module provides functionality for data splitting, normalization, transformation, and preparation for various types of machine learning models including recurrent neural networks and time series specific models like Prophet.

Key features of this module include:
- Splitting of time series data into training and testing sets with optional plotting.
- Multiple normalization techniques for feature scaling.
- Data differencing to make time series stationary.
- Box-Cox transformations to stabilize variance and make data more normal distribution-like.
- Reshaping of data for use in recurrent neural network architectures.
- Sequence generation for training sequence-to-sequence models.

### Method Overview

#### `UnifiedDataPreprocessor` Class
- `__init__(self, df, target_column, logger)`: Constructor for the class requiring a DataFrame and a target column for preprocessing.
- `get_scaler(self, scaler_type)`: Retrieves a scaler object based on the provided scaler type.
- `split_and_plot_data(self, test_size, split_date, plot)`: Splits the data into training and test sets and optionally plots the split data.
- `normalize_data(self, scaler_type, plot)`: Normalizes feature data using the specified scaler type and optionally plots the normalized features.
- `normalize_target(self, scaler_type, plot)`: Normalizes target data using the specified scaler type and optionally plots the normalized target.
- `difference_and_plot_data(self, interval, plot)`: Applies differencing to the data to make it stationary and optionally plots the differenced data.
- `box_cox_transform_and_plot(self, lambda_val, plot)`: Applies a Box-Cox transformation to the target variable and optionally plots the transformed data.
- `inverse_box_cox_and_plot(self, plot)`: Reverses the Box-Cox transformation on the target variable and optionally plots the inverse-transformed data.
- `reshape_for_recurrent(self, data)`: Reshapes the data into a format suitable for recurrent neural network models.
- `generate_sequences(self, X_data, y_data, n_steps, seq_to_seq)`: Generates sequences from the data for training sequence-to-sequence models.
- `prepare_data_for_recurrent(self, n_steps, seq_to_seq)`: Prepares the data for recurrent models by generating and reshaping sequences.
- `prepare_for_prophet(self)`: Prepares the data in a format suitable for use with the Prophet forecasting model.
- `get_preprocessed_data(self)`: Returns the preprocessed data splits.
- `__str__(self)`: Returns a string representation of the preprocessing steps applied.

### Usage Example
```python
# Assuming df is your DataFrame and 'Close' is your target column
preprocessor = UnifiedDataPreprocessor(df=df, target_column='Close')

# Split and plot data
preprocessor.split_and_plot_data(test_size=0.2, plot=True)

# Normalize features and target
preprocessor.normalize_data(scaler_type='MinMax', plot=True)
preprocessor.normalize_target(scaler_type='MinMax', plot=True)

# Differencing and plotting to achieve stationarity
preprocessor.difference_and_plot_data(interval=1, plot=True)

# Applying Box-Cox transformation and plotting
preprocessor.box_cox_transform_and_plot(lambda_val=None, plot=True)

# Preparing data for recurrent neural network models
n_steps = 10
seq_to_seq = False
X_train_seq, y_train_seq, X_test_seq, y_test_seq = preprocessor.prepare_data_for_recurrent(n_steps=n_steps, seq_to_seq=seq_to_seq)

# Preparing data for Prophet model
prophet_data = preprocessor.prepare_for_prophet()
```

## `model_ML.py` - Machine Learning Model Training and Evaluation

### Description

The `model_ML.py` module contains a set of classes that define various machine learning models. It provides functionality for training, predicting, and evaluating models, as well as for saving model predictions, accuracy metrics, and serialized model objects. It supports models such as Linear Regression, XGBoost, LightGBM, SVM, KNN, Random Forest, and Extra Trees.

Key features of this module include:
- Unified interface for different machine learning models.
- Integration with data preprocessing via the `UnifiedDataPreprocessor` class.
- Customizable model configurations and training procedures.
- Evaluation of model performance using standard metrics.
- Visualization of actual vs. predicted results.
- Storage and retrieval of model configurations and outcomes.

### Method Overview

#### Base Classes
- `BaseModel_ML`: A base class for various ML models, providing standard preprocessing, training, and evaluation functionality.
- `Linear_Regression`, `XGBoost`, `LightGBM`, `SVM`, `SVRegressor`, `KNN`, `RandomForest`, `ExtraTrees`: Inherited classes from `BaseModel_ML`, each encapsulating a specific ML algorithm.

#### Key Methods
- `train_model`: Trains the machine learning model on the provided dataset.
- `make_predictions`: Makes predictions using the trained model on both training and test datasets.
- `evaluate_model`: Evaluates the model's performance and returns a DataFrame of metrics.
- `plot_predictions`: Generates and displays plots comparing actual vs. predicted values for training and testing datasets.
- `save_predictions`: Saves the model's predictions to a CSV file.
- `save_accuracy`: Saves the model's accuracy metrics to a CSV file.
- `save_model_to_folder`: Serializes and saves the trained model to a specified directory.
- `generate_model_id`: Generates a unique identifier for the model based on its configuration and training time.

### Usage Example
```python
# Initialize data preprocessor with historical Bitcoin data and preprocess it
data_preprocessor = UnifiedDataPreprocessor(df=btc_data, target_column='Close')
data_preprocessor.split_and_plot_data(test_size=0.2, plot=False)
data_preprocessor.normalize_data(scaler_type='MinMax', plot=False)
data_preprocessor.normalize_target(scaler_type='MinMax', plot=False)

# Define model configurations
models = {
    'ML_LR': {
        'class': Linear_Regression,
        'config': {
            'regularization': 'ridge',
            'alpha': 1.0
        }
    },
    'ML_XGBoost': {
        'class': XGBoost,
        'config': {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 5
        }
    },
    # Add other models and their configurations here...
}

# Run the models and evaluate their performance
run_models(models)
```


## `model_DL.py` - Deep Learning Model Training and Evaluation

### Description

The `model_DL.py` module provides a framework for training and evaluating deep learning models for time series forecasting. It includes LSTM, GRU, Bidirectional LSTM (BiLSTM), Bidirectional GRU (BiGRU), SimpleRNN, Stacked RNN, LSTM with Attention, and CNN-LSTM models. The module leverages TensorFlow and Keras to build and train the models, and includes methods for model evaluation, plotting training history, and prediction visualization.

### Method Overview

#### Base Class
- `BaseModelLSTM`: A base class for LSTM-like models, handling model initialization, training, prediction, and evaluation.

#### Inherited Classes
- `LSTM_`: A class for building and training an LSTM model.
- `GRU_`: A class for GRU models.
- `Bi_LSTM`: A class for bidirectional LSTM models.
- `Bi_GRU`: A class for bidirectional GRU models.
- `Simple_RNN`: A class for Simple RNN models.
- `Stacked_RNN`: A class for models with a combination of LSTM and GRU layers.
- `Attention_LSTM`: A class for LSTM models with an attention mechanism.
- `CNN_LSTM`: A class for a hybrid CNN-LSTM model.

#### Key Methods
- `train_model`: Trains the model using the provided data, with options for cross-validation and early stopping.
- `make_predictions`: Generates predictions using the trained model on both training and test datasets.
- `evaluate_model`: Evaluates the model's performance and returns a DataFrame of metrics.
- `plot_history`: Plots the model's training history, showing loss over epochs.
- `plot_predictions`: Plots actual vs. predicted values for training and test sets.
- `save_model_to_folder`: Saves the trained model to disk.
- `generate_model_id`: Generates a unique identifier for the model based on its configuration.
- `save_predictions`: Saves model predictions to a CSV file.
- `save_accuracy`: Saves model evaluation metrics to a CSV file.

### Usage Example
```python
# Initialize data preprocessor with historical Bitcoin data and preprocess it
data_preprocessor = UnifiedDataPreprocessor(df=btc_data, target_column='Close')
data_preprocessor.split_and_plot_data(test_size=0.2, plot=False)
data_preprocessor.normalize_data(scaler_type='MinMax', plot=False)
data_preprocessor.normalize_target(scaler_type='MinMax', plot=False)
X_train_seq, y_train_seq, X_test_seq, y_test_seq = data_preprocessor.prepare_data_for_recurrent(n_steps=10, seq_to_seq=False)

# Define model configurations
models = {
    'DL_LSTM': {
        'class': LSTM_,  # Replace with your actual class
        'config': {
            # Model-specific configurations...
        },
        'skip': False
    },
    # Add other models and their configurations here...
}

# Run the models and evaluate their performance
run_models(models)
```



