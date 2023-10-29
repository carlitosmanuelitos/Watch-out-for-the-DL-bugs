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
