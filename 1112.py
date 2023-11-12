import logging
import numpy as np
import pandas as pd
import json
import pickle
import hashlib
import joblib
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
    def __init__(self, model_type, data_preprocessor, config):
        self.model_type = model_type
        self._validate_input(data_preprocessor.series, data_preprocessor.test_size)
        self.series = data_preprocessor.series
        self.test_size = data_preprocessor.test_size
        self.config = config
        self.params = {'model_type': model_type}
        self.params.update(config)
        self._initialize_model()
        self.logging = logging.getLogger(__name__)

        # Split the data into train and test series
        split_idx = int(len(self.series) * (1 - self.test_size))
        self.train_series = self.series[:split_idx]
        self.test_series = self.series[split_idx:]

    def _validate_input(self, series):
        """Validate the shape and type of the time series data."""
        if not isinstance(series, pd.Series):
            raise ValueError("The series should be a pandas Series.")
        if series.isnull().any():
            raise ValueError("The series contains missing values. Handle missing data before modeling.")
        if len(series.shape) != 1:
            raise ValueError("The series should be a 1D pandas Series.")




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

