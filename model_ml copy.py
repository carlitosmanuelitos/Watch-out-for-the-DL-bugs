# Need to refactor the entire class to work more similar to the dl class with the train model, make predictions etc within the base class

import logging
import numpy as np
import pandas as pd
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from joblib import dump, load
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_notebook, save
from bokeh.models import (HoverTool, ColumnDataSource, WheelZoomTool, Span, Range1d,
                          FreehandDrawTool, MultiLine, NumeralTickFormatter, Button, CustomJS)
from bokeh.layouts import column, row
from bokeh.io import curdoc, export_png
from bokeh.models.widgets import CheckboxGroup
from bokeh.themes import Theme
# Machine Learning Libraries
import sklearn
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
import xgboost as xgb



# Other settings
from IPython.display import display, HTML
import os, warnings, logging
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.3f}'.format)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


class BaseModel_ML:
    """
    A base class for machine learning models.
    This class handles data preprocessing, model training, predictions, and evaluations.
    
    - Linear Regression
    - XGBoost
    - LightGBM
    - KNN
    - SVM
    - Random Forest
    """
    def __init__(self, data_preprocessor, config, plot=True):
        self._validate_input(data_preprocessor.X_train, data_preprocessor.y_train, data_preprocessor.X_test, data_preprocessor.y_test)
        self.X_train = data_preprocessor.X_train
        self.y_train = data_preprocessor.y_train
        self.X_test = data_preprocessor.X_test
        self.y_test = data_preprocessor.y_test
        self.feature_scaler = data_preprocessor.scalers['features']
        self.target_scaler = data_preprocessor.scalers['target']
        self.data = data_preprocessor.data
        self.model_type = self.__class__.__name__  # or some other logic to set model_type
        self.config = config
        self.plot = plot
        self.logger = logging.getLogger(__name__)    
    
    def _validate_input(self, X_train, y_train, X_test, y_test):
        """Validate the shape and type of training and testing data."""
        for arr, name in [(X_train, 'X_train'), (y_train, 'y_train'), (X_test, 'X_test'), (y_test, 'y_test')]:
            if not isinstance(arr, np.ndarray) or len(arr.shape) != 2:
                raise ValueError(f"{name} should be a 2D numpy array.")
                
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
        self.logger.info(f"Configuration mapping updated in {folder_name}")

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

        # Save the model
        model_id = self.generate_model_id()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.__class__.__name__}_V{version}_{model_id}.h5"
        full_path = os.path.join(folder_name, filename)
        self.model.save(full_path)
        self.logger.info(f"Model saved to {full_path}")

    def generate_model_id(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
        model_id = f"{self.model_type}_{config_hash}"
        self.logger.info(f"Generated model ID: {model_id}")
        return model_id

    def save_predictions(self, model_id, subfolder=None, overwrite=False):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        folder = 'model_predictions'
        if subfolder:
            folder = os.path.join(folder, subfolder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, 'all_model_predictions.csv')
        
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
        self.logger.info(f"Predictions saved to {filepath}" if overwrite or not os.path.exists(filepath) else f"Predictions appended to {filepath}")

    def save_accuracy(self, model_id, subfolder=None, overwrite=False):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        folder = 'model_accuracy'
        if subfolder:
            folder = os.path.join(folder, subfolder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, 'all_model_accuracy.csv')
        
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
        self.logger.info(f"Accuracy metrics saved to {filepath}" if overwrite or not os.path.exists(filepath) else f"Accuracy metrics appended to {filepath}")
        


class Enhanced_Linear_Regression(BaseModel_ML):
    """
    Initialize the Enhanced_Linear_Regression model.
    Supports Ridge and Lasso regularization.
    """
    def __init__(self, data_preprocessor, config, plot=True):
        super().__init__(data_preprocessor, config, plot)
        self._initialize_model()

    def _initialize_model(self):
        """Choose the regression model based on the configuration."""
        if self.config['regularization'] == 'ridge':
            self.model = Ridge(alpha=self.config['alpha'])
            self.logger.info("Ridge regression model initialized.")
        elif self.config['regularization'] == 'lasso':
            self.model = Lasso(alpha=self.config['alpha'])
            self.logger.info("Lasso regression model initialized.")
        else:
            self.model = LinearRegression()
            self.logger.info("Plain Linear Regression model initialized.")

    def train_model(self):
        """Train the Linear Regression model."""
        try:
            self.model.fit(self.X_train, self.y_train)
            self.logger.info("Linear Regression model trained successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while training the model: {str(e)}")

    def make_predictions(self):
        """Make predictions using the trained model for training and test sets."""
        try:
            self.train_predictions = self.model.predict(self.X_train)
            self.test_predictions = self.model.predict(self.X_test)
            self.logger.info("Predictions made successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while making predictions: {str(e)}")

class Enhanced_XGBoost(BaseModel_ML):
    def __init__(self, data_preprocessor, config, plot=True):
        super().__init__(data_preprocessor, config, plot)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the XGBoost model based on the configuration."""
        self.model = xgb.XGBRegressor(**self.config)
        self.logger.info("XGBoost model initialized.")
        
    def train_model(self):
        """Train the XGBoost model."""
        try:
            self.model.fit(self.X_train, self.y_train)
            self.logger.info("XGBoost model trained successfully")
        except Exception as e:
            self.logger.error(f"Error occurred while training the model: {str(e)}")

    def make_predictions(self):
        """Make predictions using the trained model for training and test sets."""
        try:
            self.train_predictions = self.model.predict(self.X_train)
            self.test_predictions = self.model.predict(self.X_test)
            self.logger.info("Predictions made successfully for both training and test data")
        except Exception as e:
            self.logger.error(f"Error occurred while making predictions: {str(e)}")        

class Enhanced_LightGBM(BaseModel_ML):
    """
    Initialize the Enhanced LightGBM model.
    """
    def __init__(self, data_preprocessor, config, plot=True):
        super().__init__(data_preprocessor, config, plot)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the LightGBM model based on the configuration."""
        self.model = LGBMRegressor(**self.config)
        self.logger.info("LightGBM model initialized.")

    def train_model(self):
        try:
            self.model.fit(self.X_train, self.y_train)
            self.logger.info("LightGBM model trained successfully")
        except Exception as e:
            self.logger.error(f"Error occurred while training the model: {str(e)}")

    def make_predictions(self):
        """Make predictions using the trained model for training and test sets."""
        try:
            self.train_predictions = self.model.predict(self.X_train)
            self.test_predictions = self.model.predict(self.X_test)
            self.logger.info("Predictions made successfully for both training and test data")
        except Exception as e:
            self.logger.error(f"Error occurred while making predictions: {str(e)}")

class Enhanced_SVM(BaseModel_ML):
    """
    Initialize the Enhanced SVM model.
    """
    def __init__(self, data_preprocessor, config, plot=True):
        super().__init__(data_preprocessor, config, plot)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the SVM model based on the configuration."""
        self.model = SVR(**self.config)
        self.logger.info("SVM model initialized.")

    def train_model(self):
        """Train the SVM model."""
        try:
            self.model.fit(self.X_train, self.y_train.ravel())  # ravel() to convert y_train to 1D for SVM
            self.logger.info("SVM model trained successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while training the model: {str(e)}")

    def make_predictions(self):
        """Make predictions using the trained model."""
        try:
            self.train_predictions = self.model.predict(self.X_train)
            self.test_predictions = self.model.predict(self.X_test)
            self.logger.info("Predictions made successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while making predictions: {str(e)}")

class Enhanced_KNN(BaseModel_ML):
    """
    Initialize the Enhanced KNN model.
    """
    def __init__(self, data_preprocessor, config, plot=True):
        super().__init__(data_preprocessor, config, plot)
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the KNN model based on the configuration.
        """
        self.model = KNeighborsRegressor(**self.config)
        self.logger.info("KNN model initialized.")

    def train_model(self):
        """
        Train the KNN model.
        """
        try:
            self.model.fit(self.X_train, self.y_train.ravel())  # ravel() to convert y_train to 1D for KNN
            self.logger.info("KNN model trained successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while training the model: {str(e)}")

    def make_predictions(self):
        """Make predictions using the trained model."""
        try:
            self.train_predictions = self.model.predict(self.X_train)
            self.test_predictions = self.model.predict(self.X_test)
            self.logger.info("Predictions made successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while making predictions: {str(e)}")

class Enhanced_RandomForest(BaseModel_ML):
    """
    A class for an enhanced Random Forest Regression model.
    Inherits from the BaseModel class.
    """
    def __init__(self, data_preprocessor, config, plot=True):
        super().__init__(data_preprocessor, config, plot)
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the Random Forest model based on the configuration.
        """
        self.model = RandomForestRegressor(**self.config)
        self.logger.info("Random Forest model initialized.")

    def feature_importance(self):
        """
        Extract feature importance scores.
        """
        try:
            importance_scores = self.model.feature_importances_
            self.logger.info("Feature importance scores extracted.")
            return importance_scores
        except Exception as e:
            self.logger.error(f"Error occurred while extracting feature importance: {str(e)}")
            
    def train_model(self):
        """Make predictions using the trained model for training and test sets."""
        try:
            self.model.fit(self.X_train, self.y_train.ravel())  # Using ravel() to fit the expected shape
            self.logger.info("RandomForest model trained successfully")
        except Exception as e:
            self.logger.error(f"Error occurred while training the model: {str(e)}")

    def make_predictions(self):
        """Make predictions using the trained model."""
        try:
            self.train_predictions = self.model.predict(self.X_train)
            self.test_predictions = self.model.predict(self.X_test)
            self.logger.info("Predictions made successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while making predictions: {str(e)}")

class Enhanced_SVR(BaseModel_ML):
    """
    Initialize the Enhanced SVR model.
    """
    def __init__(self, data_preprocessor, config, plot=True):
        super().__init__(data_preprocessor, config, plot)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the SVR model based on the configuration."""
        self.model = SVR(**self.config)
        self.logger.info("SVR model initialized.")
        
    def train_model(self):
        """Train the model."""
        try:
            self.model.fit(self.X_train, self.y_train.ravel())  # Using ravel() to fit the expected shape for some models
            self.logger.info(f"{self.__class__.__name__} model trained successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while training the model: {str(e)}")

    def make_predictions(self):
        """Make predictions using the trained model."""
        try:
            self.train_predictions = self.model.predict(self.X_train)
            self.test_predictions = self.model.predict(self.X_test)
            self.logger.info("Predictions made successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while making predictions: {str(e)}")

class Enhanced_ExtraTrees(BaseModel_ML):
    """
    Initialize the Enhanced Extra Trees model.
    """
    def __init__(self, data_preprocessor, config, plot=True):
        super().__init__(data_preprocessor, config, plot)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Extra Trees model based on the configuration."""
        self.model = ExtraTreesRegressor(**self.config)
        self.logger.info("Extra Trees model initialized.")
        
    def train_model(self):
        """Train the model."""
        try:
            self.model.fit(self.X_train, self.y_train.ravel())  # Using ravel() to fit the expected shape for some models
            self.logger.info(f"{self.__class__.__name__} model trained successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while training the model: {str(e)}")

    def make_predictions(self):
        """Make predictions using the trained model."""
        try:
            self.train_predictions = self.model.predict(self.X_train)
            self.test_predictions = self.model.predict(self.X_test)
            self.logger.info("Predictions made successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while making predictions: {str(e)}")




from data_fetcher import btc_data
from data_preprocessor import UnifiedDataPreprocessor
df = btc_data.copy()

data_preprocessor = UnifiedDataPreprocessor(df, target_column='Close')
data_preprocessor.split_and_plot_data(test_size=0.2, plot=False)
data_preprocessor.normalize_data(scaler_type='MinMax', plot=False)
data_preprocessor.normalize_target(scaler_type='MinMax', plot=False)

models = {
    'Enhanced_Linear_Regression': {
        'class': Enhanced_Linear_Regression,
        'config': {
            'regularization': 'ridge',
            'alpha': 1.0
        },
        'skip': False
    },
    'Enhanced_XGBoost': {
        'class': Enhanced_XGBoost,
        'config': {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 5
        },
        'skip': False
    },
    'Enhanced_LightGBM': {
        'class': Enhanced_LightGBM,
        'config': {
            'objective': 'regression',
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 5
        },
        'skip': False
    },
    'Enhanced_SVM': {
        'class': Enhanced_SVM,
        'config': {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1
        },
        'skip': False
    },
    'Enhanced_SVR': {
        'class': Enhanced_SVR,
        'config': {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1
        },
        'skip': False
    },
    'Enhanced_KNN': {
        'class': Enhanced_KNN,
        'config': {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto'
        },
        'skip': False
    },
    'Enhanced_RandomForest': {
        'class': Enhanced_RandomForest,
        'config': {
            'n_estimators': 100,
            'criterion': 'poisson',
            'max_depth': None
        },
        'skip': False
    },
    'Enhanced_ExtraTrees': {
        'class': Enhanced_ExtraTrees,
        'config': {
            'n_estimators': 100,
            'criterion': 'squared_error',
            'max_depth': None
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
        
        model = model_class(data_preprocessor, config, plot=True)
        model.train_model()
        model.make_predictions()
        model.inverse_scale_predictions()
        train_comparison_df, test_comparison_df = model.compare_predictions()
        evaluation_results = model.evaluate_model()
        # Generate a unique model_id for this run
        model_id = model.generate_model_id()
        model.save_predictions(model_id, subfolder='model_machine_learning', overwrite=False)
        model.save_accuracy(model_id, subfolder='model_machine_learning', overwrite=False)

        display(evaluation_results)
        model.plot_predictions()
        model.save_model_to_folder(version="2")


# Run all models
run_models(models, data_preprocessor)

# Run only specific models
#run_models(models, data_preprocessor, run_only=['Enhanced_Linear_Regression', 'Enhanced_XGBoost'])

# Skip specific models
#run_models(models, data_preprocessor, skip=['Enhanced_Linear_Regression'])
