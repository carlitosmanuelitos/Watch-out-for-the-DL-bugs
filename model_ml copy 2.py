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
        self.config = config
        self.train_predictions = None
        self.test_predictions = None
        self.plot = plot
        self.logger = logging.getLogger(__name__)    
    
    def _validate_input(self, X_train, y_train, X_test, y_test):
        """Validate the shape and type of training and testing data."""
        for arr, name in [(X_train, 'X_train'), (y_train, 'y_train'), (X_test, 'X_test'), (y_test, 'y_test')]:
            if not isinstance(arr, np.ndarray) or len(arr.shape) != 2:
                raise ValueError(f"{name} should be a 2D numpy array.")

    def train_model(self):
        """Train the model."""
        try:
            self.model.fit(self.X_train, self.y_train.ravel())
            self.logger.info(f"{self.__class__.__name__} model trained successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while training the model: {str(e)}")
                
    def make_predictions(self):
        """Make predictions using the trained model."""
        try:
            self._make_raw_predictions()
            self.logger.info("Step 1: Raw predictions made.")
            
            self._make_unscaled_predictions()
            self.logger.info("Step 2: Unscaled predictions made.")
            
            self._create_comparison_dfs()
            self.logger.info("Step 3: Comparison dataframes created.")
            
            self.logger.info("Predictions made successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while making predictions: {str(e)}")
               
    def _make_raw_predictions(self):
        self.logger.debug(f"Sample raw train predictions: {self.train_predictions[:5]}")
        self.logger.debug(f"Sample raw test predictions: {self.test_predictions[:5]}")

        self.train_predictions = self.model.predict(self.X_train)
        self.test_predictions = self.model.predict(self.X_test)
        self.logger.info(f"Raw predictions made with shapes train: {self.train_predictions.shape}, test: {self.test_predictions.shape}")

    def _make_unscaled_predictions(self):
        # Perform the inverse transformation to get unscaled values
        self.logger.debug(f"Sample unscaled train predictions: {self.train_predictions[:5]}")
        self.logger.debug(f"Sample unscaled test predictions: {self.test_predictions[:5]}")

        self.train_predictions = self.target_scaler.inverse_transform(self.train_predictions.reshape(-1, 1)).flatten()
        self.test_predictions = self.target_scaler.inverse_transform(self.test_predictions.reshape(-1, 1)).flatten()
        self.logger.info(f"Unscaled predictions made with shapes train: {self.train_predictions.shape}, test: {self.test_predictions.shape}")

    def _create_comparison_dfs(self):
        # Log the attributes of the object to see what's available
        self.logger.debug(f"Object attributes: {dir(self)}")

        # Log the first few training and test values
        self.logger.debug(f"Sample actual train values: {self.y_train.flatten()[:5]}")
        self.logger.debug(f"Sample actual test values: {self.y_test.flatten()[:5]}")

        # Log sample train and test date indices
        self.logger.debug(f"Sample train date indices: {self.data.index[:5]}")
        self.logger.debug(f"Sample test date indices: {self.data.index[-5:]}")

        # Log the shape of y_train and y_test to ensure they are what you expect
        self.logger.debug(f"Shape of y_train: {self.y_train.shape}")
        self.logger.debug(f"Shape of y_test: {self.y_test.shape}")

        # Log the shape of train_predictions and test_predictions
        self.logger.debug(f"Shape of train_predictions: {self.train_predictions.shape}")
        self.logger.debug(f"Shape of test_predictions: {self.test_predictions.shape}")

        try:
            # Obtain date indices from original data
            train_date_index = self.data.index[:len(self.y_train)]
            test_date_index = self.data.index[-len(self.y_test):]

            self.train_comparison_df = pd.DataFrame({'Actual': self.y_train.flatten(), 'Predicted': self.train_predictions})
            self.train_comparison_df.set_index(train_date_index, inplace=True)

            self.test_comparison_df = pd.DataFrame({'Actual': self.y_test.flatten(), 'Predicted': self.test_predictions})
            self.test_comparison_df.set_index(test_date_index, inplace=True)

            # Log a sample of the generated dataframes
            self.logger.debug(f"Sample rows from train_comparison_df: {self.train_comparison_df.head()}")
            self.logger.debug(f"Sample rows from test_comparison_df: {self.test_comparison_df.head()}")

            self.logger.info("Comparison dataframes generated")
        except Exception as e:
            self.logger.error(f"Error occurred while generating comparison dataframes: {str(e)}")


    def evaluate_model(self):
        """Evaluate the model using various metrics for both training and test sets."""
        self.logger.debug(f"Sample rows from train_comparison_df: {self.train_comparison_df.head()}")
        self.logger.debug(f"Sample rows from test_comparison_df: {self.test_comparison_df.head()}")
        try:
            metrics = {
                'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                'R2 Score': r2_score,
                'MAE': mean_absolute_error,
                'Explained Variance': explained_variance_score
            }

            results = []
            for dataset, comparison_df in [('Train', self.train_comparison_df), ('Test', self.test_comparison_df)]:
                dataset_results = {metric_name: metric_func(comparison_df['Actual'], comparison_df['Predicted']) for metric_name, metric_func in metrics.items()}
                results.append(dataset_results)

            results_df = pd.DataFrame(results, index=['Train', 'Test'])
            return results_df
        except Exception as e:
            self.logger.error(f"Error occurred while evaluating the model: {str(e)}")

    def plot_predictions(self, plot=True):
        if not plot:
            return        
        if not hasattr(self, 'train_comparison_df') or not hasattr(self, 'test_comparison_df'):
            print("No predictions are available. Generate predictions first.")
            return
        actual_train = self.train_comparison_df['Actual']
        predicted_train = self.train_comparison_df['Predicted']
        actual_test = self.test_comparison_df['Actual']
        predicted_test = self.test_comparison_df['Predicted']
        index_train = self.train_comparison_df.index
        index_test = self.test_comparison_df.index

        # Preparing data
        source_train = ColumnDataSource(data=dict(
            index=index_train,
            actual_train=actual_train,
            predicted_train=predicted_train
        ))

        source_test = ColumnDataSource(data=dict(
            index=index_test,
            actual_test=actual_test,
            predicted_test=predicted_test
        ))

        p2 = figure(width=700, height=600, title="Training Data: Actual vs Predicted", x_axis_label='Date', y_axis_label='Value', x_axis_type="datetime")
        p3 = figure(width=700, height=600, title="Testing Data: Actual vs Predicted",x_axis_label='Date', y_axis_label='Value', x_axis_type="datetime")
        p2.line(x='index', y='actual_train', legend_label="Actual", line_width=2, source=source_train, color="green")
        p2.line(x='index', y='predicted_train', legend_label="Predicted", line_width=2, source=source_train, color="red")
        p3.line(x='index', y='actual_test', legend_label="Actual", line_width=2, source=source_test, color="green")
        p3.line(x='index', y='predicted_test', legend_label="Predicted", line_width=2, source=source_test, color="red")
        p2.legend.location = "top_left" 
        p2.legend.click_policy = "hide"
        p3.legend.location = "top_left" 
        p3.legend.click_policy = "hide"
        hover_train = HoverTool()
        hover_train.tooltips = [
            ("Date", "@index{%F}"),
            ("Actual Value", "@{actual_train}{0,0.0000}"),
            ("Predicted Value", "@{predicted_train}{0,0.0000}")
        ]
        hover_train.formatters = {"@index": "datetime"}

        hover_test = HoverTool()
        hover_test.tooltips = [
            ("Date", "@index{%F}"),
            ("Actual Value", "@{actual_test}{0,0.0000}"),
            ("Predicted Value", "@{predicted_test}{0,0.0000}")
        ]
        hover_test.formatters = {"@index": "datetime"}

        p2.add_tools(hover_train)
        p3.add_tools(hover_test)
        output_notebook()
        show(row(p2, p3), notebook_handle=True)
    
        
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

        BaseModel_ML.update_config_hash_mapping(config_hash, self.config, folder_name)

        # Save the model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{model_name}_V{version}_{config_hash}_{timestamp}.joblib"
        full_path = os.path.join(folder_name, filename)
        dump(self.model, full_path)
        self.logger.info(f"Model saved to {full_path}")
        


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

class Enhanced_XGBoost(BaseModel_ML):
    def __init__(self, data_preprocessor, config, plot=True):
        super().__init__(data_preprocessor, config, plot)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the XGBoost model based on the configuration."""
        self.model = xgb.XGBRegressor(**self.config)
        self.logger.info("XGBoost model initialized.")
        
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
        evaluation_df = model.evaluate_model()
        display(evaluation_df)
        model.plot_predictions()
        #model.save_model_to_folder(version="final")


# Run all models
#run_models(models, data_preprocessor)

# Run only specific models
run_models(models, data_preprocessor, run_only=['Enhanced_Linear_Regression'])

# Skip specific models
#run_models(models, data_preprocessor, skip=['Enhanced_Linear_Regression'])
