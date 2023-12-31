from IPython.display import display, HTML
import json, joblib, hashlib, logging, os, warnings
import numpy as np, pandas as pd
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
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.3f}'.format)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class BaseModel_ML:
    """
    A base class for machine learning models.
    This class handles data preprocessing, model training, predictions, and evaluations.
    """
    def __init__(self, model_type, data_preprocessor, config):
        self.model_type = model_type
        self._validate_input(data_preprocessor.X_train, data_preprocessor.y_train, data_preprocessor.X_test, data_preprocessor.y_test)
        self.X_train = data_preprocessor.X_train
        self.y_train = data_preprocessor.y_train
        self.X_test = data_preprocessor.X_test
        self.y_test = data_preprocessor.y_test
        self.feature_scaler = data_preprocessor.scalers['features']
        self.target_scaler = data_preprocessor.scalers['target']
        self.data = data_preprocessor.data
        self.config = config
        self.params = {'model_type': model_type}
        self.params.update(config)  # Add other config parameters to the params dictionary
        self._initialize_model()
        self.logging = logging.getLogger(f"{self.model_type}_model")
    
    def _validate_input(self, X_train, y_train, X_test, y_test):
        """Validate the shape and type of training and testing data."""
        for arr, name in [(X_train, 'X_train'), (y_train, 'y_train'), (X_test, 'X_test'), (y_test, 'y_test')]:
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"{name} should be a numpy array.")
            
            if len(arr.shape) != 2:
                raise ValueError(f"{name} should be a 2D numpy array for ML models. Found shape {arr.shape}.")

    def train_model(self, cross_val=False, n_splits=5):
        logging.info(f"Training {self.model_type} model")

        if cross_val:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            fold_no = 1
            for train_index, val_index in tscv.split(self.X_train):
                logging.info(f"Training on fold {fold_no}")
                # Split the data into training and validation sets for this fold
                X_train_fold, X_val_fold = self.X_train[train_index], self.X_train[val_index]
                y_train_fold, y_val_fold = self.y_train[train_index], self.y_train[val_index]

                self.model.fit(X_train_fold, y_train_fold)
                val_score = self.model.score(X_val_fold, y_val_fold)
                logging.info(f"Validation score for fold {fold_no}: {val_score}")

                fold_no += 1
        else:
            self.model.fit(self.X_train, self.y_train)

        logging.info("Training completed")

    def make_predictions(self):
        logging.info("Making predictions")

        self._make_raw_predictions()
        self._make_unscaled_predictions()
        self._create_comparison_dfs()

        logging.info("Predictions made")

    def _make_raw_predictions(self):
        self.train_predictions = self.model.predict(self.X_train)
        self.test_predictions = self.model.predict(self.X_test)
        logging.info(f"Raw predictions made with shapes train: {self.train_predictions.shape}, test: {self.test_predictions.shape}")

    def _make_unscaled_predictions(self):
        # Check if the shape of the predictions matches that of y_train and y_test
        if self.train_predictions.ndim > 1 and self.train_predictions.shape[1] != 1:
            logging.error(f"Unexpected number of columns in train_predictions: {self.train_predictions.shape[1]}")
            return

        if self.test_predictions.ndim > 1 and self.test_predictions.shape[1] != 1:
            logging.error(f"Unexpected number of columns in test_predictions: {self.test_predictions.shape[1]}")
            return

        # If predictions are 2D, flatten to 1D
        if self.train_predictions.ndim == 2:
            self.train_predictions = self.train_predictions.flatten()

        if self.test_predictions.ndim == 2:
            self.test_predictions = self.test_predictions.flatten()

        # Perform the inverse transformation to get unscaled values if required
        if self.target_scaler:
            self.train_predictions = self.target_scaler.inverse_transform(self.train_predictions.reshape(-1, 1)).flatten()
            self.test_predictions = self.target_scaler.inverse_transform(self.test_predictions.reshape(-1, 1)).flatten()

        logging.info(f"Unscaled predictions made with shapes train: {self.train_predictions.shape}, test: {self.test_predictions.shape}")

    def _create_comparison_dfs(self):
        # Check if the target scaler was used and inverse transform if necessary
        if self.target_scaler:
            y_train_flat = self.target_scaler.inverse_transform(self.y_train.reshape(-1, 1)).flatten()
            y_test_flat = self.target_scaler.inverse_transform(self.y_test.reshape(-1, 1)).flatten()
        else:
            y_train_flat = self.y_train
            y_test_flat = self.y_test

        # Obtain date indices from original data
        train_date_index = self.data.index[:len(y_train_flat)]
        test_date_index = self.data.index[-len(y_test_flat):]

        if y_train_flat.shape != self.train_predictions.shape:
            logging.error(f"Shape mismatch between y_train {y_train_flat.shape} and train_predictions {self.train_predictions.shape}")
        else:
            self.train_comparison_df = pd.DataFrame({'Actual': y_train_flat, 'Predicted': self.train_predictions})
            # Set date index for train_comparison_df
            self.train_comparison_df.set_index(train_date_index, inplace=True)

        if y_test_flat.shape != self.test_predictions.shape:
            logging.error(f"Shape mismatch between y_test {y_test_flat.shape} and test_predictions {self.test_predictions.shape}")
        else:
            self.test_comparison_df = pd.DataFrame({'Actual': y_test_flat, 'Predicted': self.test_predictions})
            # Set date index for test_comparison_df
            self.test_comparison_df.set_index(test_date_index, inplace=True)

    def evaluate_model(self):
            logging.info("Evaluating LSTM model")
            metrics = {'RMSE': mean_squared_error, 'R2 Score': r2_score,
                       'MAE': mean_absolute_error, 'Explained Variance': explained_variance_score}

            evaluation = {}
            for name, metric in metrics.items():
                if name == 'RMSE':
                    train_evaluation = metric(self.train_comparison_df['Actual'],
                                              self.train_comparison_df['Predicted'],
                                              squared=False)
                    test_evaluation = metric(self.test_comparison_df['Actual'],
                                             self.test_comparison_df['Predicted'],
                                             squared=False)
                else:
                    train_evaluation = metric(self.train_comparison_df['Actual'],
                                              self.train_comparison_df['Predicted'])
                    test_evaluation = metric(self.test_comparison_df['Actual'],
                                             self.test_comparison_df['Predicted'])
                evaluation[name] = {'Train': train_evaluation, 'Test': test_evaluation}

            self.evaluation_df = pd.DataFrame(evaluation)
            logging.info("Evaluation completed")
            return self.evaluation_df
    
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
        return(show(row(p2, p3), notebook_handle=True))
    
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




class Linear_Regression(BaseModel_ML):
    """
    Enhanced Linear Regression model supporting Ridge and Lasso regularization.
    Inherits from BaseModel_ML.
    """
    def _initialize_model(self):
        # Set up the regression model based on the configuration
        regularization = self.config.get('regularization', 'none').lower()
        alpha = self.config.get('alpha', 1.0)
        if regularization == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif regularization == 'lasso':
            self.model = Lasso(alpha=alpha)
        else:
            self.model = LinearRegression()
        
        # Log the initialization with model type and regularization type
        logging.info(f"{self.model_type} model initialized with {regularization} regularization and alpha={alpha}")

        # Update params with specific parameters of the regression model
        self.params.update({'regularization': regularization, 'alpha': alpha})
    
class XGBoost(BaseModel_ML):
    """
    Enhanced XGBoost model that inherits from BaseModel_ML.
    """
    def _initialize_model(self):
        """
        Initialize the XGBoost model with parameters from the config.
        """
        # Set up the XGBoost model based on the configuration
        self.model = xgb.XGBRegressor(**self.config)
        
        logging.info(f"{self.model_type} model initialized with configuration: {self.config}")
        self.params.update(self.config)

class LightGBM(BaseModel_ML):
    """
    Enhanced LightGBM model that inherits from BaseModel_ML.
    """
    def _initialize_model(self):
        # Initialize LightGBM model with config parameters
        self.model = LGBMRegressor(**self.config)
        logging.info(f"{self.model_type} model initialized with configuration: {self.config}")
        self.params.update(self.config)

class SVM(BaseModel_ML):
    """
    Enhanced SVM model that inherits from BaseModel_ML.
    """
    def _initialize_model(self):
        # Initialize SVM model with config parameters
        self.model = SVR(**self.config)
        logging.info(f"{self.model_type} model initialized with configuration: {self.config}")
        self.params.update(self.config)

class SVRegressor(BaseModel_ML):
    """
    Enhanced SVR model that inherits from BaseModel_ML.
    """
    def _initialize_model(self):
        # Initialize SVR model with config parameters
        self.model = SVR(**self.config)
        logging.info(f"{self.model_type} model initialized with configuration: {self.config}")
        self.params.update(self.config)

class KNN(BaseModel_ML):
    """
    Enhanced KNN model that inherits from BaseModel_ML.
    """
    def _initialize_model(self):
        # Initialize KNN model with config parameters
        self.model = KNeighborsRegressor(**self.config)
        logging.info(f"{self.model_type} model initialized with configuration: {self.config}")
        self.params.update(self.config)

class RandomForest(BaseModel_ML):
    """
    Enhanced Random Forest model that inherits from BaseModel_ML.
    """
    def _initialize_model(self):
        # Initialize Random Forest model with config parameters
        self.model = RandomForestRegressor(**self.config)
        logging.info(f"{self.model_type} model initialized with configuration: {self.config}")
        self.params.update(self.config)

    # The feature_importance method can be kept as is if it provides additional functionality specific to Random Forest.
    def feature_importance(self):
        """
        Extract feature importance scores.
        """
        try:
            importance_scores = self.model.feature_importances_
            logging.info("Feature importance scores extracted.")
            return importance_scores
        except Exception as e:
            logging.error(f"Error occurred while extracting feature importance: {str(e)}")

class ExtraTrees(BaseModel_ML):
    """
    Enhanced Extra Trees model that inherits from BaseModel_ML.
    """
    def _initialize_model(self):
        # Initialize Extra Trees model with config parameters
        self.model = ExtraTreesRegressor(**self.config)
        logging.info(f"{self.model_type} model initialized with configuration: {self.config}")
        self.params.update(self.config)


from data_fetcher import btc_data
from data_preprocessor import UnifiedDataPreprocessor
df = btc_data.copy()

data_preprocessor = UnifiedDataPreprocessor(df, target_column='Close')
data_preprocessor.split_and_plot_data(test_size=0.2, plot=False)
data_preprocessor.normalize_data(scaler_type='MinMax', plot=False)
data_preprocessor.normalize_target(scaler_type='MinMax', plot=False)


models = {
    'ML_LR': {
        'class': Linear_Regression,
        'config': {
            'regularization': 'ridge',
            'alpha': 1.0
        },
        'skip': False
    },
    'ML_XGBoost': {
        'class': XGBoost,
        'config': {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 5
        },
        'skip': False
    },
    'ML_LightGBM': {
        'class': LightGBM,
        'config': {
            'objective': 'regression',
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 5
        },
        'skip': False
    },
    'ML_SVM': {
        'class': SVM,
        'config': {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1
        },
        'skip': False
    },
    'ML_SVRegressor': {
        'class': SVRegressor,
        'config': {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1
        },
        'skip': False
    },
    'ML_KNN': {
        'class': KNN,
        'config': {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto'
        },
        'skip': False
    },
    'ML_RandomForest': {
        'class': RandomForest,
        'config': {
            'n_estimators': 100,
            'criterion': 'poisson',
            'max_depth': None
        },
        'skip': False
    },
    'ML_ExtraTrees': {
        'class': ExtraTrees,
        'config': {
            'n_estimators': 100,
            'criterion': 'squared_error',
            'max_depth': None
        },
        'skip': False
    }
}

def run_models(models, run_only=None, skip=None):
    for name, model_info in models.items():
        if run_only and name not in run_only:
            continue
        if skip and name in skip:
            continue
        if model_info.get('skip'):
            continue

        model_class = model_info['class']
        config = model_info['config']
        
        model = model_class(data_preprocessor=data_preprocessor, config=config, model_type=name)
        model.train_model()
        model.make_predictions()
        evaluation_df = model.evaluate_model()
        display(evaluation_df)
        model.plot_predictions(plot=True)
                
        # Generate a unique model_id for this run
        model_id = model.generate_model_id()
        model.save_predictions(model_id, subfolder='model_machine_learning', overwrite=False)
        model.save_accuracy(model_id, subfolder='model_machine_learning', overwrite=False)
        model.save_model_to_folder(version="1")


# Run all models
run_models(models)

# Run only specific models
#run_models(models, run_only=['Baby_Cow', 'Baby_coooow'])

# Skip specific models
#run_models(models, data_preprocessor, skip=['Linear_Regression'])