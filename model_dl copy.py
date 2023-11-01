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

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, LSTM, TimeDistributed, Conv1D, MaxPooling1D, Flatten,
                                    ConvLSTM2D, BatchNormalization, GRU, Bidirectional, Attention, Input,
                                    Reshape, GlobalAveragePooling1D, GlobalMaxPooling1D, Lambda, LayerNormalization, 
                                    SimpleRNN, Layer, Multiply, Add, Activation)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1, l2, l1_l2
from keras_tuner import HyperModel, RandomSearch, BayesianOptimization
from tcn import TCN
from kerasbeats import NBeatsModel
from typing import List, Optional
from typing import Optional, List, Tuple
from statsmodels.tsa.stattools import acf, pacf


# Other settings
from IPython.display import display, HTML
import os, warnings, logging
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.3f}'.format)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


# LSTM Sequece-to-One
from data_preprocessor import UnifiedDataPreprocessor
from data_fetcher import btc_data

df = btc_data.copy()
data_preprocessor = UnifiedDataPreprocessor(df, target_column='Close')
data_preprocessor.split_and_plot_data(test_size=0.2, plot=False)
data_preprocessor.normalize_data(scaler_type='MinMax',plot=False)
data_preprocessor.normalize_target(scaler_type='MinMax',plot=False)
X_train_seq, y_train_seq, X_test_seq, y_test_seq = data_preprocessor.prepare_data_for_recurrent(n_steps=10, seq_to_seq=False)
print("LSTM Sequence-to-One Data Shapes:")
print("X_train_seq:", X_train_seq.shape,"y_train_seq:", y_train_seq.shape, "X_test_seq:", X_test_seq.shape, "y_test_seq:", y_test_seq.shape)
print("----")


class BaseModelLSTM():
    """
    A base class for LSTM-like machine learning models.
    This class handles data preprocessing, model training, predictions, and evaluations.
    """
    def __init__(self, model_type, data_preprocessor, config, cross_val=False):
        self._validate_input_sequence(data_preprocessor.X_train_seq, data_preprocessor.y_train_seq, data_preprocessor.X_test_seq, data_preprocessor.y_test_seq)
        self.X_train = data_preprocessor.X_train_seq
        self.y_train = data_preprocessor.y_train_seq
        self.X_test = data_preprocessor.X_test_seq
        self.y_test = data_preprocessor.y_test_seq
        self.feature_scaler = data_preprocessor.scalers['features']
        self.target_scaler = data_preprocessor.scalers['target']
        self.data = data_preprocessor.data
        self.config = config
        self.cross_val = cross_val
        self.model_type = model_type
        self.params = {'model_type': model_type}
        self.params.update(config)
        self._initialize_model()
        self.logger = logging.getLogger(__name__)

    def _initialize_model(self):
        logging.info(f"Initializing {self.model_type} model")
        self.model = Sequential()
        
        if self.model_type in ['LSTM', 'GRU']:
            for i, unit in enumerate(self.config['units']):
                return_sequences = True if i < len(self.config['units']) - 1 else False
                layer = LSTM(units=unit, return_sequences=return_sequences) if self.model_type == 'LSTM' else GRU(units=unit, return_sequences=return_sequences)
                self.model.add(layer)
                self.model.add(Dropout(self.config['dropout']))

        elif self.model_type == 'CNN-LSTM':
            self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=self.config['input_shape']))
            self.model.add(Dropout(self.config['dropout']))
            self.model.add(LSTM(units=self.config['units'][0]))

        self.model.add(Dense(units=self.config['dense_units']))
        self.model.compile(optimizer=self.config['optimizer'], loss='mean_squared_error')
        self.model.summary()
    
    def _validate_input_sequence(self, X_train, y_train, X_test, y_test):
        """Validate the shape and type of training and testing sequence data."""
        for arr, name in [(X_train, 'X_train_seq'), (y_train, 'y_train_seq'), (X_test, 'X_test_seq'), (y_test, 'y_test_seq')]:
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"{name} should be a numpy array.")

            if len(arr.shape) < 2:
                raise ValueError(f"{name} should have at least two dimensions.")

            # Special check for X_* arrays, which should be 3D for sequence models
            if 'X_' in name and len(arr.shape) != 3:
                raise ValueError(f"{name} should be a 3D numpy array for sequence models. Found shape {arr.shape}.")
     
    def train_model(self, epochs=100, batch_size=50, early_stopping=True):
        logging.info(f"Training {self.params['model_type']} model")
        callbacks = [EarlyStopping(monitor='val_loss', patience=10)] if early_stopping else None

        if self.cross_val:
            tscv = TimeSeriesSplit(n_splits=5)
            self.history = []
            fold_no = 1
            for train, val in tscv.split(self.X_train):
                logging.info(f"Training on fold {fold_no}")
                history = self.model.fit(self.X_train[train], self.y_train[train], epochs=epochs,
                                         batch_size=batch_size, validation_data=(self.X_train[val], self.y_train[val]),
                                         callbacks=callbacks, shuffle=False)
                self.history.append(history)
                logging.info(f"Done with fold {fold_no}")
                self.model.summary()
                fold_no += 1
        else:
            self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs,
                                          batch_size=batch_size, validation_split=0.2,
                                          callbacks=callbacks, shuffle=False)
        logging.info("Training completed")
        self.model.summary()

    def make_predictions(self):
        logging.info("Making predictions")

        self._make_raw_predictions()
        self._make_unscaled_predictions()
        self._create_comparison_dfs()
        model_id = self.generate_model_id()
        self.save_predictions(model_id)

        logging.info("Predictions made")

    def _make_raw_predictions(self):
        self.train_predictions = self.model.predict(self.X_train)
        self.test_predictions = self.model.predict(self.X_test)
        logging.info(f"Raw predictions made with shapes train: {self.train_predictions.shape}, test: {self.test_predictions.shape}")

    def _make_unscaled_predictions(self):
        # Check if the shape of the predictions matches that of y_train and y_test
        if self.train_predictions.shape[:-1] != self.y_train.shape[:-1]:
            logging.error(f"Shape mismatch: train_predictions {self.train_predictions.shape} vs y_train {self.y_train.shape}")
            return

        if self.test_predictions.shape[:-1] != self.y_test.shape[:-1]:
            logging.error(f"Shape mismatch: test_predictions {self.test_predictions.shape} vs y_test {self.y_test.shape}")
            return

        # If predictions are 3D, reduce dimensionality by taking mean along last axis
        if self.train_predictions.ndim == 3:
            self.train_predictions = np.mean(self.train_predictions, axis=-1)

        if self.test_predictions.ndim == 3:
            self.test_predictions = np.mean(self.test_predictions, axis=-1)

        # Perform the inverse transformation to get unscaled values
        self.train_predictions = self.target_scaler.inverse_transform(self.train_predictions).flatten()
        self.test_predictions = self.target_scaler.inverse_transform(self.test_predictions).flatten()

        logging.info(f"Unscaled predictions made with shapes train: {self.train_predictions.shape}, test: {self.test_predictions.shape}")

    def _create_comparison_dfs(self):
        y_train_flat = self.target_scaler.inverse_transform(self.y_train).flatten()
        y_test_flat = self.target_scaler.inverse_transform(self.y_test).flatten()

        # Obtain date indices from original data
        train_date_index = self.data.index[:len(self.y_train)]
        test_date_index = self.data.index[-len(self.y_test):]

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
            model_id = self.generate_model_id()
            self.save_accuracy(model_id)
            return self.evaluation_df
   
    def plot_history(self, plot=True):
        if not plot:
            return
        if not hasattr(self, 'history'):
            print("No training history is available. Train model first.")
            return
        # Extracting loss data from training history
        train_loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = list(range(1, len(train_loss) + 1))
        # Preparing data
        source = ColumnDataSource(data=dict(
            epochs=epochs,
            train_loss=train_loss,
            val_loss=val_loss
        ))

        p1 = figure(width=700, height=600, title="Training Loss over Epochs",x_axis_label='Epochs', y_axis_label='Loss')
        hover1 = HoverTool()
        hover1.tooltips = [("Epoch", "@epochs"), ("Loss", "@{train_loss}{0,0.0000}")]
        p1.add_tools(hover1)
        hover2 = HoverTool()
        hover2.tooltips = [("Epoch", "@epochs"), ("Validation Loss", "@{val_loss}{0,0.0000}")]
        p1.add_tools(hover2)
        p1.line(x='epochs', y='train_loss', legend_label="Training Loss", line_width=2, source=source, color="green")
        p1.line(x='epochs', y='val_loss', legend_label="Validation Loss", line_width=2, source=source, color="red")
        p1.legend.location = "top_right"
        p1.legend.click_policy = "hide"

        output_notebook()
        show(p1, notebook_handle=True)

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
        """
        Update the configuration hash mapping.
        
        Parameters:
            config_hash (str): The MD5 hash of the configuration.
            config (dict): The configuration dictionary.
            folder_name (str): The name of the folder where models are saved.
        """
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
        """
        Save the model to a specified folder.
        
        Parameters:
            version (str): The version of the model.
            folder_name (str): The name of the folder where models are saved.
        """
        model_name = self.__class__.__name__  # Remove 'Enhanced_' from the class name if needed
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        self.update_config_hash_mapping(config_hash, self.config, folder_name)

        # Save the model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{model_name}_V{version}_{config_hash}_{timestamp}.h5"
        full_path = os.path.join(folder_name, filename)
        self.model.save(full_path)
        print(f"Model saved to {full_path}")

    def generate_model_id(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
        model_id = f"{self.model_type}_{config_hash}_{timestamp}"
        return model_id

    def save_predictions(self, model_id):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        folder = 'model_predictions'
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, f"{model_id}_predictions.csv")
        
        df = self.test_comparison_df.reset_index()
        df['Model Class'] = self.__class__.__name__
        df['Model ID'] = model_id
        df['Config'] = json.dumps(self.config)
        df['Date Run'] = timestamp
        
        write_header = not os.path.exists(filepath)
        with open(filepath, 'a') as f:
            df.to_csv(f, header=write_header, index=False)

    def save_accuracy(self, model_id):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        folder = 'model_accuracy'
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, f"{model_id}_accuracy.csv")
        
        df = self.evaluation_df.reset_index()
        df['Model Class'] = self.__class__.__name__
        df['Model ID'] = model_id
        df['Config'] = json.dumps(self.config)
        df['Date Run'] = timestamp
        
        write_header = not os.path.exists(filepath)
        with open(filepath, 'a') as f:
            df.to_csv(f, header=write_header, index=False)


class LSTMModel(BaseModelLSTM):
    def _initialize_model(self):
        self.model = Sequential()
        additional_params = {
            'input_shape': self.config['input_shape'],
            'num_lstm_layers': self.config['num_lstm_layers'],
            'lstm_units': self.config['lstm_units']
        }
        self.params.update(additional_params)
        
        for i in range(self.config['num_lstm_layers']):
            units = self.config['lstm_units'][i]
            return_sequences = True if i < self.config['num_lstm_layers'] - 1 else False
            self.model.add(LSTM(units, return_sequences=return_sequences))
            self.model.add(Dropout(self.config['dropout']))
        for units in self.config['dense_units']:
            self.model.add(Dense(units))
        self.model.compile(optimizer=self.config['optimizer'], loss='mean_squared_error')


models = {
    'LSTM_Original': {
        'class': LSTMModel,
        'config': {
            'input_shape': (10, 5),
            'num_lstm_layers': 2,
            'lstm_units': [50, 30],
            'dropout': 0.2,
            'dense_units': [1],
            'optimizer': 'adam'
        },
        'skip': False
    },
    'LSTM_SingleLayer_NoDropout': {
        'class': LSTMModel,
        'config': {
            'input_shape': (10, 5),
            'num_lstm_layers': 1,
            'lstm_units': [64],
            'dropout': 0.0,
            'dense_units': [1],
            'optimizer': 'adam'
        },
        'skip': False
    },
    'LSTM_ThreeLayer_VariedUnits': {
        'class': LSTMModel,
        'config': {
            'input_shape': (10, 5),
            'num_lstm_layers': 3,
            'lstm_units': [64, 32, 16],
            'dropout': 0.3,
            'dense_units': [1],
            'optimizer': 'adam'
        },
        'skip': False
    },
    'LSTM_TwoLayer_ExtraDense': {
        'class': LSTMModel,
        'config': {
            'input_shape': (10, 5),
            'num_lstm_layers': 2,
            'lstm_units': [50, 30],
            'dropout': 0.2,
            'dense_units': [16, 1],
            'optimizer': 'adam'
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
        model.train_model(epochs=100, batch_size=32)
        model.make_predictions()
        evaluation_df = model.evaluate_model()
        display(evaluation_df)
        print(f"{name} Model Evaluation:\n", evaluation_df)
        model.plot_history()
        model.plot_predictions()
        model.save_model_to_folder(version="2")



# Run all models
run_models(models)

# Run only specific models
#run_models(models, run_only=['StackedRNN','AttentionLSTM','CNNLSTM'])

# Skip specific models
#run_models(models, skip=['SimpleRNN'])












