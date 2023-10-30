import numpy as np
import pandas as pd
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from joblib import dump, load
from tensorflow.keras.models import save_model
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
n_steps = 10 
X_train_seq, y_train_seq, X_test_seq, y_test_seq = data_preprocessor.prepare_data_for_recurrent(n_steps, seq_to_seq=False)
print((data_preprocessor.X_train_seq).shape)
print((data_preprocessor.y_train_seq).shape)
print((data_preprocessor.X_test_seq).shape)
print((data_preprocessor.y_test_seq).shape)


print(hasattr(data_preprocessor, 'X_train_seq'))


class BaseModel_DL_SOA():
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
        self.logger = logging.getLogger(__name__)
    
    
    def _validate_input_sequence(self, X_train, y_train, X_test, y_test):
        """Validate the shape and type of training and testing sequence data."""
        for arr, name in [(X_train, 'X_train_seq'), (y_train, 'y_train_seq'), (X_test, 'X_test_seq'), (y_test, 'y_test_seq')]:
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"{name} should be a numpy array.")
            if len(arr.shape) < 2:
                raise ValueError(f"{name} should have at least two dimensions.")
            if 'X_' in name and len(arr.shape) != 3:
                raise ValueError(f"{name} should be a 3D numpy array for sequence models. Found shape {arr.shape}.")

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
            logging.info("Evaluating SOA models")
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

        # Check the type of model and save accordingly
        if hasattr(self.model, 'save'):
            # If the model has a 'save' method, use it (e.g., standard Keras models)
            self.model.save(full_path)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'save'):
            # If the model is a custom object with an inner Keras model
            save_model(self.model.model, full_path)
        else:
            print(f"Model of type {type(self.model)} cannot be saved.")

        print(f"Model saved to {full_path}")

class SOA_TCN(BaseModel_DL_SOA):
    def __init__(self, model_type, data_preprocessor, config):
        super().__init__(model_type, data_preprocessor, config)
        self._initialize_model()  # Call this here
    
    def _initialize_model(self):
        logger.info("Initializing the TCN model...")
        self.model = Sequential()

        # Add the TCN layer
        self.model.add(TCN(
            nb_filters=self.config.get('nb_filters', 64),
            kernel_size=self.config.get('kernel_size', 2),
            nb_stacks=self.config.get('nb_stacks', 1),
            dilations=self.config.get('dilations', [1, 2, 4, 8, 16, 32]),
            padding=self.config.get('padding', 'causal'),
            use_skip_connections=self.config.get('use_skip_connections', True),
            dropout_rate=self.config.get('dropout_rate', 0.2),
            return_sequences=self.config.get('return_sequences', False),
            activation=self.config.get('activation', 'relu'),
            kernel_initializer=self.config.get('kernel_initializer', 'he_normal'),
            use_batch_norm=self.config.get('use_batch_norm', False),
            input_shape=self.config['input_shape']
        ))

        self.model.add(Dense(1))  # Assuming regression task, adjust if needed
        self.model.compile(optimizer=self.config.get('optimizer', 'adam'), loss='mean_squared_error')
        
        logger.info("TCN model compiled successfully.")
        self.model.summary()

    def train_model(self, epochs=100, batch_size=50, early_stopping=True):
        logger.info(f"Training the TCN model for {epochs} epochs with batch size of {batch_size}...")
        callbacks = [EarlyStopping(monitor='val_loss', patience=10)] if early_stopping else None
        self.history = self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=epochs,
            batch_size=batch_size, 
            validation_split=0.2,
            callbacks=callbacks, 
            shuffle=False
        )
        logger.info("Training completed.")


class SOA_NBEATS(BaseModel_DL_SOA):
    def __init__(self, model_type, data_preprocessor, config):
        super(SOA_NBEATS, self).__init__(model_type, data_preprocessor, config)
        # Assign the sequences
        self.X_train = data_preprocessor.y_train_seq
        self.y_train = data_preprocessor.y_train_seq
        self.X_test = data_preprocessor.y_test_seq
        self.y_test = data_preprocessor.y_test_seq
        self._initialize_model()

    
    def _initialize_model(self):
        logger.info("Initializing the N-BEATS model...")
        
        # Initialize the N-BEATS model
        self.model = NBeatsModel(
            lookback=self.config.get('lookback', 7),
            horizon=self.config.get('horizon', 1),
            num_generic_neurons=self.config.get('num_generic_neurons', 512),
            num_generic_stacks=self.config.get('num_generic_stacks', 30),
            num_generic_layers=self.config.get('num_generic_layers', 4),
            num_trend_neurons=self.config.get('num_trend_neurons', 256),
            num_trend_stacks=self.config.get('num_trend_stacks', 3),
            num_trend_layers=self.config.get('num_trend_layers', 4),
            num_seasonal_neurons=self.config.get('num_seasonal_neurons', 2048),
            num_seasonal_stacks=self.config.get('num_seasonal_stacks', 3),
            num_seasonal_layers=self.config.get('num_seasonal_layers', 4),
            num_harmonics=self.config.get('num_harmonics', 1),
            polynomial_term=self.config.get('polynomial_term', 3),
            loss=self.config.get('loss', 'mae'),
            learning_rate=self.config.get('learning_rate', 0.001),
            batch_size=self.config.get('batch_size', 1024)
        )
        
        self.model.build_layer()
        self.model.build_model()

        # Compile the actual Keras model inside the NBeatsModel
        optimizer = self.config.get('optimizer', 'adam')
        loss = self.config.get('loss', 'mae')
        self.model.model.compile(optimizer=optimizer, loss=loss)

        logger.info("N-BEATS model initialized and compiled successfully.")
        self.model.model.summary()

    def train_model(self, epochs=100, batch_size=32, early_stopping=True, patience=10, val_split=0.2):
        logger.info(f"Training the N-BEATS model for {epochs} epochs with batch size of {batch_size}...")

        callbacks = []
        if early_stopping:
            es_callback = EarlyStopping(monitor='val_loss', patience=patience)
            callbacks.append(es_callback)

        # Use the actual Keras model inside the NBeatsModel for training
        self.history = self.model.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=callbacks
        )
        logger.info("Training completed.")

def wavenet_block(filters, kernel_size, dilation_rate):
    def f(input_):
        tanh_out = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal', activation='tanh')(input_)
        sigmoid_out = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal', activation='sigmoid')(input_)
        merged = Multiply()([tanh_out, sigmoid_out])
        out = Conv1D(filters=filters, kernel_size=1, activation='relu')(merged)
        skip = Conv1D(filters=filters, kernel_size=1)(out)
        
        # Adjust the residual connection's number of filters to match the input's filters
        input_residual = Conv1D(filters=filters, kernel_size=1)(input_)
        residual = Add()([skip, input_residual])
        
        return residual, skip
    return f
def build_wavenet_model(input_shape, num_blocks, filters, kernel_size):
    input_ = Input(shape=input_shape)
    x = input_
    skip_connections = []
    for dilation_rate in [2**i for i in range(num_blocks)]:
        x, skip = wavenet_block(filters, kernel_size, dilation_rate)(x)
        skip_connections.append(skip)
    x = Add()(skip_connections)
    x = Activation('relu')(x)
    x = Conv1D(filters=filters, kernel_size=1, activation='relu')(x)
    x = Conv1D(filters=filters, kernel_size=1)(x)
    x = Flatten()(x)
    output = Dense(1)(x)  # For regression; adjust if different task

    model = Model(input_, output)
    return model
class SOA_WAVENET(BaseModel_DL_SOA):
    
    def __init__(self, model_type, data_preprocessor, config):
        super(SOA_WAVENET, self).__init__(model_type, data_preprocessor, config)
        self.X_train = data_preprocessor.X_train_seq
        self.y_train = data_preprocessor.y_train_seq
        self.X_test = data_preprocessor.X_test_seq
        self.y_test = data_preprocessor.y_test_seq
        self._initialize_model()

    def _initialize_model(self):
        logger.info("Initializing the WaveNet model...")
        
        self.model = build_wavenet_model(
            input_shape=self.config['input_shape'],
            num_blocks=self.config.get('num_blocks', 4),
            filters=self.config.get('filters', 32),
            kernel_size=self.config.get('kernel_size', 2)
        )
        
        optimizer = self.config.get('optimizer', 'adam')
        loss = self.config.get('loss', 'mae')
        self.model.compile(optimizer=optimizer, loss=loss)
        
        logger.info("WaveNet model compiled successfully.")
        self.model.summary()

    def train_model(self, epochs=100, batch_size=32, early_stopping=True, patience=10, val_split=0.2):
        logger.info(f"Training the WaveNet model for {epochs} epochs with batch size of {batch_size}...")
        
        callbacks = []
        if early_stopping:
            es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
            callbacks.append(es_callback)
        
        self.history = self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=callbacks
        )
        logger.info("Training completed.")

class SOA_LSTNET(BaseModel_DL_SOA):
    def __init__(self, model_type, data_preprocessor, config):
        super(SOA_LSTNET, self).__init__(model_type, data_preprocessor, config)
        self._initialize_model()  # Call this here

    def _initialize_model(self):
        logger.info("Initializing the LSTNet model...")
        
        # Build the LSTNet model structure
        input_ = Input(shape=self.config['input_shape'])
        
        # CNN Layer
        x = Conv1D(filters=self.config.get('cnn_filters', 64), 
                   kernel_size=self.config.get('kernel_size', 3), 
                   activation='relu')(input_)
        
        # GRU Layer
        x = GRU(units=self.config.get('gru_units', 64), 
                return_sequences=True)(x)
        
        # Attention Layer
        query_value_attention_seq = Attention()([x, x])
        x = Add()([x, query_value_attention_seq])
        
        # Fully Connected Layer for Output
        x = Flatten()(x)
        output = Dense(1)(x)
        
        self.model = Model(inputs=input_, outputs=output)
        self.model.compile(optimizer=self.config.get('optimizer', 'adam'), loss=self.config.get('loss', 'mae'))
        
        logger.info("LSTNet model initialized and compiled successfully.")
        self.model.summary()

    def train_model(self, epochs=100, batch_size=32, early_stopping=True, patience=10, val_split=0.2):
        logger.info(f"Training the LSTNet model for {epochs} epochs with batch size of {batch_size}...")
        
        callbacks = []
        if early_stopping:
            es_callback = EarlyStopping(monitor='val_loss', patience=patience)
            callbacks.append(es_callback)
        
        self.history = self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=callbacks
        )
        logger.info("Training completed.")

class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        logger.info("Initializing MultiHeadSelfAttention with embed_size %d and heads %d.", embed_size, heads)
        
        self.values = Dense(self.head_dim, activation="linear")
        self.keys = Dense(self.head_dim, activation="linear")
        self.queries = Dense(self.head_dim, activation="linear")
        self.fc_out = Dense(embed_size, activation="linear")

    def call(self, values, keys, queries, mask):
        logger.info("Calling MultiHeadSelfAttention with values shape %s, keys shape %s, queries shape %s", str(values.shape), str(keys.shape), str(queries.shape))
        
        N = tf.shape(queries)[0]
        value_len, key_len, query_len = tf.shape(values)[1], tf.shape(keys)[1], tf.shape(queries)[1]

        values = tf.reshape(values, (N, value_len, self.heads, self.head_dim))
        keys = tf.reshape(keys, (N, key_len, self.heads, self.head_dim))
        queries = tf.reshape(queries, (N, query_len, self.heads, self.head_dim))

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        score = tf.einsum("nqhd,nkhd->nhqk", queries, keys)
        if mask is not None:
            score *= mask

        attention_weights = tf.nn.softmax(score / (self.embed_size ** (1 / 2)), axis=3)
        out = tf.einsum("nhql,nlhd->nqhd", attention_weights, values)
        out = tf.reshape(out, (N, query_len, self.heads * self.head_dim))
        out = self.fc_out(out)
        return out

def create_positional_encoding(max_seq_len, d_model):
    logger.info("Generating positional encoding for max_seq_len %d and d_model %d.", max_seq_len, d_model)

    pos = tf.range(max_seq_len, dtype=tf.float32)[:, tf.newaxis]
    div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
    sinusoidal_input = tf.matmul(pos, div_term[tf.newaxis, :])
    sines = tf.sin(sinusoidal_input)
    cosines = tf.cos(sinusoidal_input)
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    logger.info("Generated positional encoding with shape %s.", str(pos_encoding.shape))

    return pos_encoding
class TransformerBlock(Layer):

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        logger.info("Initializing TransformerBlock with embed_size %d, heads %d, dropout %f, forward_expansion %d.", embed_size, heads, dropout, forward_expansion)
        
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        
        self.feed_forward = tf.keras.Sequential([
            Dense(forward_expansion * embed_size, activation="relu"),
            Dense(embed_size),
        ])
        
        self.dropout = Dropout(dropout)

    def call(self, value, key, query, mask):
        logger.info("Calling TransformerBlock with value shape %s, key shape %s, query shape %s.", str(value.shape), str(key.shape), str(query.shape))
        
        attention = self.attention(value, key, query, mask)
        x = self.norm1(attention + query)
        x = self.dropout(x)
        
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        out = self.dropout(out)
        
        return out

class SOA_TRANSFORMER(BaseModel_DL_SOA):
    def __init__(self, model_type, data_preprocessor, config):
        super(SOA_TRANSFORMER, self).__init__(model_type, data_preprocessor, config)
        self._initialize_model()
    
    def _initialize_model(self):
        inputs = Input(shape=self.config['input_shape'])

        embed_size = self.config['embed_size']
        num_layers = self.config['num_layers']
        heads = self.config['heads']
        dropout = self.config['dropout']
        forward_expansion = self.config['forward_expansion']

        # Logging the initial configuration
        logger.info("Initializing Transformer with the following configuration:")
        logger.info("Embed size: %d", embed_size)
        logger.info("Number of layers: %d", num_layers)
        logger.info("Number of heads: %d", heads)
        logger.info("Dropout rate: %f", dropout)
        logger.info("Forward expansion: %d", forward_expansion)

        # Add an embedding layer
        x = Dense(embed_size)(inputs)
        logger.info("Added embedding layer with shape: %s", str(x.shape))

        positional_encoding = create_positional_encoding(self.config['input_shape'][0], embed_size)
        x += positional_encoding
        logger.info("Added positional encoding to the model")

        for i in range(num_layers):
            x = TransformerBlock(embed_size, heads, dropout, forward_expansion)(x, x, x, None)
            logger.info("Added Transformer block %d/%d", i+1, num_layers)

        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1, activation="linear")(x)
        logger.info("Added global average pooling and output layers")

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.config['optimizer'], loss=self.config['loss'])
        self.model.summary()
        logger.info("Model compiled with optimizer: %s and loss: %s", self.config['optimizer'], self.config['loss'])

    def train_model(self, epochs=100, batch_size=32, early_stopping=True, patience=10, val_split=0.2):
        logger.info("Training the Transformer model for %d epochs with batch size of %d...", epochs, batch_size)
        
        callbacks = []
        if early_stopping:
            es_callback = EarlyStopping(monitor='val_loss', patience=patience)
            callbacks.append(es_callback)
        
        self.history = self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=callbacks
        )
        logger.info("Training completed.")



from data_preprocessor import UnifiedDataPreprocessor




models = {
    'TCN': {
        'class': SOA_TCN,
        'config': {
            'input_shape': (data_preprocessor.X_train_seq.shape[1], data_preprocessor.X_train_seq.shape[2]),
            'sequence_length': data_preprocessor.X_train_seq.shape[1],
            'num_features': data_preprocessor.X_train_seq.shape[2],
            'nb_filters': 64,
            'kernel_size': 2,
            'nb_stacks': 1,
            'dilations': [1, 2, 4, 8, 16, 32],
            'padding': 'causal',
            'use_skip_connections': True,
            'dropout_rate': 0.2,
            'return_sequences': False,
            'activation': 'relu',
            'kernel_initializer': 'he_normal',
            'use_batch_norm': False,
            'optimizer': 'adam'
        },
        'skip': False
    },
    'NBEATS': {
        'class': SOA_NBEATS,
        'config': {
            'lookback': 1,  # This should be 10
            'horizon': 1,
            'num_generic_neurons': 512,
            'num_generic_stacks': 30,
            'num_generic_layers': 4,
            'num_trend_neurons': 256,
            'num_trend_stacks': 3,
            'num_trend_layers': 4,
            'num_seasonal_neurons': 2048,
            'num_seasonal_stacks': 3,
            'num_seasonal_layers': 4,
            'num_harmonics': 1,
            'polynomial_term': 3,
            'loss': 'mae',
            'learning_rate': 0.001,
            'batch_size': 1024
        },
        'skip': False
    },
    'WAVENET': {
        'class': SOA_WAVENET,
        'config': {
            'input_shape': (data_preprocessor.X_train_seq.shape[1], data_preprocessor.X_train_seq.shape[2]),
            'sequence_length': data_preprocessor.X_train_seq.shape[1],
            'num_features': data_preprocessor.X_train_seq.shape[2],
            'num_blocks': 4,
            'filters': 32,
            'kernel_size': 2,
            'optimizer': 'adam',
            'loss': 'mae'
        },
        'skip': False
    },
    'LSTNET': {
        'class': SOA_LSTNET,
        'config': {
            'input_shape': (data_preprocessor.X_train_seq.shape[1], data_preprocessor.X_train_seq.shape[2]),
            'sequence_length': data_preprocessor.X_train_seq.shape[1],
            'num_features': data_preprocessor.X_train_seq.shape[2],
            'cnn_filters': 64,
            'gru_units': 64,
            'kernel_size': 3,
            'optimizer': 'adam',
            'loss': 'mae'
        },
        'skip': False
    },
    'TRANSFORMER': {
        'class': SOA_TRANSFORMER,
        'config': {
            'input_shape': (data_preprocessor.X_train_seq.shape[1], data_preprocessor.X_train_seq.shape[2]),
            'sequence_length': data_preprocessor.X_train_seq.shape[1],
            'num_features': data_preprocessor.X_train_seq.shape[2],
            'num_layers': 2,
            'embed_size': 64,
            'heads': 4,
            'dropout': 0.2,
            'forward_expansion': 2,
            'optimizer': 'adam',
            'loss': 'mae'
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
        
        # Check if sequence_length and num_features exist in config
        sequence_length = config.get('sequence_length', None)  # Defaults to None if not found
        num_features = config.get('num_features', None)  # Defaults to None if not found

        # If they do exist, update the input shape
        if sequence_length is not None and num_features is not None:
            config['input_shape'] = (sequence_length, num_features)
        
        model = model_class(model_type=name, data_preprocessor=data_preprocessor, config=config)
        model.train_model(epochs=100, batch_size=32)
        model.make_predictions()
        evaluation_df = model.evaluate_model()
        print(f"{name} Model Evaluation:\n", evaluation_df)
        model.plot_history()
        model.plot_predictions()
        model.save_model_to_folder(version="final")



# Run all models
run_models(models, data_preprocessor)

# Run only specific models
#run_models(models, data_preprocessor, run_only=['NBEATS'])

# Skip specific models
#run_models(models, data_preprocessor, skip=['NBEATS'])
#print hello
