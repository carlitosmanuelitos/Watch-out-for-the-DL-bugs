from kerastuner import HyperModel
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
from data_preprocessor import UnifiedDataPreprocessor
from data_fetcher import btc_data
import pandas as pd


# Other settings
from IPython.display import display, HTML
import os, warnings, logging



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
print("LSTM Sequence-to-One Data Shapes:")
print("X_train_seq:", X_train_seq.shape,"y_train_seq:", y_train_seq.shape, "X_test_seq:", X_test_seq.shape, "y_test_seq:", y_test_seq.shape)
print("----")

class EnhancedRNNHyperModel(HyperModel):
    def __init__(self, data_preprocessor, model_type):  # data_preprocessor passed as argument
        self.input_shape = data_preprocessor.X_train_seq.shape[1:]
        self.X_train = data_preprocessor.X_train_seq
        self.y_train = data_preprocessor.y_train_seq
        self.X_test = data_preprocessor.X_test_seq
        self.y_test = data_preprocessor.y_test_seq
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        # Debugging Information
        print(f"Initializing EnhancedRNNHyperModel with model_type={self.model_type}")
        print(f"Input shape: {self.input_shape}")
        print(f"Training data shape: X={self.X_train.shape}, y={self.y_train.shape}")
        print(f"Test data shape: X={self.X_test.shape}, y={self.y_test.shape}")

    def build(self, hp):
        self.logger.debug(f"Building model with hyperparameters: {hp.values}")
        model = Sequential()

        if self.model_type == 'LSTM':
            num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=4)
            for i in range(num_lstm_layers):
                model.add(LSTM(
                    units=hp.Int(f'lstm_units_{i}', min_value=32, max_value=256, step=32),
                    activation=hp.Choice(f'lstm_activation_{i}', values=['tanh', 'sigmoid', 'relu']),
                    recurrent_activation=hp.Choice(f'lstm_recurrent_activation_{i}', values=['sigmoid', 'tanh', 'relu']),
                    recurrent_dropout=hp.Float(f'recurrent_dropout_{i}', min_value=0.0, max_value=0.5, step=0.05),
                    return_sequences=True if i < num_lstm_layers - 1 else hp.Boolean('return_sequences')
                ))
                model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.05)))

        elif self.model_type == 'BiLSTM':
            num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=4)
            for i in range(num_lstm_layers):
                model.add(Bidirectional(LSTM(
                    units=hp.Int(f'lstm_units_{i}', min_value=32, max_value=256, step=32),
                    activation=hp.Choice(f'lstm_activation_{i}', values=['tanh', 'sigmoid', 'relu']),
                    recurrent_activation=hp.Choice(f'lstm_recurrent_activation_{i}', values=['sigmoid', 'tanh', 'relu']),
                    recurrent_dropout=hp.Float(f'recurrent_dropout_{i}', min_value=0.0, max_value=0.5, step=0.05),
                    return_sequences=True if i < num_lstm_layers - 1 else hp.Boolean('return_sequences')
                )))
                model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.05)))

        elif self.model_type == 'GRU':
            num_gru_layers = hp.Int('num_gru_layers', min_value=1, max_value=4)
            for i in range(num_gru_layers):
                model.add(GRU(
                    units=hp.Int(f'gru_units_{i}', min_value=32, max_value=256, step=32),
                    activation=hp.Choice(f'gru_activation_{i}', values=['tanh', 'sigmoid', 'relu']),
                    recurrent_activation=hp.Choice(f'gru_recurrent_activation_{i}', values=['sigmoid', 'tanh', 'relu']),
                    recurrent_dropout=hp.Float(f'gru_recurrent_dropout_{i}', min_value=0.0, max_value=0.5, step=0.05),
                    return_sequences=True if i < num_gru_layers - 1 else hp.Boolean('gru_return_sequences')
                ))
                model.add(Dropout(hp.Float(f'gru_dropout_{i}', min_value=0.0, max_value=0.5, step=0.05)))

        elif self.model_type == 'BiGRU':
            num_gru_layers = hp.Int('num_gru_layers', min_value=1, max_value=4)
            for i in range(num_gru_layers):
                model.add(Bidirectional(GRU(
                    units=hp.Int(f'gru_units_{i}', min_value=32, max_value=256, step=32),
                    activation=hp.Choice(f'gru_activation_{i}', values=['tanh', 'sigmoid', 'relu']),
                    recurrent_activation=hp.Choice(f'gru_recurrent_activation_{i}', values=['sigmoid', 'tanh', 'relu']),
                    recurrent_dropout=hp.Float(f'gru_recurrent_dropout_{i}', min_value=0.0, max_value=0.5, step=0.05),
                    return_sequences=True if i < num_gru_layers - 1 else hp.Boolean('gru_return_sequences')
                )))
                model.add(Dropout(hp.Float(f'gru_dropout_{i}', min_value=0.0, max_value=0.5, step=0.05)))

        elif self.model_type == 'SimpleRNN':
            num_rnn_layers = hp.Int('num_rnn_layers', min_value=1, max_value=4)
            for i in range(num_rnn_layers):
                model.add(SimpleRNN(
                    units=hp.Int(f'rnn_units_{i}', min_value=32, max_value=256, step=32),
                    activation=hp.Choice(f'rnn_activation_{i}', values=['tanh', 'sigmoid', 'relu']),
                    recurrent_dropout=hp.Float(f'rnn_recurrent_dropout_{i}', min_value=0.0, max_value=0.5, step=0.05),
                    return_sequences=True if i < num_rnn_layers - 1 else hp.Boolean('rnn_return_sequences')
                ))
                model.add(Dropout(hp.Float(f'rnn_dropout_{i}', min_value=0.0, max_value=0.5, step=0.05)))

        elif self.model_type == 'StackedRNN':
            num_stacked_layers = hp.Int('num_stacked_layers', min_value=1, max_value=4)
            for i in range(num_stacked_layers):
                rnn_type = hp.Choice(f'rnn_type_{i}', values=['SimpleRNN', 'LSTM', 'GRU'])
                
                if rnn_type == 'SimpleRNN':
                    model.add(SimpleRNN(
                        units=hp.Int(f'stacked_rnn_units_{i}', min_value=32, max_value=256, step=32),
                        activation=hp.Choice(f'stacked_rnn_activation_{i}', values=['tanh', 'sigmoid', 'relu']),
                        recurrent_dropout=hp.Float(f'stacked_rnn_recurrent_dropout_{i}', min_value=0.0, max_value=0.5, step=0.05),
                        return_sequences=True if i < num_stacked_layers - 1 else hp.Boolean('stacked_rnn_return_sequences')
                    ))
                elif rnn_type == 'LSTM':
                    model.add(LSTM(
                        units=hp.Int(f'stacked_lstm_units_{i}', min_value=32, max_value=256, step=32),
                        activation=hp.Choice(f'stacked_lstm_activation_{i}', values=['tanh', 'sigmoid', 'relu']),
                        recurrent_dropout=hp.Float(f'stacked_lstm_recurrent_dropout_{i}', min_value=0.0, max_value=0.5, step=0.05),
                        return_sequences=True if i < num_stacked_layers - 1 else hp.Boolean('stacked_rnn_return_sequences')
                    ))
                elif rnn_type == 'GRU':
                    model.add(GRU(
                        units=hp.Int(f'stacked_gru_units_{i}', min_value=32, max_value=256, step=32),
                        activation=hp.Choice(f'stacked_gru_activation_{i}', values=['tanh', 'sigmoid', 'relu']),
                        recurrent_dropout=hp.Float(f'stacked_gru_recurrent_dropout_{i}', min_value=0.0, max_value=0.5, step=0.05),
                        return_sequences=True if i < num_stacked_layers - 1 else hp.Boolean('stacked_rnn_return_sequences')
                    ))
                model.add(Dropout(hp.Float(f'stacked_rnn_dropout_{i}', min_value=0.0, max_value=0.5, step=0.05)))
                    
        elif self.model_type == 'AttentionLSTM':
            num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=4)
            for i in range(num_lstm_layers):
                model.add(LSTM(
                    units=hp.Int(f'lstm_units_{i}', min_value=32, max_value=256, step=32),
                    activation=hp.Choice(f'lstm_activation_{i}', values=['tanh', 'sigmoid', 'relu']),
                    recurrent_dropout=hp.Float(f'recurrent_dropout_{i}', min_value=0.0, max_value=0.5, step=0.05),
                    return_sequences=True  # For Attention, the last LSTM layer should also return sequences
                ))
                model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.05)))
            
            # Adding attention layer
            model.add(Attention(use_scale=hp.Bool('attention_use_scale')))
            model.add(GlobalAveragePooling1D())

        elif self.model_type == 'CNNLSTM':
            num_conv_layers = hp.Int('num_conv_layers', min_value=1, max_value=4)
            for i in range(num_conv_layers):
                model.add(Conv1D(
                    filters=hp.Int(f'conv_filters_{i}', min_value=32, max_value=256, step=32),
                    kernel_size=hp.Int(f'conv_kernel_size_{i}', min_value=1, max_value=5),
                    activation=hp.Choice(f'conv_activation_{i}', values=['relu', 'tanh', 'sigmoid'])
                ))
                model.add(MaxPooling1D(pool_size=hp.Int(f'pool_size_{i}', min_value=2, max_value=4)))
            
            model.add(GlobalMaxPooling1D())
            model.add(Reshape((1, hp.Int('reshape_size', min_value=32, max_value=256, step=32))))
            
            num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=4)
            for i in range(num_lstm_layers):
                model.add(LSTM(
                    units=hp.Int(f'lstm_units_{i}', min_value=32, max_value=256, step=32),
                    activation=hp.Choice(f'lstm_activation_{i}', values=['tanh', 'sigmoid', 'relu']),
                    recurrent_dropout=hp.Float(f'recurrent_dropout_{i}', min_value=0.0, max_value=0.5, step=0.05),
                    return_sequences=True if i < num_lstm_layers - 1 else hp.Boolean('return_sequences')
                ))
                model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.05)))


        model.add(Dense(
            hp.Int('dense_units', min_value=1, max_value=3),
            activation=hp.Choice('dense_activation', values=['relu', 'linear', 'sigmoid', 'tanh'])
        ))

        model.compile(
            optimizer=hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'nadam', 'ftrl']),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        return model


best_models = {}
#model_types = ['LSTM', 'BiLSTM', 'GRU', 'BiGRU', 'SimpleRNN', 'StackedRNN', 'AttentionLSTM', 'CNNLSTM']
model_types = ['LSTM','BiLSTM','GRU']
for model_type in model_types:
    print(f"Optimizing {model_type}...")
    
    tuner = BayesianOptimization(
        hypermodel=EnhancedRNNHyperModel(data_preprocessor, model_type),  # Make sure class name is consistent
        objective='val_loss',
        max_trials=50,
        directory='bayesian_optimization',
        project_name=f'{model_type}',
        overwrite=True
    )

    # Define your callbacks here
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min')

    tuner.search(data_preprocessor.X_train_seq, data_preprocessor.y_train_seq, epochs=20, validation_split=0.2, callbacks=[early_stopping_callback,lr_schedule])

    # Get the best model for this type
    best_model = tuner.get_best_models(num_models=1)[0]
    best_models[model_type] = best_model  # Save the best model for each type



