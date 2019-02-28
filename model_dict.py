import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution1D, Lambda, \
    Convolution2D,Convolution3D, Flatten, \
    Reshape, LSTM, Dropout, TimeDistributed, BatchNormalization,AveragePooling2D
from keras.regularizers import l2
from keras.optimizers import Adam

from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import time

class TimeHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def get_prediction_model(input_shape, output_shape, model_config_df, output_act, loss_tmp, metrics):

    if 'LSTM' == model_config_df['model'].values[0]:  # 'LSTM':
        type_1_arr = model_config_df[0][0]
        type_2_arr = model_config_df[1][0]
        model_tmp = generate_LSTM_model(input_shape, output_shape, type_1_arr, type_2_arr, output_act, loss_tmp,metrics=['acc'])

    if 'CNN' == model_config_df['model'].values[0]:  # 'CNN':
        type_1_arr = model_config_df[0][0]
        type_2_arr = model_config_df[1][0]
        model_tmp = generate_CNN_model(input_shape, output_shape, type_1_arr, type_2_arr, output_act, loss_tmp,
                                       metrics=['acc'])

    if 'CLSTM' == model_config_df['model'].values[0]:  # 'CLSTM':
        type_1_arr = model_config_df[0][0]
        type_2_arr = model_config_df[1][0]
        type_3_arr = model_config_df[2][0]
        model_tmp = generate_CLSTM_model(input_shape, output_shape, type_1_arr, type_2_arr, type_3_arr, output_act,
                                         loss_tmp, metrics=['acc'])

    return model_tmp


def get_explored_prediction_model(input_shape, output_shape, model_config_df, output_act, loss_tmp, metrics=['loss']):
    config_re = re.compile(r'\d+(?:,\d+)?')

    if 'LSTM' == model_config_df['model'].values[0]:  # 'LSTM':

        config_tmp = model_config_df[0].values[0]
        config_tmp = np.array(config_re.findall(config_tmp)).astype(int)
        type_1_arr = config_tmp
        config_tmp = model_config_df[1].values[0]
        config_tmp = np.array(config_re.findall(config_tmp)).astype(int)
        type_2_arr = config_tmp
        model_tmp = generate_LSTM_model(input_shape, output_shape, type_1_arr, type_2_arr, output_act, loss_tmp,
                                        metrics=['acc'])

    if 'CNN' == model_config_df['model'].values[0]:  # 'CNN':
        config_tmp = model_config_df[0].values[0]
        config_tmp = np.array(config_re.findall(config_tmp)).astype(int)
        type_1_arr = config_tmp
        config_tmp = model_config_df[1].values[0]
        config_tmp = np.array(config_re.findall(config_tmp)).astype(int)
        type_2_arr = config_tmp
        model_tmp = generate_CNN_model(input_shape, output_shape, type_1_arr, type_2_arr, output_act, loss_tmp,
                                       metrics=['acc'])

    if 'CLSTM' == model_config_df['model'].values[0]:  # 'CLSTM':
        config_tmp = model_config_df[0].values[0]
        config_tmp = np.array(config_re.findall(config_tmp)).astype(int)
        type_1_arr = config_tmp
        config_tmp = model_config_df[1].values[0]
        config_tmp = np.array(config_re.findall(config_tmp)).astype(int)
        type_2_arr = config_tmp
        config_tmp = model_config_df[2].values[0]
        config_tmp = np.array(config_re.findall(config_tmp)).astype(int)
        type_3_arr = config_tmp
        model_tmp = generate_CLSTM_model(input_shape, output_shape, type_1_arr, type_2_arr, type_3_arr, output_act,
                                         loss_tmp, metrics=['acc'])

    return model_tmp


def generate_LSTM_model(input_shape, output_shape, type_1_arr, type_2_arr, output_act, loss_tmp, metrics):
    lstm_layers = type_1_arr
    dense_layers = type_2_arr

    model = Sequential()
    if len(lstm_layers) == 1:
        model.add(LSTM(units=lstm_layers[0], return_sequences=False, input_shape=input_shape))
    else:
        for layer_num in range(len(lstm_layers)):
            lstm_cell = lstm_layers[layer_num]
            if layer_num == 0:
                model.add(LSTM(units=lstm_cell, return_sequences=True, input_shape=input_shape))
            if layer_num != 0 & (layer_num + 1) != len(lstm_layers):
                model.add(LSTM(units=lstm_cell, return_sequences=True))
            if (layer_num + 1) == len(lstm_layers):
                model.add(LSTM(lstm_cell, return_sequences=False))

    for dense in dense_layers:
        model.add(Dense(dense, activation=output_act))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape))
    model.compile(loss=loss_tmp, optimizer='adam', metrics=metrics)

    return model


# without time distributed dynamic
def generate_CLSTM_model(input_shape, output_shape, type_1_arr, type_2_arr, type_3_arr, output_act, loss_tmp,
                         metrics=['loss']):
    filters = type_1_arr
    lstm_layers = type_2_arr
    dense_layers = type_3_arr

    model = Sequential()

    if len(filters) == 1:
        model.add(Conv1D(filters=filters[0], kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    else:
        for fil_num in range(len(filters)):
            fil = filters[fil_num]
            if fil_num == 0:
                model.add(Conv1D(filters=filters[0], kernel_size=3, padding='same', activation='relu',
                                 input_shape=input_shape))
            else:
                model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(output_shape))
    if len(lstm_layers) == 1:
        model.add(LSTM(units=lstm_layers[0], return_sequences=False))
    else:
        for layer_num in range(len(lstm_layers)):
            lstm_cell = lstm_layers[layer_num]
            if layer_num == 0:
                model.add(LSTM(units=lstm_cell, return_sequences=True))
            if layer_num != 0 & (layer_num + 1) != len(lstm_layers):
                model.add(LSTM(units=lstm_cell, return_sequences=True))
            if (layer_num + 1) == len(lstm_layers):
                model.add(LSTM(lstm_cell, return_sequences=False))

    for dense in dense_layers:
        model.add(Dense(dense, activation=output_act))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape))
    model.compile(loss=loss_tmp, optimizer='adam', metrics=metrics)
    return model


# without time distributed
def generate_CNN_model(input_shape, output_shape, type_1_arr, type_2_arr, output_act, loss_tmp,
                       metrics=['loss']):
    filters = type_1_arr
    dense_layers = type_2_arr

    model = Sequential()

    if len(filters) == 1:
        model.add(Conv1D(filters=filters[0], kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    else:
        for fil_num in range(len(filters)):
            fil = filters[fil_num]
            if fil_num == 0:
                model.add(Conv1D(filters=filters[0], kernel_size=3, padding='same', activation='relu',
                                 input_shape=input_shape))
            else:
                model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())

    for dense in dense_layers:
        model.add(Dense(dense, activation=output_act))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='relu'))
    model.compile(loss=loss_tmp, optimizer='adam', metrics=metrics)
    return model

