import sys
import numpy as np

import sys

from keras.layers.convolutional import *
from keras import initializers
import random
import tensorflow as tf
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import pickle
from keras import backend
from keras.models import Model
from keras import optimizers
from keras.layers import *
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Activation
from keras_contrib.layers import CRF
from keras_contrib.metrics.crf_accuracies import *
from keras_contrib.losses.crf_losses import *
from keras.layers.advanced_activations import LeakyReLU, PReLU

from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import KFold
from utils import *
import keras
K = keras.backend
# def get_sepsis_score(values, column_names):
#     x_mean = np.array([
#         83.8996, 97.0520,  36.8055,  126.2240, 86.2907,
#         66.2070, 18.7280,  33.7373,  -3.1923,  22.5352,
#         0.4597,  7.3889,   39.5049,  96.8883,  103.4265,
#         22.4952, 87.5214,  7.7210,   106.1982, 1.5961,
#         0.6943,  131.5327, 2.0262,   2.0509,   3.5130,
#         4.0541,  1.3423,   5.2734,   32.1134,  10.5383,
#         38.9974, 10.5585,  286.5404, 198.6777])
#     x_std = np.array([
#         17.6494, 3.0163,  0.6895,   24.2988, 16.6459,
#         14.0771, 4.7035,  11.0158,  3.7845,  3.1567,
#         6.2684,  0.0710,  9.1087,   3.3971,  430.3638,
#         19.0690, 81.7152, 2.3992,   4.9761,  2.0648,
#         1.9926,  45.4816, 1.6008,   0.3793,  1.3092,
#         0.5844,  2.5511,  20.4142,  6.4362,  2.2302,
#         29.8928, 7.0606,  137.3886, 96.8997])
#     c_mean = np.array([60.8711, 0.5435, 0.0615, 0.0727, -59.6769, 28.4551])
#     c_std = np.array([16.1887, 0.4981, 0.7968, 0.8029, 160.8846, 29.5367])

#     x = values[:, 0:34]
#     c = values[:, 34:40]
#     x_norm = np.nan_to_num((x - x_mean) / x_std)
#     c_norm = np.nan_to_num((c - c_mean) / c_std)

#     beta = np.array([
#         0.1806,  0.0249, 0.2120,  -0.0495, 0.0084,
#         -0.0980, 0.0774, -0.0350, -0.0948, 0.1169,
#         0.7476,  0.0323, 0.0305,  -0.0251, 0.0330,
#         0.1424,  0.0324, -0.1450, -0.0594, 0.0085,
#         -0.0501, 0.0265, 0.0794,  -0.0107, 0.0225,
#         0.0040,  0.0799, -0.0287, 0.0531,  -0.0728,
#         0.0243,  0.1017, 0.0662,  -0.0074, 0.0281,
#         0.0078,  0.0593, -0.2046, -0.0167, 0.1239])
#     rho = 7.8521
#     nu = 1.0389

#     xstar = np.concatenate((x_norm, c_norm), axis=1)
#     exp_bx = np.exp(np.matmul(xstar, beta))
#     l_exp_bx = pow(4 / rho, nu) * exp_bx

#     scores = 1 - np.exp(-l_exp_bx)
#     labels = (scores > 0.45)
#     return (scores, labels)

def compute_sepsis_score(input_file):
    K.clear_session()
    num_features = 40
    input_x = Input(shape=(None, num_features))
    x = input_x

    n_feature_maps = num_features
    # print('build conv_x')
    # conv_x = keras.layers.normalization.BatchNormalization()(x)
    # conv_x = Activation('relu')(conv_x)
    # conv_x = Dropout(0.2)(conv_x)
    # conv_x = Conv1D(n_feature_maps, 25, strides=1, padding="same")(conv_x)
    # # conv_x = MaxPooling1D(padding='same')(conv_x)
    # conv_x = keras.layers.normalization.BatchNormalization()(conv_x)

    # # shortcut_y = MaxPooling1D(padding='same')(x)
    # shortcut_y = x
    # print('Merging skip connection')
    # y = add([shortcut_y, conv_x])

    # print("Y output shaspe ", y.get_shape())
    y = x
    for k in range(3):
        # print("k = ", k)
        # print('build conv_x')
        x1 = y
        conv_x = keras.layers.normalization.BatchNormalization()(x1)
        conv_x = Activation('relu')(conv_x)
        conv_x = Dropout(0.5)(conv_x)
        conv_x = Conv1D(n_feature_maps, 12, strides=1, padding="same")(conv_x)
        # if k == 0:
        #     conv_x = MaxPooling1D(padding='same')(conv_x)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)
        # conv_x = Conv1D(n_feature_maps, 6, strides=1, padding="same")(conv_x)

        # conv_x = Conv1D(n_feature_maps, 6, strides=1, padding="same")(conv_x)
        # # if k == 0:
        # #     conv_x = MaxPooling1D(padding='same')(conv_x)
        # conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        # conv_x = Activation('relu')(conv_x)

        conv_x = Bidirectional(LSTM(
            units=n_feature_maps/2, return_sequences=True, recurrent_dropout=0.5))(conv_x)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        # conv_x = Bidirectional(LSTM(units=n_feature_maps/2, return_sequences=True, recurrent_dropout=0.5) )(conv_x)
        # conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        # conv_x = Activation('relu')(conv_x)
        # shortcut_y = MaxPooling1D(padding='same')(x1)
        shortcut_y = x1
        # print('Merging skip connection')
        y = add([shortcut_y, conv_x])

    # x = Bidirectional(LSTM(units=8, return_sequences=True, activation='tanh',
    #                                 recurrent_dropout=0.5) )(y)  # variational biLSTM
    # x = Bidirectional(LSTM(units=32, return_sequences=True, activation='tanh',
    #                                 recurrent_dropout=0.5) )(x)
    # y = x
    # for i in range()
    # x = BatchNormalization()(x)

    # for i in range(5):
    #     x1 = x
    #     # x1 = BatchNormalization()(x1)
    #     x1 = Bidirectional(LSTM(units=20, return_sequences=True,
    #                                 recurrent_dropout=0.5) )(x1)
    #     x = add([x1, x])
        # x = BatchNormalization()(x)

    # x =
    # # x = TimeDistributed(Dense(128,activation='softmax'))(x)
    # x = TimeDistributed(Dense(20,activation='relu',kernel_regularizer=regularizers.l2(0.01)))(x)
    # x = TimeDistributed(Dense(20,activation='relu',kernel_regularizer=regularizers.l2(0.01)))(x)
    # x = TimeDistributed(Dense(100,kernel_initializer=keras.initializers.he_uniform(seed=None),
    #     activation='relu',kernel_regularizer=regularizers.l2(0.001))) (x)
    # x = Dropout(0.1)(x)
    # x = Dense(40,kernel_initializer='random_uniform',
    #     activation='relu', kernel_regularizer=regularizers.l2(0.001))(y)

    # x = Dense(20,kernel_initializer='random_uniform',
    #     activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = y

    # x = TimeDistributed(Dense(100,kernel_initializer='random_uniform',
    #     activation='relu', kernel_regularizer=regularizers.l2(0.001)))(x)

    # x = TimeDistributed(Dense(80,kernel_initializer='random_uniform',
    #     activation='relu'
    #     # , kernel_regularizer=regularizers.l1(0.001)

    #     ))(x)

    # # x = TimeDistributed(Dense(40,kernel_initializer='random_uniform',
    # #     activation='relu'
    # #     , kernel_regularizer=regularizers.l2(0.01)

    # #     ))(x)

    # x = Dropout(0.8)(x)
    # # x = TimeDistributed(Dense(40,kernel_initializer=keras.initializers.he_uniform(seed=None),
    # #     activation='relu', kernel_regularizer=regularizers.l2(0.001)))(x)
    # # x = Dropout(0.1)(x)
    # x = TimeDistributed(Dense(20,kernel_initializer='random_uniform',
    #     activation='relu'
    #     # , kernel_regularizer=regularizers.l1(0.001)
    #     ))(x)
    crf = CRF(2, sparse_target=False, learn_mode='marginal')  # CRF layer
    out = crf(x)  # output
    # out = Dense(1, activation='sigmoid')(x)
    # model = Model(input_x, out)

    # x = y

    # x = Dropout(0.2)(x)

    # out = TimeDistributed(Dense(2, kernel_initializer='random_uniform',
    #                             activation='softmax'
    #                             ))(x)
    model = Model(input_x, out)

    adam = optimizers.Adam(lr=0.001, epsilon=1e-5,
                           clipvalue=0.8, amsgrad=True)
    model.compile(optimizer=adam, loss=crf_loss, metrics=[], sample_weight_mode = 'temporal')


    model.load_weights(filepath='best_model_lstm_dense.ckpt')


    #read challenge data
    data = get_data_from_file(input_file)
    X_test, _ = prepare_input_for_lstm_crf([data], is_training=False)

    #normalize test data
    min_data = np.load('min_data.txt.npy')
    max_data = np.load('max_data.txt.npy')

    for idx, t_sequence in enumerate(X_test):
        X_test[idx] = (t_sequence - min_data) / \
            (max_data - min_data + 1e-8)
    scores = model.predict(np.array(X_test[0]).reshape((1, len(X_test[0]), 40)))[0]
    return scores    




def read_challenge_data(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        values = np.loadtxt(f, delimiter='|')
    # ignore SepsisLabel column if present
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        values = values[:, :-1]
    return (values, column_names)


if __name__ == '__main__':
    threshold = 0.8
    if len(sys.argv) != 3:
        sys.exit('Usage: %s input[.psv] output[.out]' % sys.argv[0])

    record_name = sys.argv[1]
    if record_name.endswith('.psv'):
        record_name = record_name[:-4]

    # read input data
    input_file = record_name + '.psv'

    scores = compute_sepsis_score(input_file)


    # (values, column_names) = read_challenge_data(input_file)

    # # generate predictions
    # (scores, labels) = get_sepsis_score(values, column_names)

    # write predictions to output file
    output_file = sys.argv[2]
    output_str = 'PredictedProbability|PredictedLabel\n'
    for p in scores:
        output_str += str(p[1])+ '|'
        if p[1] > threshold:
            output_str += '1\n'
        else:
            output_str += '0\n'
    with open(output_file, 'w') as f:
        f.write(output_str)
