#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K
import keras_metrics as km
from keras import optimizers, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional, Input, Concatenate, BatchNormalization
from keras.layers import TimeDistributed, CuDNNLSTM
from keras.utils import Sequence, multi_gpu_model, plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback, CSVLogger
from livelossplot.keras import PlotLossesCallback
from scipy.stats import zscore
import random

## tuning
from tune_server import server_setup
from datetime import datetime

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)

data = pd.read_pickle("/tf/data_all.pkl").sort_values(['id','ICULOS'], ascending=True)

data.describe().to_csv('/tf/data_describe_setB.csv')

# need to record mean and std of zscore of each feature
data[list(data)[:35]] = data[list(data)[:35]].apply(zscore)
data[list(data)[38]] = zscore(data[list(data)[38]])

data_index = data["id"].unique().tolist()

train_index = random.sample(data_index, int(0.8*len(data_index)))
test_index = list(set(data_index) - set(train_index))
valid_index = random.sample(train_index, int(0.2*len(train_index)))
train_index = list(set(train_index) - set(valid_index))

train_data = data[data["id"].isin(train_index)]
test_data = data[data["id"].isin(test_index)]
valid_data = data[data["id"].isin(valid_index)]

np.random.shuffle(train_index)
np.random.shuffle(test_index)
np.random.shuffle(valid_index)


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, data, index):
        'Initialization'
        self.data = data #data of 42 columns
        self.index = index #list of id
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.index)
    
    def __getitem__(self, idx):
        'Generate one batch of data'
        i = self.index[idx]
        # vtial
        x1 = np.array(self.data.drop(columns=['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
                                              'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen',
                                              'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2','HospAdmTime','SepsisLabel','ICULOS']))
        x1 = x1[x1[:,-1]==i]
        x1 = x1[:,:-1]
        x1 = np.reshape(x1,(1,np.shape(x1)[0],np.shape(x1)[1]))
        
        
        # sofa
        x2 = np.array(self.data.drop(columns=['BaseExcess', 'HCO3', 'pH', 'SaO2', 'AST', 'BUN','Alkalinephos', 'Calcium', 'Chloride', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
                                              'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime','SepsisLabel','ICULOS']))
        x2 = x2[x2[:,-1]==i]
        x2 = x2[:,:-1]
        x2 = np.reshape(x2,(1,np.shape(x2)[0],np.shape(x2)[1]))
        
        
        # demographic
        x3 = np.array(self.data.drop(columns=['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
                                              'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total',
                                              'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets','SepsisLabel','ICULOS']))
        x3 = x3[x3[:,-1]==i]
        x3 = x3[:,:-1]
        x3 = np.reshape(x3,(1,np.shape(x3)[0],np.shape(x3)[1]))

        y = np.array(self.data[["SepsisLabel","id"]])
        y = y[y[:,-1]==i]
        y = y[:,:-1]
        y = np.reshape(y,(1,np.shape(y)[0],np.shape(y)[1]))
        return [x1, x2, x3], [y]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.index)

def main(hparams):
    train_data_generator = DataGenerator(train_data, train_index)
    valid_data_generator = DataGenerator(valid_data, valid_index)
    test_data_generator = DataGenerator(test_data, test_index)

    input_vital = Input(shape = (None,8), name='vital_input')
    input_sofa = Input(shape = (None,13), name='sofa_input')
    input_demographic = Input(shape = (None,5), name='demographic_input')

    lstm_vital_1 = Bidirectional(CuDNNLSTM(8, return_sequences=True, name='lstm_vital_1'))(input_vital)
    lstm_vital_2 = Bidirectional(CuDNNLSTM(8, return_sequences=True, name='lstm_vital_2'))(lstm_vital_1)
    bn_vital = BatchNormalization()(lstm_vital_2)
    Dense_vital_1 = Dense(units=20, name = 'Dense_vital_1', activation='relu')(bn_vital)
    drop_vital = Dropout(0.2)(Dense_vital_1)

    lstm_sofa_1 = Bidirectional(CuDNNLSTM(13, return_sequences=True, name='sofa_lstm_1'))(input_sofa)
    lstm_sofa_2 = Bidirectional(CuDNNLSTM(13, return_sequences=True, name='sofa_lstm_2'))(lstm_sofa_1)
    lstm_sofa_3 = Bidirectional(CuDNNLSTM(13, return_sequences=True, name='sofa_lstm_3'))(lstm_sofa_2)
    lstm_sofa_4 = Bidirectional(CuDNNLSTM(13, return_sequences=True, name='sofa_lstm_4'))(lstm_sofa_3)
    bn_sofa = BatchNormalization()(lstm_sofa_4)
    Dense_sofa_1 = Dense(units=40, name = 'Dense_sofa_2', activation='relu')(bn_sofa)
    drop_sofa = Dropout(0.2)(Dense_sofa_1)

    dense_demographic_1 = Dense(units=20, name = 'demographic_dense_1', activation='relu')(input_demographic)
    dense_demographic_2 = Dense(units=10, name = 'demographic_dense_2', activation='relu')(dense_demographic_1)
    dense_demographic_3 = Dense(units=10, name = 'demographic_dense_3', activation='relu')(dense_demographic_2)
    bn_demographic = BatchNormalization()(dense_demographic_3)
    drop_demographic = Dropout(0.2)(bn_demographic)


    merge = Concatenate()([drop_vital,drop_sofa,drop_demographic])

    output_1 = Dense(units=20, name = 'output_dense_1', activation='relu')(merge)
    bn_out = BatchNormalization()(output_1)
    drop_1 = Dropout(0.2)(bn_out)
    output_2 = Dense(units=1, name = 'output_dense_2', activation='sigmoid')(drop_1)

    model = Model(inputs = [input_vital, input_sofa, input_demographic], outputs = output_2)


    plot_model(model, show_shapes=True, show_layer_names=True, to_file='/tf/model/model_mulit-task-addLayer_setB_v2-1.png')

    def focal_loss(y_true, y_pred):
        gamma = hparams["forcal_gamma"]
        alpha = hparams["forcal_alpha"]
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

    # def auc(y_true, y_pred):
    #     auc = tf.metrics.auc(y_true, y_pred)[1]
    #     K.get_session().run(tf.local_variables_initializer())
    #     return auc

    # def mean_pred(y_true, y_pred):
    #     return K.mean(y_pred)

    # def mean_true(y_true, y_pred):
    #     d = y_true - 1
    #     return K.mean(y_true)

    # def precision(y_true, y_pred):
    #     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    #     precision = true_positives / (predicted_positives + K.epsilon())
    #     return precision

    # def recall(y_true, y_pred):
    #     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    #     recall = true_positives / (possible_positives + K.epsilon())
    #     return recall

    # def fbeta_score(y_true, y_pred, beta=3):
    #     if beta < 0:
    #         raise ValueError('The lowest choosable beta is zero (only precision).')
            
    #     # If there are no true positives, fix the F score at 0 like sklearn.
    #     if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
    #         return 0

    #     p = precision(y_true, y_pred)
    #     r = recall(y_true, y_pred)
    #     bb = beta ** 2
    #     fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    #     return fbeta_score

    adam = optimizers.Adam(lr=hparams["adam_lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=focal_loss, optimizer=adam, metrics=[km.f1_score(),
                                                            km.precision(),
                                                            km.recall(),
                                                            km.true_positive(),
                                                            km.false_negative(),
                                                            km.false_positive(),
                                                            km.true_negative()])


    # reduceLR = ReduceLROnPlateau(monitor='val_fbeta_score', factor=0.6, patience=5, verbose=1, mode='max', cooldown=3, min_lr=1e-8)
    checkpointer = ModelCheckpoint('/tf/temp_model/model_sample_multi-task_addLayer_setB_v2-1_{epoch:03d}.hdf5', verbose=1)
    csv_logger = CSVLogger('/tf/temp_model/model_log_multi-task_addLayer_setB_v2-1.log')


    train_history = model.fit_generator(generator = train_data_generator,
                                        validation_data = valid_data_generator,
                                        epochs = hparams["run_epochs"],
                                        steps_per_epoch = train_data_generator.__len__(),
                                        validation_steps = valid_data_generator.__len__(),
                                        max_queue_size = 48,
                                        workers = 48,
                                        callbacks = [checkpointer, csv_logger, PlotLossesCallback(fig_path='/tf/temp_model/v2.png')],
                                        use_multiprocessing = True,
                                        verbose = 2)


    evaluate = model.evaluate_generator(generator=test_data_generator , steps=test_data_generator.__len__(),
                                        max_queue_size=48, workers=48, use_multiprocessing=True, verbose=2)
    print(evaluate)



    # save model

    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    model_save_path = '/tf/model/model_multi-task_addLayer_setB_v2-1-{}.h5'.format(current_time)
    model_save_weights_path = '/tf/model/model_weights_multi-task_addLayer_setB_v2-1-{}.h5'.format(current_time)
    model_log_save_path = '/tf/model/model_log_multi-task_addLayer_setB_v2-1-{}.csv'.format(current_time)

    model.save(model_save_path)
    model.save_weights(model_save_weights_path)
    pd.DataFrame(model.history.history).to_csv(model_log_save_path, index=False)
    print('Finish')
    return evaluate[1]

if __name__=="__main__":
    domain_socket = "/tmp/tuneconn"
    conn_authkey = b'physionet'
    server_setup(domain_socket, conn_authkey, main)
