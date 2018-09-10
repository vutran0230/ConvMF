'''
Created on Dec 8, 2015

@author: donghyun
'''
import numpy as np

np.random.seed(1337)

from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Dense, Conv1D, GlobalMaxPool1D, Concatenate, LSTM
from keras.preprocessing import sequence


class CNN_module():
    '''
    classdocs
    '''
    batch_size = 128
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 5

    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, nb_filters, init_W=None):
        self.max_len = max_len
        max_features = vocab_size
        vanila_dimension = 200
        projection_dimension = output_dimesion

        filter_lengths = [3, 4, 5]
        x = Input(shape=(max_len,), dtype='int32',name='input')

        '''Embedding Layer'''
        x_emb = Embedding(input_dim=max_features,
                              output_dim=emb_dim,
                              input_length=max_len,
                              weights=[init_W / 20] if init_W is not None else None,
                              name='sentence_embeddings')(x)

        '''Convolution Layer & Max Pooling Layer'''
        y = [GlobalMaxPool1D()(Conv1D(nb_filters, filter_length, activation="relu")(x_emb))
            for filter_length in filter_lengths]
        
        if len(y) > 1:
            y = Concatenate()(y)
        else:
            y = y[0]

        #''' or using LSTM & Max Pooling Layer'''
        #y = GlobalMaxPool1D()(LSTM(nb_filters,return_sequences=True)(x_emb))

        
        #''' or using LSTM with final state'''
        #y = LSTM(nb_filters)(x_emb)

        '''Dropout Layer'''
        y = Dense(vanila_dimension, activation='tanh', name='fully_connect')(y)
        y = Dropout(dropout_rate, name='dropout')(y)
        '''Projection Layer & Output Layer'''
        y = Dense(projection_dimension, activation='tanh', name='output')(y)

        # Output Layer
        self.model = Model(x, y)
        self.model.compile(optimizer='rmsprop', loss='mse')

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    def train(self, X_train, V, item_weight, seed):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        np.random.seed(seed)
        X_train = np.random.permutation(X_train)
        np.random.seed(seed)
        V = np.random.permutation(V)
        np.random.seed(seed)
        item_weight = np.random.permutation(item_weight)

        print("Train...CNN module")
        history = self.model.fit(X_train,V,
                                 verbose=0, batch_size=self.batch_size, epochs=self.nb_epoch,
                                 sample_weight={'output': item_weight})
        return history

    def get_projection_layer(self, X_train):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        Y = self.model.predict(X_train, batch_size=self.batch_size)
        return Y
