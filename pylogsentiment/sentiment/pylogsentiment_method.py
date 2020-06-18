import os
import numpy as np
import pickle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from keras.models import Sequential, load_model
from keras.layers import Embedding, SpatialDropout1D, Dense, GRU
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from pylogsentiment.imbalance.sampling import Sampling


class PyLogSentimentMethod(object):
    def __init__(self, x_train, y_train, x_val, y_val, word_index, embedding_matrix, sampler_name='tomek-links'):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.word_index = word_index
        self.embedding_matrix = embedding_matrix
        self.sampler_name = sampler_name
        self.model = None
        self.dropout = 0.4
        self.units = 256
        self.activation = 'tanh'
        self.batch_size = 128
        self.epochs = 20
        self.MAX_PAD = 10
        self.GLOVE_DIM = 50
        self.MAX_NUM_WORDS = 400000
        self.sampling = Sampling(sampler_name)
        self.sampler = self.sampling.get_sampler()
        self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets'))

    @staticmethod
    def __evaluation(true_label, predicted_label):
        precision, recall, f1, _ = precision_recall_fscore_support(true_label, predicted_label, average='macro')
        accuracy = accuracy_score(true_label, predicted_label)

        return precision * 100, recall * 100, f1 * 100, accuracy * 100

    def train_pylogsentiment(self):
        # build model and compile
        embedding_layer = Embedding(self.MAX_NUM_WORDS+1,
                                    self.GLOVE_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.MAX_PAD,
                                    trainable=False)
        model = Sequential()
        model.add(embedding_layer)
        model.add(SpatialDropout1D(self.dropout))
        model.add(GRU(self.units, dropout=self.dropout, recurrent_dropout=self.dropout, activation=self.activation))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        sampled_data_path = os.path.join(self.base_path, 'train_val_resample.pickle')
        if self.sampler_name != '':
            if os.path.exists(sampled_data_path):
                with open(sampled_data_path, 'rb') as f:
                    resample_pickle = pickle.load(f)
                    train_resample = resample_pickle['train_resample']
                    train_label_resample = resample_pickle['train_label_resample']
                    val_resample = resample_pickle['val_resample']
                    val_label_resample = resample_pickle['val_label_resample']

            else:
                # sample the data
                print('Resampling data ...')
                train_resample, train_label_resample = self.sampler.fit_resample(self.x_train, self.y_train)
                train_resample = np.asarray(train_resample)
                train_label_resample = to_categorical(train_label_resample)

                val_resample, val_label_resample = self.sampler.fit_resample(self.x_val, self.y_val)
                val_resample = np.asarray(val_resample)
                val_label_resample = to_categorical(val_label_resample)

                train_val = {
                    'train_resample': train_resample,
                    'train_label_resample': train_label_resample,
                    'val_resample': val_resample,
                    'val_label_resample': val_label_resample
                }

                with open(sampled_data_path, 'wb') as f:
                    pickle.dump(train_val, f, protocol=pickle.HIGHEST_PROTOCOL)

            # save the best model
            model_file = os.path.join(self.base_path, 'best-model-pylogsentiment.hdf5')
            earlystop = EarlyStopping(monitor='loss', patience=5, verbose=1, mode='min')
            checkpoint = ModelCheckpoint(model_file, monitor='loss', verbose=1, save_best_only=True,
                                         save_weights_only=False, mode='min')

            # training
            model.fit(train_resample, train_label_resample, validation_data=(val_resample, val_label_resample),
                      batch_size=self.batch_size, epochs=self.epochs, callbacks=[earlystop, checkpoint])

        else:
            x_train = np.asarray(self.x_train)
            y_train = to_categorical(self.y_train)
            x_val = np.asarray(self.x_val)
            y_val = to_categorical(self.y_val)
            model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=self.batch_size, epochs=self.epochs)

        self.model = model
        print(model.summary())

    def test_pylogsentiment(self, x_test, y_test):
        # load model + test
        x_test = np.asarray(x_test)

        model_file = os.path.join(self.base_path, 'best-model-pylogsentiment.hdf5')
        model = load_model(model_file)

        y_prob = model.predict(x_test)
        y_pred = y_prob.argmax(axis=-1)

        # evaluation metrics
        precision, recall, f1, accuracy = self.__evaluation(y_test, y_pred)

        return precision, recall, f1, accuracy
