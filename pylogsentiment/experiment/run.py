import os
import sys
import pickle
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


class RunPylogsentiment(object):
    # run the pylogsentiment on a particular dataset
    def __init__(self, dataset):
        self.dataset = dataset
        self.dropout = 0.4
        self.MAX_PAD = 10
        self.GLOVE_DIM = 50
        self.units = 256
        self.activation = 'tanh'
        self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets'))

    @staticmethod
    def __evaluation(true_label, predicted_label):
        precision, recall, f1, _ = precision_recall_fscore_support(true_label, predicted_label, average='macro')
        accuracy = accuracy_score(true_label, predicted_label)

        return precision * 100, recall * 100, f1 * 100, accuracy * 100

    def __read_data(self):
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', self.dataset))
        groundtruth_file = os.path.join(current_path, 'log.all.pickle')

        with open(groundtruth_file, 'rb') as f:
            data = pickle.load(f)

        # get data
        data_list = []
        data_label = []
        for line_id, properties in data.items():
            data_label.append(properties['label'])
            data_list.append(properties['message'])

        return data_list, data_label

    def __load_word_index(self):
        path = os.path.join(self.base_path, 'word_index.pickle')
        with open(path, 'rb') as f:
            word_index = pickle.load(f)

        return word_index

    def __get_numerics_padding(self, data_list, word_index):
        # get integer representation of log message based on word index
        numeric_data = []
        for message in data_list:
            numeric_message = []
            for word in message:
                try:
                    numeric_message.append(word_index[word])
                except KeyError:
                    numeric_message.append(0)

            numeric_data.append(numeric_message)

        # padding
        data_pad = pad_sequences(numeric_data, maxlen=self.MAX_PAD, padding='post', truncating='post')

        return data_pad

    def __load_model(self, x_test, y_test):
        # load model
        model_file = os.path.join(self.base_path, 'best-model-pylogsentiment.hdf5')
        model = load_model(model_file)

        # convert and predict
        x_test = np.asarray(x_test)
        y_prob = model.predict(x_test)
        y_pred = y_prob.argmax(axis=-1)

        # evaluation metrics
        precision, recall, f1, accuracy = self.__evaluation(y_test, y_pred)

        return precision, recall, f1, accuracy

    def detect_anomaly(self):
        x, y = self.__read_data()
        word_index = self.__load_word_index()
        x_pad = self.__get_numerics_padding(x, word_index)
        precision, recall, f1, accuracy = self.__load_model(x_pad, y)

        print(self.dataset, precision, recall, f1, accuracy)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        dataset_name = sys.argv[1]
        detection = RunPylogsentiment(dataset_name)
        detection.detect_anomaly()

    else:
        print('python run.py dataset_name')
        sys.exit(1)
