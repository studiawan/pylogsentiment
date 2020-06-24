import os
import pickle
import re
import csv
import numpy as np
from optparse import OptionParser
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk import corpus
from nerlogparser.nerlogparser import Nerlogparser


class PyLogSentiment(object):
    def __init__(self, log_file, output_file=None):
        self.log_file = log_file
        self.output_file = output_file
        self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
        self.parser = Nerlogparser()
        self.stopwords = corpus.stopwords.words('english')
        self.MAX_PAD = 10

    def __parse_logs(self):
        # parse log files
        parsed_logs = self.parser.parse_logs(self.log_file)

        return parsed_logs

    def __preprocess(self, message):
        # split
        message = message.lower()
        message = message.replace('=', ' ')
        message = message.replace('/', ' ')
        message = message.replace('-', ' ')
        line = message.split()

        # get alphabet only
        line_split = []
        for li in line:
            alphabet_only = re.sub('[^a-zA-Z]', '', li)
            if alphabet_only != '':
                line_split.append(alphabet_only)

        # remove word with length only 1 character
        for index, word in enumerate(line_split):
            if len(word) == 1:
                line_split[index] = ''

        # remove stopwords
        preprocessed_message = []
        for word in line_split:
            if word != '':
                if word not in self.stopwords:
                    preprocessed_message.append(word)

        return preprocessed_message

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

    def __load_model(self, x_test):
        # load model
        model_file = os.path.join(self.base_path, 'best-model-pylogsentiment.hdf5')
        model = load_model(model_file)

        # convert and predict
        x_test = np.asarray(x_test)
        y_prob = model.predict(x_test)
        y_pred = y_prob.argmax(axis=-1)

        return y_pred

    def __save_to_csv(self, y_pred):
        # output csv file
        if self.output_file is None:
            self.output_file = os.path.join(self.log_file + '.anomaly-results.csv')

        f_csv = open(self.output_file, 'wt')
        writer = csv.writer(f_csv)

        # save anomaly detection results to csv
        with open(self.log_file) as f:
            for line_index, line in enumerate(f):
                if line not in ['\n', '\r\n']:
                    writer.writerow([y_pred[line_index], line.rstrip()])

        print('Write anomaly detection results to:', self.output_file)
        f_csv.close()

    def detect_anomaly(self):
        # parse logs
        parsed_logs = self.__parse_logs()

        # preprocess
        data_list = []
        for line_id, entities in parsed_logs.items():
            preprocessed_message = self.__preprocess(entities['message'])
            data_list.append(preprocessed_message)

        # convert to integer and padding
        word_index = self.__load_word_index()
        x_pad = self.__get_numerics_padding(data_list, word_index)

        # anomaly detection
        model_file = os.path.join(self.base_path, 'best-model-pylogsentiment.hdf5')
        model = load_model(model_file)

        # convert and predict
        x_test = np.asarray(x_pad)
        y_prob = model.predict(x_test)
        y_pred = y_prob.argmax(axis=-1)

        # save results to csv
        self.__save_to_csv(y_pred)


def main():
    parser = OptionParser(usage='usage: pylogsentiment [options]')
    parser.add_option('-i', '--input',
                      action='store',
                      dest='input_file',
                      help='Input log file.')
    parser.add_option('-o', '--output',
                      action='store',
                      dest='output_file',
                      help='Anomaly detection results.')

    # get options
    (options, args) = parser.parse_args()
    input_file = options.input_file
    output_file = options.output_file

    if options.input_file:
        pylogsentiment = PyLogSentiment(input_file, output_file)
        pylogsentiment.detect_anomaly()

    else:
        print('Please see help: pylogsentiment -h')


if __name__ == "__main__":
    main()
