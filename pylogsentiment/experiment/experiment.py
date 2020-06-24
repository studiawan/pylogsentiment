import os
import sys
import csv
import pickle
from pylogsentiment.embedding.keras_embedding import KerasEmbedding
from pylogsentiment.sentiment.pylogsentiment_method import PyLogSentimentMethod


class Experiment(object):
    def __init__(self, datasets, method):
        self.datasets = datasets
        self.method = method
        self.dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets'))

    def __get_embedding(self):
        keras_embedding = KerasEmbedding(self.datasets)
        x_train, y_train, x_val, y_val, word_index, embedding_matrix = keras_embedding.get_data_and_embedding()

        return x_train, y_train, x_val, y_val, word_index, embedding_matrix

    def __read_test_set(self, dataset):
        test_path = os.path.join(self.dataset_path, dataset, 'test.pickle')
        with open(test_path, 'rb') as handle:
            data = pickle.load(handle)

        x_test = data['x_test']
        y_test = data['y_test']

        return x_test, y_test

    def run(self):
        # embedding and initiate model
        x_train, y_train, x_val, y_val, word_index, embedding_matrix = self.__get_embedding()
        model = PyLogSentimentMethod(x_train, y_train, x_val, y_val, word_index, embedding_matrix)

        # train        
        if self.method == 'pylogsentiment':
            model.train_pylogsentiment()
        else:
            print('Unsupported method.')
            sys.exit(1)

        # set evaluation file
        evaluation_file = os.path.join(self.dataset_path, self.method + '.evaluation.csv')
        f = open(evaluation_file, 'wt')
        writer = csv.writer(f)

        # test to each dataset, save performance metrics
        for dataset in self.datasets:
            x_test, y_test = self.__read_test_set(dataset)
            precision, recall, f1, accuracy = model.test_pylogsentiment(x_test, y_test)
            writer.writerow([dataset, self.method, precision, recall, f1, accuracy])

        f.close()


if __name__ == '__main__':
    dataset_list = ['casper-rw', 'dfrws-2009-jhuisi', 'dfrws-2009-nssal', 'honeynet-challenge7',
                    'zookeeper', 'hadoop', 'blue-gene', 'spark', 'honeynet-challenge5', 'windows']

    if len(sys.argv) < 2:
        print('Please input method name.')
        print('python experiment.py pylogsentiment')
        sys.exit(1)

    else:
        input_name = sys.argv[1]
        experiment = Experiment(dataset_list, input_name)
        experiment.run()
