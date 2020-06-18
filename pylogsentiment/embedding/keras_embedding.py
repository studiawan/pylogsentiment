import os
import numpy as np
import pickle
import random
from math import floor
from keras.preprocessing.sequence import pad_sequences


class KerasEmbedding(object):
    def __init__(self, datasets):
        self.datasets = datasets
        self.glove_file = 'glove.6B.50d.txt'
        self.TRAIN_SIZE = 0.6
        self.VAL_SIZE = 0.2
        self.MAX_PAD = 10
        self.GLOVE_DIM = 50
        self.MAX_NUM_WORDS = 400000
        self.dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets'))
        self.random_seed = 100

    def __read_embedding(self):
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'glove'))
        glove_path = os.path.join(current_path, self.glove_file)
        word_index = {}
        embedding_matrix = np.zeros((self.MAX_NUM_WORDS+1, self.GLOVE_DIM))

        index = 1   # start from 1 because 0 is for out of vocabulary
        with open(glove_path, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vectors = np.asarray(values[1:], dtype='float32')
                word_index[word] = index
                embedding_matrix[index] = vectors
                index += 1

        self.__save_word_index(word_index)
        return word_index, embedding_matrix

    def __read_dataset(self, dataset):
        groundtruth_file = os.path.join(self.dataset_path, dataset, 'log.all.pickle')

        with open(groundtruth_file, 'rb') as f:
            data = pickle.load(f)

        # get data
        data_list = []
        data_label = []
        for line_id, properties in data.items():
            data_label.append(properties['label'])
            data_list.append(properties['message'])
        length = len(data_label)

        return data_list, data_label, length

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

    def __save_word_index(self, word_index):
        word_index_path = os.path.join(self.dataset_path, 'word_index.pickle')
        with open(word_index_path, 'wb') as f:
            pickle.dump(word_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __save_train_val(self, train, val):
        train_path = os.path.join(self.dataset_path, 'train.pickle')
        with open(train_path, 'wb') as f:
            pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)

        val_path = os.path.join(self.dataset_path, 'val.pickle')
        with open(val_path, 'wb') as f:
            pickle.dump(val, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __save_test(self, test, dataset):
        # test is a pickle (data, label)
        test_path = os.path.join(self.dataset_path, dataset, 'test.pickle')
        with open(test_path, 'wb') as handle:
            pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __split_dataset(self, data_pad, labels, dataset_name, dataset_length):
        train_path = os.path.join(self.dataset_path, 'train.pickle')
        val_path = os.path.join(self.dataset_path, 'val.pickle')

        # if splitted dataset already exist, we do not split, just read
        if os.path.isfile(train_path) is True and os.path.isfile(val_path) is True:
            with open(train_path, 'rb') as f:
                train_data = pickle.load(f)

            with open(val_path, 'rb') as f:
                val_data = pickle.load(f)

            x_train_all = train_data['x_train']
            y_train_all = train_data['y_train']
            x_val_all = val_data['x_val']
            y_val_all = val_data['y_val']

        # else: split
        else:
            x_train_all, y_train_all = [], []
            x_val_all, y_val_all = [], []
            prev_index = 0

            for dataset_index, length in enumerate(dataset_length):
                # set index
                start_index = prev_index
                end_index = start_index + length
                prev_index = end_index

                # get data and label per set
                data_per_set = data_pad[start_index:end_index]
                label_per_set = labels[start_index:end_index]

                # check normal and anomaly
                normal_index = []
                anomaly_index = []
                for line_id, label in enumerate(label_per_set):
                    if label == 1:
                        normal_index.append(line_id)
                    elif label == 0:
                        anomaly_index.append(line_id)

                # initialize train, val, and test per set
                train_data = []
                train_label = []
                val_data = []
                val_label = []
                test_data = []
                test_label = []

                # random sequence for normal
                list_len = len(normal_index)
                random.Random(self.random_seed).shuffle(normal_index)

                # split train, val, and test for normal data
                train_length = floor(self.TRAIN_SIZE * list_len)
                val_length = floor(self.VAL_SIZE * list_len)
                for index in normal_index[:train_length]:
                    train_data.append(data_per_set[index])
                    train_label.append(label_per_set[index])

                for index in normal_index[train_length:train_length+val_length]:
                    val_data.append(data_per_set[index])
                    val_label.append(label_per_set[index])

                for index in normal_index[train_length+val_length:]:
                    test_data.append(data_per_set[index])
                    test_label.append(label_per_set[index])

                # random sequence for anomaly
                negative_len = len(anomaly_index)
                random.Random(self.random_seed).shuffle(anomaly_index)

                # split train, val, and test for anomaly data
                train_length = floor(self.TRAIN_SIZE * negative_len)
                val_length = floor(self.VAL_SIZE * negative_len)
                for index in anomaly_index[:train_length]:
                    train_data.append(data_per_set[index])
                    train_label.append(label_per_set[index])

                for index in anomaly_index[train_length:train_length+val_length]:
                    val_data.append(data_per_set[index])
                    val_label.append(label_per_set[index])

                for index in anomaly_index[train_length+val_length:]:
                    test_data.append(data_per_set[index])
                    test_label.append(label_per_set[index])

                x_train_all.extend(train_data)
                y_train_all.extend(train_label)
                x_val_all.extend(val_data)
                y_val_all.extend(val_label)

                test_dict = {
                    'x_test': test_data,
                    'y_test': test_label
                }
                self.__save_test(test_dict, dataset_name[dataset_index])

            # save train, val, test set
            train_dict = {
                'x_train': x_train_all,
                'y_train': y_train_all,
            }
            val_dict = {
                'x_val': x_val_all,
                'y_val': y_val_all
            }
            self.__save_train_val(train_dict, val_dict)

        return x_train_all, y_train_all, x_val_all, y_val_all

    def get_data_and_embedding(self):
        # read word embedding
        word_index, embedding_matrix = self.__read_embedding()

        # read all datasets and merge them
        dataset_list = []
        label_list = []
        dataset_name = []
        dataset_length = []

        for dataset in self.datasets:
            data, label, length = self.__read_dataset(dataset)
            dataset_list.extend(data)
            label_list.extend(label)
            dataset_name.append(dataset)
            dataset_length.append(length)

        # pad and split
        data_pad = self.__get_numerics_padding(dataset_list, word_index)
        x_train, y_train, x_val, y_val = self.__split_dataset(data_pad, label_list, dataset_name, dataset_length)

        return x_train, y_train, x_val, y_val, word_index, embedding_matrix


if __name__ == '__main__':
    keras_embedding = KerasEmbedding(['casper-rw', 'dfrws-2009-jhuisi', 'dfrws-2009-nssal', 'honeynet-challenge7'])
    keras_embedding.get_data_and_embedding()
