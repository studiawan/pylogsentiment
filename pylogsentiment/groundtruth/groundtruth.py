import os
import pickle
import re
import sys
from time import time
from nltk import corpus
from nerlogparser.nerlogparser import Nerlogparser
from pylogsentiment.misc.grammar import LogGrammar


class GroundTruth(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.parser = Nerlogparser()
        self.stopwords = corpus.stopwords.words('english')

    @staticmethod
    def __loading(number):
        if number % 1000 == 0:
            s = str(number) + ' ...'
            print(s, end='')
            print('\r', end='')

    @staticmethod
    def __read_wordlist(log_type):
        # read word list of particular log type
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'wordlist'))

        # open word list files in the specified directory
        wordlist_path = os.path.join(current_path, log_type + '.txt')
        with open(wordlist_path, 'r') as f:
            wordlist_temp = f.readlines()

        # get word list
        wordlist = []
        for wl in wordlist_temp:
            wordlist.append(wl.strip())

        return wordlist

    def __get_preprocessed_logs(self, log_file):
        # parse log files
        parsed_logs = self.parser.parse_logs(log_file)

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

    @staticmethod
    def __set_anomaly_label(wordlist, parsed_logs):
        anomaly_label = {}

        # check sentiment for each log line
        for line_id, parsed in parsed_logs.items():
            log_lower = parsed['message'].lower().strip()

            # 0 = negative
            # 1 = positive
            label = 1
            for word in wordlist:
                # negative sentiment
                if word in log_lower:
                    label = 0
                    anomaly_label[line_id] = label
                    break

            # positive sentiment
            if label == 1:
                anomaly_label[line_id] = label

        return anomaly_label

    def __save_groundtruth(self, groundtruth):
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', self.dataset))
        groundtruth_file = os.path.join(current_path, 'log.all.pickle')
        with open(groundtruth_file, 'wb') as handle:
            pickle.dump(groundtruth, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_ground_truth(self):
        # get log file
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets',
                                                    self.dataset, 'logs'))
        log_files = os.listdir(current_path)

        groundtruth = {}
        groundtruth_id = 0
        for log_file in log_files:
            # set path
            file_path = os.path.join(current_path, log_file)

            parsed_logs = self.__get_preprocessed_logs(file_path)

            # get log type
            log_type = log_file.split('.')[0].lower()

            # set label for each line in a log file
            wordlist = self.__read_wordlist(log_type)
            print('\nProcessing', log_file, '...')

            # get label
            anomaly_label = self.__set_anomaly_label(wordlist, parsed_logs)

            for line_id, label in anomaly_label.items():
                preprocessed_message = self.__preprocess(parsed_logs[line_id]['message'])

                groundtruth[groundtruth_id] = {
                    'message': preprocessed_message,
                    'label': label
                }
                groundtruth_id += 1

        # save ground truth
        self.__save_groundtruth(groundtruth)

    def get_ground_truth_warn(self):
        # get log file
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets',
                                                    self.dataset, 'logs'))
        log_files = os.listdir(current_path)

        groundtruth = {}
        groundtruth_id = 0
        for log_file in log_files:
            # set path
            file_path = os.path.join(current_path, log_file)

            # log parsing
            parsed_logs = self.__get_preprocessed_logs(file_path)

            for line_id, entities in parsed_logs.items():
                label = 0
                flag = 0
                if 'status' in entities.keys() or 'service' in entities.keys():
                    try:
                        if 'WARN' in entities['status']:
                            label = 0
                            flag = 1

                    except KeyError:
                        if 'WARN' in entities['service']:
                            label = 0
                            flag = 1

                if flag == 0:
                    label = 1

                preprocessed_message = self.__preprocess(entities['message'])
                groundtruth[groundtruth_id] = {
                    'message': preprocessed_message,
                    'label': label
                }
                groundtruth_id += 1
                self.__loading(groundtruth_id)

        # save ground truth
        self.__save_groundtruth(groundtruth)

    def get_ground_truth_dash(self, log_type):
        # get log file
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
                                                    'datasets', self.dataset, 'logs'))
        log_files = os.listdir(current_path)

        groundtruth = {}
        groundtruth_id = 0
        grammar_parser = LogGrammar(log_type)

        for log_file in log_files:
            # set path
            file_path = os.path.join(current_path, log_file)

            with open(file_path, 'r') as f:
                for line in f:
                    # parse log entry
                    if log_type == 'blue-gene':
                        parsed = grammar_parser.parse_bluegenelog(line)                   

                    # check 'sock' entity
                    label = 0
                    if 'sock' in parsed.keys():
                        if parsed['sock'] != '-':
                            label = 0
                        elif parsed['sock'] == '-':
                            label = 1

                    # check 'message' entity
                    if 'message' in parsed.keys():
                        preprocessed_message = self.__preprocess(parsed['message'])
                    else:
                        preprocessed_message = 'null'

                    groundtruth[groundtruth_id] = {
                        'message': preprocessed_message,
                        'label': label
                    }
                    groundtruth_id += 1
                    self.__loading(groundtruth_id)
        
        # save ground truth
        self.__save_groundtruth(groundtruth)

if __name__ == '__main__':
    dataset_list = ['casper-rw', 'dfrws-2009-jhuisi', 'dfrws-2009-nssal', 'honeynet-challenge7',
                    'blue-gene', 'zookeeper', 'hadoop', 'honeynet-challenge5']
    if len(sys.argv) < 2:
        print('Please input dataset name.')
        print('python groundtruth.py dataset_name')
        print('Supported datasets:', dataset_list)
        sys.exit(1)

    else:
        start = time()
        dataset_name = sys.argv[1]
        gt = GroundTruth(dataset_name)

        if dataset_name in ['blue-gene']:
            gt.get_ground_truth_dash(dataset_name)

        elif dataset_name in ['zookeeper', 'hadoop']:
            gt.get_ground_truth_warn()

        elif dataset_name in ['casper-rw', 'dfrws-2009-jhuisi', 'dfrws-2009-nssal', 'honeynet-challenge7',
                              'honeynet-challenge5']:
            gt.get_ground_truth()

        # print runtime
        duration = time() - start
        minute, second = divmod(duration, 60)
        hour, minute = divmod(minute, 60)
        print("Runtime: %d:%02d:%02d" % (hour, minute, second))
