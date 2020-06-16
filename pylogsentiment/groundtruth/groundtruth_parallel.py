import multiprocessing
import os
import re
import pickle
import sys
from itertools import zip_longest
from nltk import corpus
from pylogsentiment.misc.grammar import LogGrammar


def loading(number):
    if number % 1000 == 0:
        s = str(number) + ' ...'
        print(s, end='')
        print('\r', end='')


def preprocess(message):
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
    preprocessed = []
    for word in line_split:
        if word != '':
            if word not in stopwords:
                preprocessed.append(word)

    return preprocessed


def save_groundtruth(groundtruth_all):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', dataset))
    groundtruth_file = os.path.join(path, 'log.all.pickle')
    with open(groundtruth_file, 'wb') as handle:
        pickle.dump(groundtruth_all, handle, protocol=pickle.HIGHEST_PROTOCOL)


def process_chunk(line):
    if line is not None:
        label = 0
        parsed_logs = None        

        if log_type == 'spark' or log_type == 'windows':
            if log_type == 'spark':
                parsed_logs = grammar_parser.parse_spark(line)
            elif log_type == 'windows':
                parsed_logs = grammar_parser.parse_windows(line)
            log_lower = parsed_logs['message'].lower()

            # get label
            label = 1
            for word in wordlist:
                # negative sentiment
                if word in log_lower:
                    label = 0
                    break

        # preprocess message
        try:
            preprocessed = preprocess(parsed_logs['message'])
        except KeyError:
            preprocessed = ''

        return label, preprocessed

    else:
        return None, None


def grouper(n, iterable, padvalue=None):
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)


def read_wordlist(logtype):
    # read word list of particular log type
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'wordlist'))

    # open word list files in the specified directory
    wordlist_path = os.path.join(path, logtype + '.txt')
    with open(wordlist_path, 'r') as f:
        wordlist_temp = f.readlines()

    # get word list
    word_list = []
    for wl in wordlist_temp:
        word_list.append(wl.strip())

    return word_list


if __name__ == '__main__':
    dataset = sys.argv[1]
    log_type = dataset.split('.')[0]
    stopwords = corpus.stopwords.words('english')
    chunk_size = 1000

    if log_type == 'spark' or log_type == 'windows':
        wordlist = read_wordlist(log_type)

    # get log file
    current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets',
                                                dataset, 'logs'))
    log_files = os.listdir(current_path)

    # set ground truth
    groundtruth = {}
    groundtruth_id = 0
    grammar_parser = LogGrammar(log_type)

    # create pool for multiprocessing
    cpu_total = multiprocessing.cpu_count()
    p = multiprocessing.Pool(cpu_total)

    # for directory in log_files:
    #    subfiles = os.listdir(os.path.join(current_path, directory))
    for log_file in log_files:
        # set path
        file_path = os.path.join(current_path, log_file)
        file_handler = open(file_path, 'r', errors='ignore')

        # process in chunk
        for chunk in grouper(chunk_size, file_handler):
            results = p.map(process_chunk, chunk)
            for data_label, preprocessed_message in results:
                if data_label is not None:
                    groundtruth[groundtruth_id] = {
                        'message': preprocessed_message,
                        'label': data_label
                    }
                    groundtruth_id += 1
                    loading(groundtruth_id)

        file_handler.close()

    save_groundtruth(groundtruth)
