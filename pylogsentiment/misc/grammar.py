import sys
from pyparsing import Word, alphas, Combine, nums, Regex, ParseException, string, Optional


class LogGrammar(object):
    """A class to define the format (grammar) of a log file. We heavily rely on pyparsing in this case.
    """

    def __init__(self, log_type=None):
        """The constructor of LogGrammar.
        """
        self.log_type = log_type
        if self.log_type == 'blue-gene':
            self.bluegene_grammar = self.__get_bluegene_grammar()
        elif self.log_type == 'spark':
            self.spark_grammar = self.__get_spark_grammar()
        elif self.log_type == 'windows':
            self.windows_grammar = self.__get_windows_grammar()

    @staticmethod
    def loading(number):
        if number % 1000 == 0:
            s = str(number) + ' ...'
            print(s, end='')
            print('\r', end='')

    @staticmethod
    def __get_bluegene_grammar():
        """The definition of BlueGene/L grammar.

        The BlueGene/L logs can be downloaded from [Useninx2006a]_ and
        this data was used in [Stearley2008]_.

        Returns
        -------
        bluegene_grammar    :
            Grammar for BlueGene/L supercomputer logs.

        References
        ----------
        .. [Usenix2006a]  The HPC4 data. URL: https://www.usenix.org/cfdr-data#hpc4
        .. [Stearley2008] Stearley, J., & Oliner, A. J. Bad words: Finding faults in Spirit's syslogs.
                          In 8th IEEE International Symposium on Cluster Computing and the Grid, pp. 765-770.
        """
        ints = Word(nums)

        sock = Word(alphas + '-' + '_')
        number = ints
        date = Combine(ints + '.' + ints + '.' + ints)
        core1 = Word(alphas + nums + '-' + ':' + '_')
        datetime = Combine(ints + '-' + ints + '-' + ints + '-' + ints + '.' + ints + '.' + ints + '.' + ints)
        core2 = Word(alphas + nums + '-' + ':' + '_')
        source = Word(alphas + '(' + ')')
        service = Word(alphas + '_')
        info_type = Word(alphas)
        message = Regex('.*')

        # blue gene log grammar
        bluegene_grammar = sock + number + date + core1 + datetime + core2 + source + service + info_type + message
        return bluegene_grammar

    def parse_bluegenelog(self, log_line):
        """Parse the BlueGene/L logs based on defined grammar.

        Parameters
        ----------
        log_line    : str
            A log line to be parsed

        Returns
        -------
        parsed      : dict[str, str]
            A parsed BlueGene/L log.
        """
        parsed = dict()
        try:
            parsed_bluegenelog = self.bluegene_grammar.parseString(log_line)
            parsed['sock'] = parsed_bluegenelog[0]
            parsed['number'] = parsed_bluegenelog[1]
            parsed['date'] = parsed_bluegenelog[2]
            parsed['core1'] = parsed_bluegenelog[3]
            parsed['timestamp'] = parsed_bluegenelog[4]
            parsed['core2'] = parsed_bluegenelog[5]
            parsed['source'] = parsed_bluegenelog[6]
            parsed['service'] = parsed_bluegenelog[7]
            parsed['info_type'] = parsed_bluegenelog[8]
            parsed['message'] = parsed_bluegenelog[9]

        except ParseException as e:
            print(repr(e))
            print(log_line)

        return parsed    

    @staticmethod
    def __get_spark_grammar():
        ints = Word(nums)

        date = Optional(Combine(ints + '/' + ints + '/' + ints))
        time = Optional(Combine(ints + ":" + ints + ":" + ints))
        status = Optional(Word(string.ascii_uppercase))
        service = Optional(Word(alphas + nums + '/' + '-' + '_' + '.' + '[' + ']' + ':' + '$'))
        message = Regex('.*')

        spark_grammar = date.setResultsName('date') + time.setResultsName('time') + status.setResultsName('status') + \
            service.setResultsName('service') + message.setResultsName('message')

        return spark_grammar

    def parse_spark(self, log_line):
        parsed = dict()
        try:
            parsed_spark = self.spark_grammar.parseString(log_line)
            parsed['date'] = parsed_spark.date
            parsed['time'] = parsed_spark.time
            parsed['status'] = parsed_spark.status
            parsed['service'] = parsed_spark.service
            parsed['message'] = parsed_spark.message

        except ParseException as e:
            print(repr(e))
            print(log_line)

        return parsed

    @staticmethod
    def __get_windows_grammar():
        ints = Word(nums)

        date = Optional(Combine(ints + '-' + ints + '-' + ints))
        time = Optional(Combine(ints + ":" + ints + ":" + ints + ','))
        status = Optional(Word(string.ascii_uppercase + string.ascii_lowercase))
        service = Optional(Word(string.ascii_uppercase))
        message = Regex('.*')

        windows_grammar = date.setResultsName('date') + time.setResultsName('time') + \
            status.setResultsName('status') + service.setResultsName('service') + message.setResultsName('message')

        return windows_grammar

    def parse_windows(self, log_line):
        parsed = dict()
        try:
            parsed_windows = self.windows_grammar.parseString(log_line)
            parsed['date'] = parsed_windows.date
            parsed['time'] = parsed_windows.time
            parsed['status'] = parsed_windows.status
            parsed['service'] = parsed_windows.service
            parsed['message'] = parsed_windows.message

        except ParseException as e:
            print(repr(e))
            print(log_line)

        return parsed


if __name__ == '__main__':
    logtype = sys.argv[1]
    filename = '/home/hudan/Git/pylogsentiment/datasets/' + logtype + '/logs/' + logtype + '.log'
    lg = LogGrammar(logtype)

    index = 0
    with open(filename, 'r', encoding='utf-8-sig', errors='ignore') as f:
        for line in f:
            # get parsed line and print
            parsed_line = None            
            if logtype == 'bgl':
                parsed_line = lg.parse_bluegenelog(line)
            elif logtype == 'spark':
                parsed_line = lg.parse_spark(line)
            elif logtype == 'windows':
                parsed_line = lg.parse_windows(line)

            lg.loading(index)
            index += 1
            # print(parsed_line)
