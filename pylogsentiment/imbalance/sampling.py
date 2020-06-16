import os
from imblearn.under_sampling import TomekLinks


class Sampling(object):
    def __init__(self, sampler):
        self.sampler = sampler
        self.random_seed = 101
        self.njobs = os.cpu_count()

    def get_sampler(self):
        sampler = None

        if self.sampler == 'tomek-links':
            sampler = TomekLinks(random_state=self.random_seed, n_jobs=self.njobs)

        return sampler
