from lazychef.data_sources import DataSource, LambdaDataSource
from huzzer.huzz import huzzer
from random import Random
import numpy as np


class HuzzerSource(DataSource):
    def _process(self, ident):
        assert ident.isdigit(), 'huzzer got {}, when it should take a number'.format(ident)
        return huzzer(int(ident))


class CharSplitter(DataSource):
    """
    Take some code, and extract `string_length` characters and the proceeding character.
    The characters are all reaturned in one string, i.e. the final character of the string is the target value.

    takes urls as '<code_seed>/<splitting_seed>' where both seeds are integers

    TODO: better url parsing for same code with different splitting
    """

    def __init__(self, code_ds, string_length=32):
        self.data_source = code_ds
        self.string_length = string_length
        self.rand = Random()

    def _process(self, ident):
        seeds = ident.split('/')
        assert len(seeds) == 2, 'Got {} instead of <code_seed>,<splitting_seed>'.format(seeds)

        code_seed, splitting_seed = seeds

        assert code_seed.isdigit(), \
            'CharSplitter got {} for code_seed, when it should take a number'.format(code_seed)
        assert splitting_seed.isdigit(), \
            'CharSplitter got {} for splitting_seed, when it should take a number'.format(splitting_seed)

        code = self.data_source[code_seed]

        # split code deteministically
        self.rand.seed(int(splitting_seed))
        result_char_idx = self.rand.randint(0, len(code)-1)

        prior_string = code[:result_char_idx]
        if len(prior_string) > self.string_length:
            prior_string = prior_string[-self.string_length:]
        else:
            prior_string = prior_string.rjust(self.string_length, ' ')

        return prior_string + code[result_char_idx]


def OneHotVecotorizer(split_ds):
    """
    Take ascii strings from a CharSplitter like datasource and turns the chars into one-hot vectors of length 128.
    """
    def one_hoterize(data):
        for x in data:
            assert ord(x) < 128, 'character {} in {} is not ascii'.format(x, data)

        one_hots = np.zeros((len(data), 128), np.uint8)
        one_hots[np.arange(len(data)), [ord(c) for c in data]] = 1

        return one_hots

    return LambdaDataSource(one_hoterize, split_ds)


def vec_to_char(vec):
    return chr(np.nonzero(vec)[0][0])
