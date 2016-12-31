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
    Take some code, and pick a random character for the proceeding character. Returns all code up to that character.
    The characters are all reaturned in one string, i.e. the final character of the string is the target value.

    Takes urls as '<code_seed>/<splitting_seed>' where both seeds are integers.
    """

    def __init__(self, code_ds):
        self.data_source = code_ds
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
        return prior_string + code[result_char_idx]


def OneHotVecotorizer(split_ds, total_string_length=33):
    """
    Take ascii strings from a CharSplitter like datasource and turns the chars into one-hot vectors of length 128.
    """
    def one_hoterize(data):
        nonlocal total_string_length  # NOQA
        data = data[-total_string_length:]
        for x in data:
            assert ord(x) < 128, 'character {} in {} is not ascii'.format(x, data)

        one_hots = np.zeros((len(data), 128), np.uint8)
        one_hots[np.arange(len(data)), [ord(c) for c in data]] = 1

        # add padding
        if one_hots.shape[0] != total_string_length:
            padding_to_add = total_string_length - one_hots.shape[0]
            padding = np.zeros((padding_to_add, 128), np.uint8)
            one_hots = np.concatenate((padding, one_hots))

        assert one_hots.shape == (total_string_length, 128)
        return one_hots

    return LambdaDataSource(one_hoterize, split_ds)


def vec_to_char(vec):
    return chr(np.nonzero(vec)[0][0])
