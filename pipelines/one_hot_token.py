import numpy as np
from lazychef.data_sources import LambdaDatasource, ArrayDatasource, CachedArrayDatasource, CachedDatasource
from lazychef.generators import DatasetGenerator, ShuffleDatasetCallback, LogEpochEndCallback
from .data_sources import HuzzerSource, TokenDatasource, OneHotVecotorizer


def one_hot_token_pipeline(
    ttv=None,
    for_cnn=True,
    length=256
):
    """
    """
    data_source = OneHotVecotorizer(
        TokenDatasource(HuzzerSource()),
        54,
        length
    )

    if for_cnn:
        data_source = reshape_for_cnn(data_source)

    return data_source


def one_hot_token_dataset(
    batch_size,
    number_of_batches,
    length=128,
    cache_path=None,
):
    data_source = one_hot_token_pipeline(for_cnn=False, length=length)

    fs_data_source = FixedSizeArrayDatasource(data_source, batch_size * number_of_batches)

    if cache_path is not None:
        fs_data_source = CachedArrayDatasource(fs_data_source, cache_path)

    callbacks = [ShuffleDatasetCallback(seed=1337), LogEpochEndCallback()]

    generator = DatasetGenerator(
        [fs_data_source],
        batch_size,
        callbacks
    )
    return generator


def one_hot_variable_length_token_dataset(
    batch_size,
    number_of_batches,
    cache_path=None,
    zero_front_pad=0
):
    token_pipeline = one_hot_token_pipeline(for_cnn=False, length=None)

    if zero_front_pad > 0:
        token_pipeline = LambdaDatasource(token_pipeline, pad_zeros(zero_front_pad))

    if cache_path is not None:
        fs_data_source = CachedDatasource(token_pipeline, cache_path)

    fs_data_source = FixedSizeArrayDatasource(
        fs_data_source, batch_size * number_of_batches
    )
    callbacks = [ShuffleDatasetCallback(seed=1337), LogEpochEndCallback()]

    generator = DatasetGenerator(
        [fs_data_source],
        batch_size,
        callbacks
    )
    return generator


def reshape_for_cnn(ds):
    def f(x):
        return np.reshape(x, x.shape + (1,))
    return LambdaDatasource(ds, f)


class FixedSizeArrayDatasource(ArrayDatasource):
    """
    Takes a datasource (like the one_hot_token_pipeline), which take ints as strings, and makes a
    fixed size ArrayDatasource.
    """
    def __init__(self, ds, size):
        self.ds = ds
        self.size = size

    def _process(self, index):
        assert index >= 0 and index < self.size, 'Index {} out of bounds'.format(index)
        return self.ds[str(index)]

    def __len__(self):
        return self.size


def pad_zeros(padding):
    def pad(x):
        return np.concatenate(
            (
                np.zeros((padding, x.shape[1]), dtype=x.dtype),
                x
            ),
            axis=0
        )

    return pad
