import numpy as np
from lazychef.data_sources import LambdaDatasource, ArrayDatasource
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
    length=128
):
    data_source = one_hot_token_pipeline(for_cnn=False, length=length)
    fs_data_source = FixedSizeTokenSource(data_source, batch_size * number_of_batches)

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


class FixedSizeTokenSource(ArrayDatasource):
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
