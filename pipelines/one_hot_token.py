import numpy as np
from lazychef.data_sources import LambdaDatasource
from .data_sources import HuzzerSource, TokenDatasource, OneHotVecotorizer


def one_hot_token_pipeline(
    ttv=None,
    use_cache=False,
    for_cnn=True,
    cache_name='one_hot_token_pipeline',
    length=256
):
    """
    For models which predict the next character in a sequence.
    """
    data_source = OneHotVecotorizer(
        TokenDatasource(HuzzerSource()),
        54,
        length
    )

    if for_cnn:
        data_source = reshape_for_cnn(data_source)

    return data_source


def reshape_for_cnn(ds):
    def f(x):
        return np.reshape(x, x.shape + (1,))
    return LambdaDatasource(f, ds)
