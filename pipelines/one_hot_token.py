import numpy as np
from lazychef.data_sources import (
    CachedTTVArrayLikeDataSource, TTVArrayLikeDataSource, LambdaDataSource
)
from os.path import join as join_path

from .data_sources import HuzzerSource, TokenDataSource, OneHotVecotorizer


def one_hot_token_pipeline(ttv, use_cache=False, cache_name='one_hot_token_pipeline'):
    """
    For models which predict the next character in a sequence.
    """

    data_source = reshape_for_cnn(
        OneHotVecotorizer(
            TokenDataSource(HuzzerSource()), 54, 256)
    )

    if use_cache:
        return CachedTTVArrayLikeDataSource(
            ttv=ttv,
            data_name='data',
            cache_name=join_path('pipelines', cache_name),
            data_source=data_source
        )
    else:
        return TTVArrayLikeDataSource(
            ttv=ttv, data_source=data_source
        )


def reshape_for_cnn(ds):
    def f(x):
        return np.reshape(x, x.shape + (1,))
    return LambdaDataSource(f, ds)
