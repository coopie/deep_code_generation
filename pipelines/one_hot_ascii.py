from lazychef.data_sources import CachedTTVArrayLikeDataSource, TTVArrayLikeDataSource
from os.path import join as join_path

from .data_sources import HuzzerSource, CharSplitter, OneHotVecotorizer


def one_hot_ascii_pipeline(ttv, string_length=32, use_cache=True, cache_name='one_hot_ascii_pipeline'):
    """
    For models which predict the next character in a sequence.
    """

    data_source = OneHotVecotorizer(CharSplitter(HuzzerSource()), total_string_length=string_length+1)

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
