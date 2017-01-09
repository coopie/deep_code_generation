
import numpy as np

from pipelines.data_sources import HuzzerSource, CharSplitter, OneHotVecotorizerASCII, OneHotVecotorizer, TokenDataSource


def test_huzzer():
    """
    Test deteminism.
    """
    huzz = HuzzerSource()

    for i in range(20):
        x = huzz[str(i)]
        y = huzz[str(i)]
        assert x == y, 'Accessing same element produces different outcome.'


def test_char_splitter():
    char_splitter = CharSplitter(HuzzerSource())

    # check deteminism
    for i in range(20):
        x = char_splitter['{0}/{0}'.format(i)]
        y = char_splitter['{0}/{0}'.format(i)]
        assert x == y, 'Accessing same element produces different outcome.'


def test_one_hot_ascii():
    expected_length = 10
    one_hotter = OneHotVecotorizerASCII(CharSplitter(HuzzerSource()), total_string_length=expected_length)

    # check deteminism and dimensions
    for i in range(20, 60):
        x = one_hotter['{0}/{0}'.format(i)]
        y = one_hotter['{0}/{0}'.format(i)]
        assert np.array_equal(x, y), 'Accessing same element produces different outcome.'
        assert x.shape == (expected_length, 128), 'Incorrect shape for output.'


def test_one_hot():
    expected_shape = (256, 54)
    one_hotter = OneHotVecotorizer(TokenDataSource(HuzzerSource()), 54, 256)

    # check deteminism and dimensions
    for i in range(20, 60):
        x = one_hotter['{0}'.format(i)]
        y = one_hotter['{0}'.format(i)]
        assert np.array_equal(x, y), 'Accessing same element produces different outcome.'
        assert x.shape == expected_shape, 'Incorrect shape for output.'
