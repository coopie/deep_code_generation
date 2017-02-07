import project_context  # NOQA
import numpy as np
from lazychef.data_sources import Datasource

from pipelines.generators import SequenceGenerator


def test_sequence_gen():
    sg = SequenceGenerator(DummyDataSource(), batch_size=2)

    x, y = next(sg)

    assert len(x) == 2
    assert len(y) == 2

    assert np.all(x[0] == data[0, :-1])
    assert np.all(x[1] == data[1, :-1])

    assert np.all(y[0] == data[0, -1])
    assert np.all(y[1] == data[1, -1])

    # probabalistic checking that shuflfing works
    for i in range(200):
        sg = SequenceGenerator(DummyDataSource(), batch_size=2)
        sg.shuffle()
        new_x, new_y = next(sg)
        if (not np.all(new_x == x)) or (not np.all(new_y == y)):
            return

    # we reach here if shuffling 20 times yeilds same results
    assert False, 'shuffling the generator failed to produce different results after 200 attmepts'


a = np.array([0, 1, 2, 3])
b = np.array([0, 2, 4, 6])
c = np.array([0, 3, 6, 9])


data = np.array([
    [a, a, a, b],
    [a, a, a, b],
    [b, b, b, c],
    [a, a, a, b],
    [b, c, b, c],
    [a, b, a, b],
    [b, b, c, c],
    [a, a, a, a],
    [b, b, c, c],
    [c, c, c, c]
])


class DummyDataSource(Datasource):
    def _process(self, key):
        return data[key]

    def __len__(self):
        return len(data)
