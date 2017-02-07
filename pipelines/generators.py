from lazychef.generators import Generator
import numpy as np


class SequenceGenerator(Generator):
    def __init__(self, data_source, batch_size=128):
        super().__init__([data_source], batch_size)
        self.shuffle_idxs = np.arange(len(data_source))

    def __next__(self):
        if self.chunk_index == len(self):
            self.chunk_index = 0

        data_idxs = self.shuffle_idxs[self.chunk_index:self.chunk_index + self.batch_size]

        data = self.data_sources[0][data_idxs]

        self.chunk_index += self.batch_size
        data = np.array(data)

        x = data[np.arange(self.batch_size), :-1]
        y = data[np.arange(self.batch_size), -1]

        return (x, y)

    def shuffle(self):
        np.random.shuffle(self.shuffle_idxs)

    def __len__(self):
        """Return the size of the data, to the nearest `batch_size`"""
        return len(self.data_sources[0]) - (len(self.data_sources[0]) % self.batch_size)
