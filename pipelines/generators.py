from lazychef.generators import Generator
import numpy as np


class SequenceGenerator(Generator):
    def __init__(self, data_source, batch_size=128):
        super().__init__(data_source, batch_size)
        self.shuffle_idxs = np.arange(len(data_source))

    def __next__(self):
        if self.chunk_index == len(self):
            self.chunk_index = 0

        data_idxs = self.shuffle_idxs[self.chunk_index:self.chunk_index + self.batch_size]

        data = self.data_source_x[data_idxs]

        # data = self.data_source_x[self.chunk_index:self.chunk_index + self.batch_size]
        self.chunk_index += self.batch_size
        data = np.array(data)

        x = data[np.arange(self.batch_size), :-1]
        y = data[np.arange(self.batch_size), -1]

        return (x, y)

    def shuffle(self):
        np.random.shuffle(self.shuffle_idxs)
