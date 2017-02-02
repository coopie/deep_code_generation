from lazychef.generators import DatasetGenerator, ShuffleDatasetCallback, LogEpochEndCallback
from lazychef.data_sources import LambdaArrayDatasource
from keras.datasets import mnist
from .utils import reshape_for_cnn


def mnist_unlabeled():
    (x_train, _), (_, _) = mnist.load_data()
    return x_train / 255


def mnist_unlabeled_generator(batch_size, for_cnn=True):
    data_source = mnist_unlabeled()
    if for_cnn:
        data_source = LambdaArrayDatasource(data_source, reshape_for_cnn)

    return DatasetGenerator(
        [data_source],
        batch_size,
        callbacks=[ShuffleDatasetCallback(seed=1337), LogEpochEndCallback()]
    )
