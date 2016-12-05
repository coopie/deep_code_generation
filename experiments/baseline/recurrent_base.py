import project_context  #  NOQA

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import LambdaCallback, ModelCheckpoint

from tqdm import tqdm, trange

from pipelines.one_hot_ascii import one_hot_ascii_pipeline
from pipelines.generators import SequenceGenerator


def run_experiment():
    # building test, train, validation split
    number_of_samples = 1280000
    string_length = 32

    # make the test, train, validation split
    ttv = make_ttv(number_of_samples)

    # set up pipeline

    print('Setting up data pipeline')
    data_pipeline = one_hot_ascii_pipeline(ttv, string_length=string_length, use_cache=False)

    train_data = data_pipeline.get_set('train')
    train_gen = SequenceGenerator(data_source=train_data, batch_size=128)

    validation_data = data_pipeline.get_set('validation')
    validation_gen = SequenceGenerator(data_source=validation_data, batch_size=128)

    def shuffle_train_data(*args):
        nonlocal train_gen
        train_gen.shuffle()

    model = make_model(string_length)

    checkpoint = ModelCheckpoint(
        'experiments/baseline/recurrent_base_models/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        save_best_only=True
    )
    data_shuffler = LambdaCallback(on_epoch_end=shuffle_train_data)

    model.fit_generator(
        generator=train_gen,
        samples_per_epoch=len(train_gen),
        nb_epoch=5,

        validation_data=validation_gen,
        nb_val_samples=len(validation_gen),

        callbacks=[checkpoint, data_shuffler]
    )

    pass


def make_model(string_length):
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(string_length, 128)))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


def make_example_uri(ident):
    return '{0}/{0}'.format(ident)


def make_ttv(number_of_samples):
    train_size = int(0.8 * number_of_samples)
    test_size = int(0.1 * number_of_samples)
    validation_size = int(0.1 * number_of_samples)  # NOQA

    train_idxs = range(0, train_size)
    test_idxs = range(train_size, train_size+test_size)
    validation_idxs = range(train_size + test_size, number_of_samples)

    ttv = {
        'test': [make_example_uri(x) for x in test_idxs],
        'train': [make_example_uri(x) for x in train_idxs],
        'validation': [make_example_uri(x) for x in validation_idxs]
    }
    return ttv


if __name__ == '__main__':
    run_experiment()
