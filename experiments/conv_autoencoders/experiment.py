import project_context  #  NOQA

from sys import argv

from keras.models import Model
from keras.layers.convolutional import Convolution2D
from keras.layers import Input
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten, Lambda
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras import backend as K

from tqdm import tqdm, trange

from lazychef.keras_generators import KerasGenerator
from pipelines.one_hot_token import one_hot_token_pipeline



def run_experiment(option):
    # building test, train, validation split
    # number_of_samples = 1280000
    number_of_samples = 32000
    BATCH_SIZE=128

    # make the test, train, validation split
    ttv = make_ttv(number_of_samples)

    # set up pipeline

    print('Setting up data pipeline')
    data_pipeline = one_hot_token_pipeline(ttv, use_cache=False)

    train_data = data_pipeline.get_set('train')
    train_gen = KerasGenerator(data_source=train_data, batch_size=BATCH_SIZE)

    validation_data = data_pipeline.get_set('validation')
    validation_gen = KerasGenerator(data_source=validation_data, batch_size=BATCH_SIZE)

    def shuffle_train_data(*args):
        nonlocal train_gen
        train_gen.shuffle()
    data_shuffler = LambdaCallback(on_epoch_end=shuffle_train_data)

    if option == 'simple':
        model, encoder, generator = make_simple_cnn(BATCH_SIZE)
    elif option == 'conv1':
        model, encoder, generator = make_cnn_1(BATCH_SIZE)
    else:
        print('INVALID OPTION')
        exit(1)


    # checkpoint = ModelCheckpoint(
    #     'experiments/token_VAE/'+ option +'_models/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
    #     save_best_only=True
    # )
    def save_models(epoch, logs):
        val_loss = logs.get('val_loss')
        path = \
            'experiments/token_VAE/{}_models/weights.{epoch:02d}-{val_loss:.2f}'.format(
                option, epoch, val_loss
            )
        model.save(path + '{}.hdf5'.format('model'))
        encoder.save(path + '{}.hdf5'.format('encoder'))
        generator.save(path + '{}.hdf5'.format('generator'))

    save_models = LambdaCallback(on_epoch_end=save_models)

    model.fit_generator(
        generator=train_gen,
        samples_per_epoch=len(train_gen),
        nb_epoch=5,

        validation_data=validation_gen,
        nb_val_samples=len(validation_gen),

        callbacks=[save_models, data_shuffler]
    )

    pass


def make_simple_cnn(batch_size):
    latent_dim = 32
    epsilon_std = 1.0
    intermediate_dim = 512

    x = Input((256, 54))
    f = Flatten()(x)
    h = Dense(intermediate_dim, activation='relu')(f)
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., std=epsilon_std)
        return z_mean + K.exp(z_log_sigma) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(256*54, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    decoded_reshaped = Reshape((256, 54))(x_decoded_mean)

    # end-to-end autoencoder
    vae = Model(x, decoded_reshaped)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, Reshape((256, 54))(_x_decoded_mean))

    # define loss
    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return xent_loss + kl_loss

    vae.compile(optimizer='rmsprop', loss=vae_loss)


    return vae, encoder, generator


def make_basic_cnn(batch_size):
    # taken from https://blog.keras.io/building-autoencoders-in-keras.html
    # to give me a starting model to modify

    batch_size = 32

    input_size = (256, 54, 1)

    latent_dim = 32
    intermediate_dim = 128
    epsilon_std = 1.0
    nb_filters = 64

    x = Input(batch_shape=(batch_size,) + input_size)
    conv_1 = Convolution2D(img_chns, 2, 2, border_mode='same', activation='relu')(x)
    conv_2 = Convolution2D(nb_filters, 2, 2,
                           border_mode='same', activation='relu',
                           subsample=(2, 2))(conv_1)
    conv_3 = Convolution2D(nb_filters, nb_conv, nb_conv,
                           border_mode='same', activation='relu',
                           subsample=(1, 1))(conv_2)
    conv_4 = Convolution2D(nb_filters, nb_conv, nb_conv,
                           border_mode='same', activation='relu',
                           subsample=(1, 1))(conv_3)
    flat = Flatten()(conv_4)
    hidden = Dense(intermediate_dim, activation='relu')(flat)

    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)


    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., std=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_var])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_hid = Dense(intermediate_dim, activation='relu')
    decoder_upsample = Dense(nb_filters * 14 * 14, activation='relu')

    output_shape = (batch_size, 14, 14, nb_filters)

    decoder_reshape = Reshape(output_shape[1:])
    decoder_deconv_1 = Deconvolution2D(nb_filters, nb_conv, nb_conv,
                                       output_shape,
                                       border_mode='same',
                                       subsample=(1, 1),
                                       activation='relu')
    decoder_deconv_2 = Deconvolution2D(nb_filters, nb_conv, nb_conv,
                                       output_shape,
                                       border_mode='same',
                                       subsample=(1, 1),
                                       activation='relu')

    output_shape = (batch_size, 29, 29, nb_filters)
    decoder_deconv_3_upsamp = Deconvolution2D(nb_filters, 2, 2,
                                              output_shape,
                                              border_mode='valid',
                                              subsample=(2, 2),
                                              activation='relu')
    decoder_mean_squash = Convolution2D(img_chns, 2, 2,
                                        border_mode='valid',
                                        activation='sigmoid')

    hid_decoded = decoder_hid(z)
    up_decoded = decoder_upsample(hid_decoded)
    reshape_decoded = decoder_reshape(up_decoded)
    deconv_1_decoded = decoder_deconv_1(reshape_decoded)
    deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
    x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
    x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

    def vae_loss(x, x_decoded_mean):
        # NOTE: binary_crossentropy expects a batch_size by dim
        # for x and x_decoded_mean, so we MUST flatten these!
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = img_rows * img_cols * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    vae = Model(x, x_decoded_mean_squash)
    vae.compile(optimizer='rmsprop', loss=vae_loss)
    vae.summary()


def make_example_uri(ident):
    return '{0}'.format(ident)


def make_ttv(number_of_samples):
    train_size = int(0.9 * number_of_samples)
    validation_size = int(0.1 * number_of_samples)  # NOQA

    train_idxs = range(0, train_size)
    validation_idxs = range(train_size, number_of_samples)

    ttv = {
        'train': [make_example_uri(x) for x in train_idxs],
        'validation': [make_example_uri(x) for x in validation_idxs]
    }
    return ttv


if __name__ == '__main__':
    options = {'simple'}

    args = argv[1:]
    assert len(args) == 1, 'You must provide one argument'
    option = args[0]
    assert option in options, 'options can be {}'.format(options)

    run_experiment(option)
