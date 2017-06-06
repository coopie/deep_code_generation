import project_context  # NOQA
from model_utils.loss_functions import vae_loss, vae_cross_entropy_loss
from model_utils.ops import vae_resampling

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.training import queue_runner
from tensorflow.python.framework import dtypes
from tensorflow.contrib import layers
from tensorflow.contrib.learn.python.learn.dataframe.queues import feeding_queue_runner as fqr
slim = tf.contrib.slim


# Names for the tensors representing input or output
NAMES = {
    'encoder_input': 'encoder_input:0',
    'encoder_output': 'encoder_output:0',
    'decoder_input': 'decoder_input:0',
    'decoder_output': 'decoder_output:0',
    'loss': 'loss:0',
    'train_on_batch': 'train_on_batch:0'
}


def build_simple_network(x, x_shape, latent_dim=16, epsilon_std=0.01):
    """
    Build a VAE with the encoder and decoder being single fully connected layers.

    Returns: A dict corresponding to useful input and output tensors of the nework

    Arguments:
        * x: the input tensor(placeholder or queue.dequeue op) for the netowork.
    """
    x_flat = slim.flatten(x)

    mus = slim.fully_connected(
        x_flat, latent_dim, scope='encoder_output', activation_fn=tf.tanh
    )
    log_sigmas = slim.fully_connected(
        x_flat, latent_dim, scope='encoder_sigmas', activation_fn=tf.tanh
    )

    def sampling(z_mean, z_log_sigma):
        epsilon = tf.random_normal(
            shape=tf.shape(z_mean),
            mean=0., stddev=epsilon_std
        )
        return z_mean + tf.exp(z_log_sigma) * epsilon

    z_resampled = tf.identity(sampling(mus, log_sigmas), name='decoder_input')

    x_decoded_mean_reshaped = build_decoder(z_resampled, x_shape)

    def vae_loss(x, x_decoded):
        reconstruction_loss = tf.reduce_mean(tf.square(x - x_decoded))
        tf.summary.scalar('reconstruction_loss', reconstruction_loss)

        kl_loss = - 0.5 * tf.reduce_mean(
            1 + log_sigmas - tf.square(mus) - tf.exp(log_sigmas)
        )
        tf.summary.scalar('kl_loss', kl_loss)

        total_loss = tf.add(reconstruction_loss, kl_loss, name='loss')
        tf.summary.scalar('total_loss', total_loss)
        return total_loss

    loss = vae_loss(x, x_decoded_mean_reshaped)

    optimizer = tf.train.AdamOptimizer()

    tf.identity(slim.learning.create_train_op(loss, optimizer), name='train_on_batch')

    return NAMES


def build_simple_network2(x, x_shape, latent_dim, kl_limit=0.1, epsilon_std=0.01):
    """
    TODO: what this is - cross entropy + vae limie
    """
    x_flat = slim.flatten(x)

    mus = slim.fully_connected(
        x_flat, latent_dim, scope='encoder_output', activation_fn=tf.tanh
    )
    log_sigmas = slim.fully_connected(
        x_flat, latent_dim, scope='encoder_sigmas', activation_fn=tf.tanh
    )

    z_resampled = tf.identity(
        vae_resampling(mus, log_sigmas, epsilon_std),
        name='decoder_input'
    )

    x_decoded_mean_reshaped = build_decoder(z_resampled, x_shape, activation=tf.nn.relu6)

    loss = vae_cross_entropy_loss(
        x, x_decoded_mean_reshaped, mus, log_sigmas, kl_limit=kl_limit
    )

    optimizer = tf.train.AdamOptimizer()

    tf.identity(slim.learning.create_train_op(loss, optimizer), name='train_on_batch')

    return NAMES


def conv_arg_scope2():
    return slim.arg_scope(
        [layers.conv2d, layers.conv2d_transpose],
        activation_fn=tf.nn.elu,
        weights_regularizer=slim.l1_regularizer(0.001),
        biases_regularizer=slim.l1_regularizer(0.001)
    )

def conv_arg_scope_final():
    return slim.arg_scope(
        [layers.conv2d, layers.conv2d_transpose],
        activation_fn=tf.nn.relu,
    )

def conv_arg_scope():
    return slim.arg_scope(
        [layers.conv2d, layers.conv2d_transpose],
        activation_fn=tf.nn.elu,
        normalizer_fn=layers.batch_norm,
        normalizer_params={'scale': True}
    )


def build_conv1(x, x_shape, latent_dim=32, epsilon_std=0.01):
    with conv_arg_scope():
        z_mus, z_log_sigmas = build_conv1_encoder(x, latent_dim)

        z_resampled = build_resampling(z_mus, z_log_sigmas, epsilon_std)

        x_decoded_mean_reshaped = build_conv1_decoder(z_resampled, x_shape)
        loss = vae_loss(x, x_decoded_mean_reshaped, z_mus, z_log_sigmas)
        optimizer = tf.train.AdamOptimizer()
        tf.identity(slim.learning.create_train_op(loss, optimizer), name='train_on_batch')
    return NAMES


def build_conv1_encoder(x, latent_dim):
    print('building encoder')
    x_conv = tf.expand_dims(x, -1)
    print('input: {}'.format(x_conv.get_shape()))

    net = layers.conv2d(x_conv, 32, 5, stride=(2, 1))
    print('conv: {}'.format(net.get_shape()))

    net = layers.conv2d(net, 32, 5, stride=(2, 1))
    print('conv: {}'.format(net.get_shape()))

    net = layers.conv2d(net, 64, 5, stride=(2, 3))
    print('conv: {}'.format(net.get_shape()))

    net = layers.conv2d(net, 64, 5)
    print('conv: {}'.format(net.get_shape()))
    # net = layers.conv2d(net, 128, 5, stride=2, padding='VALID')
    net = layers.conv2d(net, 64, 5, stride=(2, 3))
    print('conv: {}'.format(net.get_shape()))
    # net = layers.dropout(net, keep_prob=0.9)
    # print('conv: {}'.format(net.get_shape()))
    net = layers.conv2d(net, 64, 5)
    print('conv: {}'.format(net.get_shape()))

    net = layers.conv2d(net, 128, 3, stride=(2, 1), padding='VALID')
    print('conv: {}'.format(net.get_shape()))

    net = layers.conv2d(net, 128, 3, stride=(2, 1), padding='VALID')
    print('conv: {}'.format(net.get_shape()))

    mus = layers.conv2d(net, latent_dim, (1, 2), padding='VALID', activation_fn=tf.tanh)
    mus = layers.flatten(mus)
    print('mus: {}'.format(mus.get_shape()))

    sigs = layers.conv2d(net, latent_dim, (1, 2), padding='VALID', activation_fn=tf.tanh)
    sigs = layers.flatten(sigs)
    print('sigs: {}'.format(sigs.get_shape()))

    mus = tf.identity(mus, 'encoder_output')
    sigs = tf.identity(sigs, 'encoder_sigmas')
    return mus, sigs


def build_conv1_decoder(z_resampled, x_shape):
    print('decoder_structure')
    net = tf.expand_dims(z_resampled, 1)
    net = tf.expand_dims(net, 1)
    print('expanded latent rep: {}'.format(net.get_shape()))

    net = layers.conv2d_transpose(net, 128, (2, 2), padding='VALID')
    print('deconv: {}'.format(net.get_shape()))

    net = layers.conv2d_transpose(net, 128, (3, 2), padding='VALID')
    print('deconv: {}'.format(net.get_shape()))

    net = layers.conv2d_transpose(net, 64, 5, stride=2)
    print('deconv:{}'.format(net.get_shape()))

    net = layers.conv2d_transpose(net, 64, 5)
    print('deconv:{}'.format(net.get_shape()))

    net = layers.conv2d_transpose(net, 64, 5, stride=(2, 3))
    print('deconv:{}'.format(net.get_shape()))

    net = layers.conv2d_transpose(net, 64, 5)
    print('deconv:{}'.format(net.get_shape()))

    net = layers.conv2d_transpose(net, 64, 5, stride=(2, 3))
    print('deconv:{}'.format(net.get_shape()))

    net = layers.conv2d_transpose(net, 54, 5, stride=(2, 1))
    print('deconv:{}'.format(net.get_shape()))

    net = layers.conv2d_transpose(net, 54, 5, stride=(2, 1))
    print('deconv:{}'.format(net.get_shape()))

    net = layers.conv2d_transpose(net, 32, 3)
    print('deconv:{}'.format(net.get_shape()))

    x_decoded_mean = layers.conv2d_transpose(
        net, 1, 1, activation_fn=tf.nn.sigmoid
    )

    x_decoded_mean_reshaped = tf.reshape(
        x_decoded_mean, [-1, *x_shape], name='decoder_output'
    )
    return x_decoded_mean_reshaped


def build_conv2(x, x_shape, latent_dim, epsilon_std=0.01):
    """
    Essentially builds the same model as conv1 with a few exceptions:
        * the latent dim is higher
        * the output is softmaxed over the x axis - each token output is softmaxed
        * the KL loss is also limited to a small value - 0.025. taken from looking at
            the loss of the conv1 experiments
    """
    with conv_arg_scope():
        z_mus, z_log_sigmas = build_conv1_encoder(x, latent_dim)

        z_resampled = build_resampling(z_mus, z_log_sigmas, epsilon_std)

        x_decoded_mean_reshaped = build_conv1_decoder(z_resampled, x_shape)
        print('output_shape: {}'.format(x_decoded_mean_reshaped.get_shape()))
        x_decoded_mean_reshaped_softmaxed = tf.nn.softmax(x_decoded_mean_reshaped, dim=-1)
        loss = vae_loss(x, x_decoded_mean_reshaped_softmaxed, z_mus, z_log_sigmas, kl_limit=0.025)
        optimizer = tf.train.AdamOptimizer()
        tf.identity(slim.learning.create_train_op(loss, optimizer), name='train_on_batch')
    return NAMES


def build_conv3(x, x_shape, latent_dim=64, epsilon_std=0.01):
    """
    Essentially builds the same model as conv2 with a few exceptions:
        * the latent dim is higher
        * the KL loss is also limited to a smaller value - 0.0025. taken from looking at
            the loss of the conv2 experiments
    """
    with conv_arg_scope():
        z_mus, z_log_sigmas = build_conv1_encoder(x, latent_dim)

        z_resampled = build_resampling(z_mus, z_log_sigmas, epsilon_std)

        x_decoded_mean_reshaped = build_conv1_decoder(z_resampled, x_shape)
        x_decoded_mean_reshaped_softmaxed = tf.nn.softmax(x_decoded_mean_reshaped, dim=-1)
        print('output_shape: {}'.format(x_decoded_mean_reshaped_softmaxed.get_shape()))
        loss = vae_loss(x, x_decoded_mean_reshaped_softmaxed, z_mus, z_log_sigmas, kl_limit=0.0025)
        optimizer = tf.train.AdamOptimizer()
        tf.identity(slim.learning.create_train_op(loss, optimizer), name='train_on_batch')
    return NAMES


def build_conv4(x, x_shape, latent_dim=64, epsilon_std=0.01):
    """
    Essentially builds the same model as conv3 with a few exceptions:
        * the KL loss is also limited to a larger value - 0.01. taken from looking at
            the loss of the conv3 experiments
    """
    with conv_arg_scope():
        z_mus, z_log_sigmas = build_conv1_encoder(x, latent_dim)

        z_resampled = build_resampling(z_mus, z_log_sigmas, epsilon_std)

        x_decoded_mean_reshaped = build_conv1_decoder(z_resampled, x_shape)
        x_decoded_mean_reshaped_softmaxed = tf.nn.softmax(x_decoded_mean_reshaped, dim=-1)
        print('output_shape: {}'.format(x_decoded_mean_reshaped_softmaxed.get_shape()))
        loss = vae_loss(x, x_decoded_mean_reshaped_softmaxed, z_mus, z_log_sigmas, kl_limit=0.01)
        optimizer = tf.train.AdamOptimizer()
        tf.identity(slim.learning.create_train_op(loss, optimizer), name='train_on_batch')
    return NAMES


def build_special_conv(x, x_shape, latent_dim, epsilon_std=0.01):
    """
    TODO: explain what makes this so special.
    """
    with conv_arg_scope():
        z_mus, z_log_sigmas = build_special_conv_encoder(x, latent_dim)

        z_resampled = build_resampling(z_mus, z_log_sigmas, epsilon_std)

        x_decoded_mean = build_special_conv_decoder(z_resampled, x_shape)
        loss = vae_cross_entropy_loss(x, x_decoded_mean, z_mus, z_log_sigmas, kl_limit=0.1)
        optimizer = tf.train.AdamOptimizer()
        tf.identity(slim.learning.create_train_op(loss, optimizer), name='train_on_batch')
    return NAMES


def build_special_conv_low_kl(x, x_shape, latent_dim, epsilon_std=0.01):
    with conv_arg_scope():
        z_mus, z_log_sigmas = build_special_conv_encoder(x, latent_dim)

        z_resampled = build_resampling(z_mus, z_log_sigmas, epsilon_std)

        x_decoded_mean = build_special_conv_decoder(z_resampled, x_shape)
        loss = vae_cross_entropy_loss(
            x, x_decoded_mean, z_mus, z_log_sigmas,
            kl_limit=0.1,
            kl_scale=0.01
        )
        optimizer = tf.train.AdamOptimizer()
        tf.identity(slim.learning.create_train_op(loss, optimizer), name='train_on_batch')
    return NAMES


def build_special_conv_encoder(x, latent_dim):
    print('building encoder')
    x_conv = tf.expand_dims(x, -1)
    print('input: {}'.format(x_conv.get_shape()))

    net = layers.conv2d(x_conv, 64, (1, 54), padding='VALID')
    print('conv: {}'.format(net.get_shape()))

    net = layers.conv2d(net, 64, (6, 1), stride=4)
    print('conv: {}'.format(net.get_shape()))

    net = layers.conv2d(net, 128, (6, 1), stride=4)
    print('conv: {}'.format(net.get_shape()))

    net = layers.conv2d(net, 128, (4, 1), stride=2)
    print('conv: {}'.format(net.get_shape()))

    z_mus = layers.flatten(
        layers.conv2d(net, latent_dim, (4, 1), stride=2, padding='VALID', activation_fn=tf.tanh)
    )

    z_log_sigmas = layers.flatten(
        layers.conv2d(net, latent_dim, (4, 1), stride=2, padding='VALID', activation_fn=tf.tanh)
    )
    return z_mus, z_log_sigmas


def build_special_conv_decoder(z_resampled, x_shape):
    print('decoder_structure')
    net = tf.expand_dims(z_resampled, 1)
    net = tf.expand_dims(net, 1)
    print('expanded latent rep: {}'.format(net.get_shape()))

    net = layers.conv2d_transpose(net, 128, (4, 1), padding='VALID')
    print('deconv: {}'.format(net.get_shape()))

    net = layers.conv2d_transpose(net, 128, (4, 1), stride=(2, 1))
    print('deconv: {}'.format(net.get_shape()))

    net = layers.conv2d_transpose(net, 64, (6, 1), stride=(4, 1))
    print('deconv: {}'.format(net.get_shape()))

    net = layers.conv2d_transpose(net, 64, (6, 1), stride=(4, 1))
    print('deconv: {}'.format(net.get_shape()))

    net = layers.conv2d_transpose(net, 1, (1, 54), stride=(1, 1), padding='VALID', activation_fn=tf.nn.relu6)
    print('deconv: {}'.format(net.get_shape()))

    x_decoded_mean = tf.squeeze(net, -1)
    print('final shape: {}'.format(x_decoded_mean.get_shape()))

    return x_decoded_mean


def build_special_conv2(x, x_shape, latent_dim, epsilon_std=0.01):
    """
    TODO: what this is - (one conv then fully connected)
    """

    num_filters = 64
    with conv_arg_scope():
        z_mus, z_log_sigmas = build_special_conv2_encoder(x, latent_dim, num_filters)
        z_resampled = build_resampling(z_mus, z_log_sigmas, epsilon_std)

        x_decoded_mean = build_special_conv2_decoder(z_resampled, x_shape, num_filters)
        loss = vae_cross_entropy_loss(x, x_decoded_mean, z_mus, z_log_sigmas, kl_limit=0.1)
        optimizer = tf.train.AdamOptimizer()
        tf.identity(slim.learning.create_train_op(loss, optimizer), name='train_on_batch')
    return NAMES


def build_special_conv2_l1(
    x, x_shape, latent_dim, filter_length=1, num_filters=64, epsilon_std=0.01
):
    with conv_arg_scope2():
        z_mus, z_log_sigmas = build_special_conv2_encoder(
            x, latent_dim, num_filters, filter_length
        )
        z_resampled = build_resampling(z_mus, z_log_sigmas, epsilon_std)

        x_decoded_mean = build_special_conv2_decoder(
            z_resampled, x_shape, num_filters, filter_length
        )
        loss = vae_cross_entropy_loss(x, x_decoded_mean, z_mus, z_log_sigmas, kl_limit=0.1)
        optimizer = tf.train.AdamOptimizer()
        tf.identity(slim.learning.create_train_op(loss, optimizer), name='train_on_batch')
    return NAMES


def build_special_conv2_encoder(x, latent_dim, num_filters, filter_length=1):
    print('building encoder')
    x_conv = tf.expand_dims(x, -1)
    print('input: {}'.format(x_conv.get_shape()))

    net = layers.conv2d(x_conv, num_filters, (filter_length, 54), padding='VALID')
    print('conv: {}'.format(net.get_shape()))

    net = layers.flatten(net)
    print('flatten: {}'.format(net.get_shape()))

    mus = slim.fully_connected(
        net, latent_dim, scope='encoder_output', activation_fn=tf.tanh
    )
    log_sigmas = slim.fully_connected(
        net, latent_dim, scope='encoder_sigmas', activation_fn=tf.tanh
    )
    return mus, log_sigmas


def build_special_conv2_decoder(z_resampled, x_shape, num_filters, filter_length=1):
    print('decoder_structure')

    net = slim.fully_connected(
        z_resampled, num_filters * 128, activation_fn=tf.nn.relu
    )
    print('fully_connected rep: {}'.format(net.get_shape()))
    net = tf.reshape(net, (-1, 128, 1, num_filters))
    print('reshaped rep: {}'.format(net.get_shape()))

    net = layers.conv2d_transpose(
        net, 1, (filter_length, 54),
        stride=(1, 1), padding='VALID', activation_fn=tf.nn.relu6
    )
    print('deconv: {}'.format(net.get_shape()))

    feature_map_width = net.get_shape()[1].value
    print('feature_map_width: {}'.format(feature_map_width))
    if feature_map_width != 128:
        difference = feature_map_width - 128
        net = net[:, (difference // 2):-(difference // 2 + (difference % 2)), :, :]
        print('trimmed dimensions: {}'.format(net.get_shape()))

    x_decoded_mean = tf.squeeze(net, -1)
    print('final shape: {}'.format(x_decoded_mean.get_shape()))
    return x_decoded_mean


def build_special_conv4_l1(
    x, x_shape, latent_dim, filter_length=1, num_filters=64, epsilon_std=0.01
):
    with conv_arg_scope2():
        z_mus, z_log_sigmas, dense_layer_size = build_special_conv4_encoder(
            x, latent_dim, num_filters, filter_length
        )
        z_resampled = build_resampling(z_mus, z_log_sigmas, epsilon_std)

        x_decoded_mean = build_special_conv4_decoder(
            z_resampled, x_shape, num_filters, filter_length, dense_layer_size
        )
        loss = vae_cross_entropy_loss(x, x_decoded_mean, z_mus, z_log_sigmas, kl_limit=0.1)
        optimizer = tf.train.AdamOptimizer()
        tf.identity(slim.learning.create_train_op(loss, optimizer), name='train_on_batch')
    return NAMES


def build_special_conv4_final(
    x, x_shape, latent_dim, filter_length=1, num_filters=64, epsilon_std=0.01
):
    with conv_arg_scope_final():
        z_mus, z_log_sigmas, dense_layer_size = build_special_conv4_encoder(
            x, latent_dim, num_filters, filter_length
        )
        z_resampled = build_resampling(z_mus, z_log_sigmas, epsilon_std)

        x_decoded_mean = build_special_conv4_decoder(
            z_resampled, x_shape, num_filters, filter_length, dense_layer_size
        )
        loss = vae_cross_entropy_loss(x, x_decoded_mean, z_mus, z_log_sigmas)
        optimizer = tf.train.AdamOptimizer()
        tf.identity(slim.learning.create_train_op(loss, optimizer), name='train_on_batch')
    return NAMES


def build_special_conv4_encoder(x, latent_dim, num_filters, filter_length=1):
    print('building encoder')
    x_conv = tf.expand_dims(x, -1)
    print('input: {}'.format(x_conv.get_shape()))

    net = layers.conv2d(x_conv, num_filters, (filter_length, 54), padding='VALID')
    print('conv: {}'.format(net.get_shape()))

    net = layers.conv2d(net, num_filters, (5, 1), padding='VALID')
    print('conv: {}'.format(net.get_shape()))

    net = layers.flatten(net)
    print('flatten: {}'.format(net.get_shape()))
    dense_layer_size = net.get_shape()[1].value

    mus = slim.fully_connected(
        net, latent_dim, scope='encoder_output', activation_fn=tf.tanh
    )
    log_sigmas = slim.fully_connected(
        net, latent_dim, scope='encoder_sigmas', activation_fn=tf.tanh
    )
    return mus, log_sigmas, dense_layer_size


def build_special_conv4_decoder(z_resampled, x_shape, num_filters, filter_length=1, dense_layer_size=None):
    sequence_length = x_shape[0]

    print('decoder_structure')
    if dense_layer_size is None:
        raise RuntimeError('dense_layer_size is not initialised.')

    net = slim.fully_connected(
        z_resampled, dense_layer_size, activation_fn=tf.nn.relu
    )
    print('fully_connected rep: {}'.format(net.get_shape()))
    net = tf.reshape(net, (-1, dense_layer_size // num_filters, 1, num_filters))
    print('reshaped rep: {}'.format(net.get_shape()))

    net = layers.conv2d_transpose(
        net, num_filters, (5, 1), padding='VALID'
    )
    print('deconv: {}'.format(net.get_shape()))

    net = layers.conv2d_transpose(
        net, 1, (filter_length, 54),
        stride=(1, 1), padding='VALID',
    )
    print('deconv: {}'.format(net.get_shape()))

    feature_map_width = net.get_shape()[1].value
    print('feature_map_width: {}'.format(feature_map_width))
    if feature_map_width != sequence_length:
        # difference = feature_map_width - sequence_length
        # net = net[:, (difference // 2):-(difference // 2 + (difference % 2)), :, :]
        net = net[:, :sequence_length, :, :]
        print('trimmed dimensions: {}'.format(net.get_shape()))

    x_decoded_mean = tf.squeeze(net, -1)
    print('final shape: {}'.format(x_decoded_mean.get_shape()))
    return x_decoded_mean


def build_resampling(z_mus, z_log_sigmas, epsilon_std):
    def sampling(z_mean, z_log_sigma):
        epsilon = tf.random_normal(
            shape=tf.shape(z_mean),
            mean=0., stddev=epsilon_std
        )
        return z_mean + tf.exp(z_log_sigma) * epsilon

    z_resampled = tf.identity(sampling(z_mus, z_log_sigmas), name='decoder_input')
    return z_resampled


def build_decoder(z_resampled, x_shape, activation=tf.sigmoid):
    x_decoded_mean = slim.fully_connected(
        z_resampled,
        x_shape[0] * x_shape[1],
        scope='x_decoded_mean',
        activation_fn=activation
    )
    x_decoded_mean_reshaped = tf.reshape(
        x_decoded_mean, [-1, *x_shape], name='decoder_output'
    )
    return x_decoded_mean_reshaped


def build_queue(batch_generator, batch_size, example_shape, capacity=10):
    """
    TODO: refactor this out when it's working well
    args:
        batch_generator: a function which takes args `placeholder` which is the Placeholder to feed data in to.

    """
    # The queue takes only one thing at the moment, a batch of images
    types = [dtypes.float32]
    shapes = [(batch_size, *example_shape)]

    queue = data_flow_ops.FIFOQueue(
          capacity,
          dtypes=types,
          shapes=shapes
      )

    input_name = 'batch'
    placeholder = tf.placeholder(types[0], shape=shapes[0], name=input_name)
    enqueue_ops = [queue.enqueue(placeholder)]

    def feed_function():
        return {
            input_name + ':0': batch_generator()
        }

    runner = fqr.FeedingQueueRunner(
        queue=queue,
        enqueue_ops=enqueue_ops,
        feed_fns=[feed_function]
    )
    queue_runner.add_queue_runner(runner)

    return queue
