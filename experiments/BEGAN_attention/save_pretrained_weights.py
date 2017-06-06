import tensorflow as tf
from os.path import join

from models import build_single_program_encoder, build_attention1_decoder
build_attention1_decoder

path_to_model = 'experiments/RVAE_attention/basic_attention1_128'

z_size = 128
max_len = 4
input_sequences = tf.zeros((1, 4, 54))
sequence_lengths = tf.zeros((1), tf.int32)
encoder_output = build_single_program_encoder(input_sequences, sequence_lengths, z_size)
decoder_output = build_attention1_decoder(
    encoder_output[:, :z_size], sequence_lengths, max_len, 54
)

with tf.variable_scope('', reuse=True):
    discriminator_var_list = {

        'discriminator/decoder_fully_connected/bias': tf.get_variable('decoder_fully_connected/bias', [54]),
        'discriminator/decoder_fully_connected/weights': tf.get_variable('decoder_fully_connected/weights', [128, 54]),
        'discriminator/decoder_rnn/lstm_cell/biases': tf.get_variable('decoder_rnn/lstm_cell/biases', [512]),
        'discriminator/decoder_rnn/lstm_cell/weights': tf.get_variable('decoder_rnn/lstm_cell/weights', [256,512]),
        'discriminator/rnn/lstm_cell/biases': tf.get_variable('rnn/lstm_cell/biases', [1024]),
        'discriminator/rnn/lstm_cell/weights': tf.get_variable('rnn/lstm_cell/weights', [310,1024]),
        'discriminator/simple_attention/bias': tf.get_variable('simple_attention/bias', [128]),
        'discriminator/simple_attention/weights': tf.get_variable('simple_attention/weights', [128,128]),

    }
    generator_var_list = {

        'generator/decoder_fully_connected/bias': tf.get_variable('decoder_fully_connected/bias', [54]),
        'generator/decoder_fully_connected/weights': tf.get_variable('decoder_fully_connected/weights', [128, 54]),
        'generator/decoder_rnn/lstm_cell/biases': tf.get_variable('decoder_rnn/lstm_cell/biases', [512]),
        'generator/decoder_rnn/lstm_cell/weights': tf.get_variable('decoder_rnn/lstm_cell/weights', [256,512]),
        'generator/rnn/lstm_cell/biases': tf.get_variable('rnn/lstm_cell/biases', [1024]),
        'generator/rnn/lstm_cell/weights': tf.get_variable('rnn/lstm_cell/weights', [310,1024]),
        'generator/simple_attention/bias': tf.get_variable('simple_attention/bias', [128]),
        'generator/simple_attention/weights': tf.get_variable('simple_attention/weights', [128,128]),

    }


with tf.Session() as sess:
    print('restoring vars')
    reloader = tf.train.Saver()
    reloader.restore(sess, join(path_to_model, 'model.ckpt-61802'))

    generator_saver = tf.train.Saver(
        var_list=generator_var_list
    )
    generator_saver.save(sess, 'experiments/BEGAN_attention/generator_weights.cpkt')
    discriminator_saver = tf.train.Saver(
        var_list=discriminator_var_list
    )
    discriminator_saver.save(sess, 'experiments/BEGAN_attention/discriminator_weights.cpkt')


# dump from inspect_checkpoint
# beta1_power (DT_FLOAT) []
# beta2_power (DT_FLOAT) []
# decoder_fully_connected/bias (DT_FLOAT) [54]
# decoder_fully_connected/bias/Adam (DT_FLOAT) [54]
# decoder_fully_connected/bias/Adam_1 (DT_FLOAT) [54]
# decoder_fully_connected/weights (DT_FLOAT) [128,54]
# decoder_fully_connected/weights/Adam (DT_FLOAT) [128,54]
# decoder_fully_connected/weights/Adam_1 (DT_FLOAT) [128,54]
# decoder_rnn/lstm_cell/biases (DT_FLOAT) [512]
# decoder_rnn/lstm_cell/biases/Adam (DT_FLOAT) [512]
# decoder_rnn/lstm_cell/biases/Adam_1 (DT_FLOAT) [512]
# decoder_rnn/lstm_cell/weights (DT_FLOAT) [256,512]
# decoder_rnn/lstm_cell/weights/Adam (DT_FLOAT) [256,512]
# decoder_rnn/lstm_cell/weights/Adam_1 (DT_FLOAT) [256,512]
# global_step (DT_INT64) []
# rnn/lstm_cell/biases (DT_FLOAT) [1024]
# rnn/lstm_cell/biases/Adam (DT_FLOAT) [1024]
# rnn/lstm_cell/biases/Adam_1 (DT_FLOAT) [1024]
# rnn/lstm_cell/weights (DT_FLOAT) [310,1024]
# rnn/lstm_cell/weights/Adam (DT_FLOAT) [310,1024]
# rnn/lstm_cell/weights/Adam_1 (DT_FLOAT) [310,1024]
# simple_attention/bias (DT_FLOAT) [128]
# simple_attention/bias/Adam (DT_FLOAT) [128]
# simple_attention/bias/Adam_1 (DT_FLOAT) [128]
# simple_attention/weights (DT_FLOAT) [128,128]
# simple_attention/weights/Adam (DT_FLOAT) [128,128]
# simple_attention/weights/Adam_1 (DT_FLOAT) [128,128]
