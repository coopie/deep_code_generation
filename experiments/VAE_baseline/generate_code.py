import numpy as np

from huzzer.tokenizing import TOKEN_MAP

import tensorflow as tf
from tensorflow.python.training.supervisor import Supervisor

from models import build_decoder

NAMES = {
    'encoder_input': 'encoder_input:0',
    'encoder_output': 'encoder_output:0',
    'decoder_input': 'decoder_input:0',
    'decoder_output': 'decoder_output:0',
    'loss': 'loss:0',
    'train_on_batch': 'train_on_batch:0'
}


def generate_code():
    z = tf.placeholder(tf.float32, shape=(1, 16), name='decoder_input')
    build_decoder(z, (256, 54))

    saver = tf.train.Saver()
    with tf.Session() as sess:

        # new_saver = tf.train.import_meta_graph('experiments/VAE_baseline/simple/model.ckpt-2871.meta')
        saver.restore(sess, tf.train.latest_checkpoint('experiments/VAE_baseline/simple'))

        def g(latent_rep=None):
            if latent_rep is None:
                latent_rep = np.random.normal(0, 1, 16)
            return sess.run(
                        NAMES['decoder_output'],
                        feed_dict={
                            NAMES['decoder_input']: np.reshape(latent_rep, (1, -1))
                        }
                    )

        # # latent_rep = np.zeros(16, dtype='float32')
        # latent_rep = np.array([-0.18552721,  0.17082049, 0.1876981, -0.17597072, -0.18672395,
        #  0.20680845, -0.18589398, -0.15539576, -0.17138806, -0.16285755,
        #  0.19973838, -0.20951441, -0.18218575,  0.15672302,  0.32241789,
        #  0.22256292], dtype='float32')
        generated = g()[0]
        tokens = np.argmax(generated, axis=-1)

        def token_to_string(t):
            if t == 0:
                return ''
            return TOKEN_MAP[t-1]

        text = ' '.join([token_to_string(t) for t in tokens])
        print(text)



if __name__ == '__main__':
    with tf.Graph().as_default():
        generate_code()
