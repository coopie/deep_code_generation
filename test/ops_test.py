import tensorflow as tf
import numpy as np

import project_context  # NOQA
from model_utils.ops import (
    resampling,
    get_sequence_lengths,
    compute_attention_vector,
)


class ModelOpsTest(tf.test.TestCase):

    def test_get_sequence_lengths(self):
        """
        Test a batch of sequence length 2, 3, 4
        """
        with self.test_session():
            data = np.array([
                [[0, 1], [1, 0], [1, 0], [1, 0]],
                [[0, 1], [0, 1], [1, 0], [1, 0]],
                [[0, 1], [0, 1], [0, 1], [1, 0]],
            ])
            example_batch = tf.constant(data, tf.float32)
            lengths = get_sequence_lengths(example_batch)
            self.assertAllEqual(lengths.eval(), [2, 3, 4])

    def test_compute_attention_vector(self):
        """
        Test a batch of coefficients with a batch of vectors
        """
        with self.test_session():
            desired_coefs = np.array([
                [0.2, 0.1, 0.7],
                [0.1, 0.6, 0.3]
            ])
            pre_softmaxed = np.log(desired_coefs)
            batch_coefs = tf.constant(
                pre_softmaxed,
                dtype=tf.float32
            )
            vectors = tf.constant(
                np.array([
                    [
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                    ],
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]
                    ]
                ]),
                dtype=tf.float32
            )
            attention_vectors = compute_attention_vector(vectors, batch_coefs)
            np.testing.assert_almost_equal(
                attention_vectors.eval(),
                np.array([
                    [0,   0,   1],
                    [0.1, 0.6, 0.3]
                ])
            )
