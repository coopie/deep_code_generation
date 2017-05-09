import tensorflow as tf
import numpy as np

import project_context  # NOQA
from model_utils.loss_functions import (
    ce_loss_for_sequence_batch,
)


class ModelOpsTest(tf.test.TestCase):

    def test_ce_loss_for_sequence_batch(self):
        """
        Test a batch of sequence length 2, 3, 4
        """
        with self.test_session():
            max_length = 4
            labels_data = np.array([
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            ])
            labels = tf.constant(labels_data, tf.float32)
            # change values of last 2 values on batch[1],
            logits_data = np.array([
                [[9999, 0], [9999, 0], [9999, 0], [9999, 0]],
                [[9999, 0], [9999, 0], [4, 1], [5, 1]],
            ])
            logits = tf.constant(logits_data, tf.float32)
            sequence_lengths = tf.constant(np.array([4, 2]), tf.float32)
            ce = ce_loss_for_sequence_batch(
                logits, labels, sequence_lengths, max_length
            )

            np.testing.assert_almost_equal(
                ce.eval(),
                np.array([0, 0])
            )
