import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.training import queue_runner
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.dataframe.queues import feeding_queue_runner as fqr
slim = tf.contrib.slim


def build_single_output_queue(batch_generator, output_shape, capacity=10):
    """
    args:
        batch_generator: a function which returns the data to be fed into the queue.
            Must return something of output_shape at every call.

    """
    # The queue takes only one thing at the moment, a batch of images
    types = [dtypes.float32]
    shapes = [output_shape]

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
