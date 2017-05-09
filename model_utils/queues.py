import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.training import queue_runner
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.dataframe.queues import feeding_queue_runner as fqr
slim = tf.contrib.slim


def build_single_output_queue(
    batch_generator, output_shape, capacity=10, type=dtypes.float32
):
    """
    args:
        batch_generator: a function which returns the data to be fed into the queue.
            Must return something of output_shape at every call.

    """
    # The queue takes only one thing at the moment, a batch of images
    shapes = [output_shape]
    types = [type]

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
            # placeholder: batch_generator()
        }

    runner = fqr.FeedingQueueRunner(
        queue=queue,
        enqueue_ops=enqueue_ops,
        feed_fns=[feed_function]
    )
    queue_runner.add_queue_runner(runner)

    return queue


def build_multiple_output_queue(batch_generator, output_shapes, types, capacity=10):
    """
    where batch_generator returns multiple values
    """
    assert len(output_shapes) == len(types), \
        'lengths of batch_generators, output_shapes, types do not match'
    shapes = output_shapes

    queue = data_flow_ops.FIFOQueue(
          capacity,
          dtypes=types,
          shapes=shapes
      )

    input_name = 'batch'
    placeholders = [
        tf.placeholder(types[i], shape=shapes[i], name=input_name + str(i))
        for i in range(len(output_shapes))
    ]

    enqueue_ops = [queue.enqueue(placeholder) for placeholder in placeholders]
    # enqueue_ops = [queue.enqueue_many(placeholders)]


    def feed_function():
        outputs = batch_generator()
        return {
            input_name + str(i) + ':0': outputs[i]
            for i in range(len(outputs))
        }

    runner = fqr.FeedingQueueRunner(
        queue=queue,
        enqueue_ops=[enqueue_ops],
        feed_fns=[feed_function]
    )
    queue_runner.add_queue_runner(runner)

    return queue
