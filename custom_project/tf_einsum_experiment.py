import tensorflow as tf
from tensorflow.python.ops import math_ops
def rank3_mul():
    '''
    Test tf.einsum with rank 3 Tensor
    '''
    batch_size = 2
    seq_length = 3
    hidden_vector_size = 4

    rank2state = tf.random_normal(shape=[batch_size, hidden_vector_size])
    rank3states = tf.random_normal(shape=[batch_size, seq_length, hidden_vector_size])
    ''' Error information for below line
Traceback (most recent call last):
  File "tf_einsum_experiment.py", line 20, in <module>
    rank3_mul()
  File "tf_einsum_experiment.py", line 12, in rank3_mul
    result = tf.einsum("bi,bTi->bi", rank2state, rank3states)
  File "/Users/peli/anaconda3/lib/python3.5/site-packages/tensorflow/python/ops/special_math_ops.py", line 210, in einsum
    axes_to_sum)
  File "/Users/peli/anaconda3/lib/python3.5/site-packages/tensorflow/python/ops/special_math_ops.py", line 256, in _einsum_reduction
    raise ValueError()
ValueError
    '''
    #result = tf.einsum("bi,bTi->bT", rank2state, rank3states)
    '''
    math_ops.reduce_sum(decoder_state * encoder_states_dot_W, [2, 3])
    '''
    rank2state = tf.expand_dims(rank2state, axis=1)
    result = math_ops.reduce_sum(rank2state * rank3states, axis=-1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _r2, _r3, _result = sess.run([rank2state, rank3states, result])
        print("r2")
        print(_r2)
        print("r3")
        print(_r3)
        print("result")
        print(_result)

if __name__ == "__main__":
    rank3_mul()