import tensorflow as tf
from tensorflow.python.ops import nn_ops
import numpy as np
def n_matmul():
    '''
    This function to prove that conv2d(h, W, [1,1,1,1], "SAME") can use for
        v = matmul(h, W)
    Where in batch scenarios
        v has shape (batch, v_length, vec_size)
        h has shape (batch, h_length, vec_size)
        W has shape (h_length, v_length)
    After expand W to shape (1, 1, h_length, v_length)
        and expand h to shape (batch_size, v_length, 1, vec_size)
        then call conv2d
    '''
    batch_size = 5
    attn_len = 3
    attn_size = 2
    vec_size = 4

    W = tf.constant(np.random.randn(1, 1, attn_size, vec_size), dtype=tf.float32)
    h = tf.constant(np.random.randn(batch_size, attn_len, 1, attn_size), dtype=tf.float32)
    v = nn_ops.conv2d(h, W, [1, 1, 1, 1], "SAME")

    W_reshape = tf.reshape(W, shape=(attn_size, vec_size))
    h_reshape = tf.reshape(h[-1,:,:,:], shape=(attn_len, attn_size))
    v_reshape = tf.matmul(h_reshape, W_reshape)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result, h_, W_ = sess.run([v,h, W])
        print(result[-1])
        print(result.shape)
        print("h shape:{}".format(h_.shape))
        print("W shape:{}".format(W_.shape))
        r_1 = sess.run(v_reshape)
        print(r_1)
        print(r_1.shape)

if __name__ == "__main__":
    n_matmul()

