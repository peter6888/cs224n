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

    W_original = tf.constant(np.random.randn(attn_size, vec_size), dtype=tf.float32)
    h_original = tf.constant(np.random.randn(batch_size, attn_len, attn_size), dtype=tf.float32)
    W = tf.expand_dims(tf.expand_dims(W_original, 0), 0)
    h = tf.expand_dims(h_original, 2)
    v = nn_ops.conv2d(h, W, [1, 1, 1, 1], "SAME")

    W_reshape = tf.reshape(W, shape=(attn_size, vec_size))
    h_reshape = tf.reshape(h[-1,:,:,:], shape=(attn_len, attn_size))
    v_reshape = tf.matmul(h_reshape, W_reshape)
    
    h_dot_W = tf.einsum('ijk,kl->ijl', h_original, W_original)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result, h_, W_ = sess.run([v,h, W])
        print(result[-1])
        print(result.shape)
        print("h shape:{}".format(h_.shape))
        print("W shape:{}".format(W_.shape))
        r_1, r_2 = sess.run([v_reshape, h_dot_W])
        print(r_1)
        print(r_1.shape)
        print("tf.einsum('ijk,kl->ijl')")
        print(r_2)


if __name__ == "__main__":
    n_matmul()

