'''
This is an example model to use tf.layers.dense to group vectors
Data have two group of vectors.
    1) all vectors have same size, say 3
    2) group A vectors have larger values on one index, say 1
    3) group B vectors have larger values on another index, say 2
    4) The other index are relative small random data
    Examples:
        [100, 0.1, 0.1] in group A
        [0.1, 100, 0.1] in group B

Task, given two vectors, your model could tell if they are in same group
'''

import tensorflow as tf
import numpy as np
def generate_data(counts, size=3):
    ret = []

    def get_a_b():
        large_value = 100.0
        a_index = 0
        b_index = 1
        np_a, np_b = np.ones(shape=size), np.ones(shape=size)
        np_a[a_index] = large_value
        np_b[b_index] = large_value
        return np.random.randn(size) * np_a, np.random.randn(size) * np_b

    for _ in range(counts):
        a1, b1 = get_a_b()
        a2, b2 = get_a_b()
        ret.append((a1, a2, 1))
        ret.append((b1, b2, 1))
        ret.append((a1, b1, 0))
        ret.append((a2, b2, 0))

    return ret

def vector_compare_concat(v_a, v_b):
    '''
    return the compare vector which defined by z(v_a,v_b) = tf.concat(v_a, v_b, v_a - v_b, v_a * v_b)
    Args:
        v_a: the left vector, shape [batch_size, vector_size]
        v_b: the right vector, shape [batch_size, vector_size]

    Returns:
        z_a_b: the compare vector, shape [batch_size, vector_size]
    '''
    z_a_b = tf.concat([v_a, v_b, v_a - v_b, v_a * v_b], axis=1)
    return z_a_b

def divide_data(inputs):
    '''
    divide data to train, validation and test sets
    Args:
        inputs: list of vector pairs

    Returns: dictionary of lists for "train", "val" and "test". And 90%, 5%, 5% for each
    '''
    ret = {}
    np.random.shuffle(inputs)
    n = len(inputs)
    ret["train"] = inputs[:int(0.9 * n)]
    ret["val"]   = inputs[int(0.9 * n):int(0.95 * n)]
    ret["test"]  = inputs[int(0.95 * n):]

    return ret

if __name__ == "__main__":
    l = generate_data(100)
    dict_l = divide_data(l)
    print(len(dict_l["train"]),len(dict_l["val"]), len(dict_l["test"]))
    a, b, c = zip(*dict_l["test"])
    tensor_a = tf.placeholder(dtype=tf.float32)
    tensor_b = tf.placeholder(dtype=tf.float32)
    z = vector_compare_concat(tensor_a, tensor_b)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _z = sess.run(z, feed_dict={tensor_a: a, tensor_b: b})
        print(_z.shape) # output shape [batch_size, 4 * vector_size]
