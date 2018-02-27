'''
eExperienments on Tensorflow's rnn cells
'''
import tensorflow as tf
import numpy as np

def test_grucell():
    batch_size = 5
    input_size = 10
    hidden_size = 3
    input_data = tf.placeholder(tf.float32, [batch_size, input_size])

    cell = tf.nn.rnn_cell.GRUCell(hidden_size)

    initial_state = cell.zero_state(batch_size, tf.float32)

    hidden_state = initial_state

    output_of_cell, hidden_state = cell(input_data, hidden_state)
    inputs = np.random.randn(batch_size, input_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run((output_of_cell, hidden_state), feed_dict={input_data:inputs}))

test_grucell()
#sample output
'''
2018-02-27 11:04:34.659133: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.2 AVX AVX2 FMA
(array([[-0.322102  ,  0.26052436,  0.03516981],
       [ 0.0235789 ,  0.13357721, -0.01362107],
       [ 0.01783257, -0.3178206 , -0.27473962],
       [-0.05341873,  0.17133303,  0.21480843],
       [-0.03101489, -0.154895  ,  0.40087253]], dtype=float32), array([[-0.322102  ,  0.26052436,  0.03516981],
       [ 0.0235789 ,  0.13357721, -0.01362107],
       [ 0.01783257, -0.3178206 , -0.27473962],
       [-0.05341873,  0.17133303,  0.21480843],
       [-0.03101489, -0.154895  ,  0.40087253]], dtype=float32))
'''