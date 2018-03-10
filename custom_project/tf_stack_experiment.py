import tensorflow as tf
import numpy as np

def tfstack_from_tensorlist():
    batch_size = 16
    decoder_hidden_size = 512
    encoder_hidden_size = 256
    decoder_t = 6
    vsize = 50000

    tensorlist = []
    attn_score1 = np.random.randn(batch_size, decoder_t)
    attn_score1 = tf.convert_to_tensor(attn_score1, np.float32)
    attn_score2 = np.random.randn(batch_size, decoder_t)
    attn_score2 = tf.convert_to_tensor(attn_score2, np.float32)
    tensorlist.append(attn_score1)
    tensorlist.append(attn_score2)

    print(attn_score1.get_shape())
    print(tf.stack(tensorlist).get_shape())

    return

if __name__=="__main__":
    tfstack_from_tensorlist()

