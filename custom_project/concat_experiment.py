'''
Experiments for tf.concat
'''
import tensorflow as tf
def concat_1d():
    print("concat 1d experiment")
    a = tf.constant([1,2])
    b = tf.constant([2,3,5])
    c = tf.concat([a,b], axis=0) #need use axis=0
    with tf.Session() as sess:
        np_a, np_b, np_c = sess.run([a,b,c])
        print("input \t\na {}\nb {}\noutput c {}".format(np_a, np_b, np_c))

def concat_2d():
    print("concat 2d experiment")
    a = tf.constant([[1,2],[3,4]])
    b = tf.constant([[3,4,5], [5,6,7]])
    c = tf.concat([a,b], axis=1) #need use axis = 1
    with tf.Session() as sess:
        np_a, np_b, np_c = sess.run([a,b,c])
        print("input \t\na {}\nb {}\noutput c {}".format(np_a, np_b, np_c))

if __name__=="__main__":
    concat_1d()
    concat_2d()