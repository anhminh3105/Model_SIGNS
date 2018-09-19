from resnets_model import *
import tensorflow as tf

def main():
    tf.reset_default_graph()

    with tf.Session() as sess:
        np.random.seed(1)
        A_prev = tf.placeholder("float", shape=(3, 4, 4, 6))
        X = np.random.randn(3, 4, 4, 6)
        A = identity_block(A_prev, 2, [2, 4, 6], 1, 'a') # test identity_block()
        #A = conv_block(A_prev, 2, [2, 4, 6], 1, 'a')
        sess.run(tf.global_variables_initializer())
        out = A.eval(feed_dict={A_prev: X, K.learning_phase(): 0})
        print("out=\n", str(out))
    
if __name__ == '__main__':
    main()
    