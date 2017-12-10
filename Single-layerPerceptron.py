from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score

def main():
    data=input_data.read_data_sets('Single_layer_Perceptron/')
    x=tf.placeholder('float',shape=[None,784])
    y=tf.placeholder('float',shape=[None,1])
    w=tf.Variable(tf.ones([784,1]))
    b=tf.Variable([0.])
    logits=tf.nn.xw_plus_b(x,w,b)
    learning_rate=0.1

    loss=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
    train_op=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    predict=tf.nn.softmax(logits)

    xtest, ytest = data.test.next_batch(1000)
    TP = tf.equal(ytest,predict)
    accuracy = tf.reduce_mean(tf.cast(TP, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            xb,yb=data.train.next_batch(200)
            sess.run(train_op,feed_dict={x:xb,y:np.reshape(yb,[-1,1])})


        accuracyResult = sess.run( accuracy, feed_dict={x: xtest, y: np.reshape(ytest,[-1,1])})
        print('accuracy:%f'%accuracyResult)

if __name__=='__main__':
    main()