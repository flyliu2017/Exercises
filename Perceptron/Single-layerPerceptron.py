from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def main():
    data=input_data.read_data_sets('Single_layer_Perceptron/')
    x=tf.placeholder('float',shape=[None,784])
    y=tf.placeholder('int64')
    w=tf.Variable(tf.ones([784,10]))
    b=tf.Variable(tf.ones([10]))
    logits=tf.nn.xw_plus_b(x,w,b)
    learning_rate=0.01

    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    train_op=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    predict=tf.nn.softmax(logits)

    xtest, ytest = data.test.next_batch(1000)
    TP = tf.equal(y,tf.argmax(predict,1))
    accuracy = tf.reduce_mean(tf.cast(TP, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            xb,yb=data.train.next_batch(200)
            sess.run(train_op,feed_dict={x:xb,y:yb})


        accuracyResult = sess.run( accuracy, feed_dict={x: xtest, y: ytest})
        print('accuracy:%f'%accuracyResult)

if __name__=='__main__':
    main()