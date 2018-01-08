import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def conv(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def weight_var(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_var(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def cnn(x):
    with tf.name_scope('reshape'):
        data = tf.reshape(x, shape=[-1, 28, 28, 1])

    with tf.name_scope('conv1'):
        w_conv1 = weight_var([5, 5, 1, 32])
        b_conv1 = bias_var([32])
        h_conv1 = tf.nn.relu(conv(data, w_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        w_conv2 = weight_var([5, 5, 32, 64])
        b_conv2 = bias_var([64])
        h_conv2 = tf.nn.relu(conv(h_pool1, w_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1'):
        w_fc1 = weight_var([7*7*64,1024])
        b_fc1= bias_var([1024])
        h_pool2_flat=tf.reshape(h_pool2,shape=[-1,7*7*64])

        h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

    with tf.name_scope('dropout'):
        keep_prob=tf.placeholder(tf.float32)
        h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

    with tf.name_scope('fc2'):
        w_fc2 = weight_var([1024,10])
        b_fc2= bias_var([10])

        y_pred=tf.nn.relu(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

    return y_pred,keep_prob

def main():
    mnist=input_data.read_data_sets('data/')
    x=tf.placeholder(tf.float32,[None,784])
    y=tf.placeholder(tf.int64,[None])
    y_pred,keep_prob=cnn(x)

    x_test=mnist.test.images
    y_test=mnist.test.labels

    with tf.name_scope('loss'):
        cross_entropy=tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_pred)
        loss=tf.reduce_mean(cross_entropy)

    with tf.name_scope('train'):
        train_op=tf.train.AdamOptimizer(1e-4).minimize(loss)

    with tf.name_scope('accuracy'):
        correct=tf.cast(tf.equal(tf.argmax(y_pred,1),y),tf.float32)
        accuracy=tf.reduce_mean(correct)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch=mnist.train.next_batch(50)

            train_op.run(feed_dict={x:batch[0],y:batch[1],keep_prob:0.5})

            if (i+1)%100 ==0:
                acc=accuracy.eval(feed_dict={
                    x:x_test,y:y_test,keep_prob:1.0
                })
                print('step %d, training accuracy %g' % (i+1, acc))

if __name__=='__main__':
    main()




