import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('/root/PycharmProjects/tf/data',one_hot=True)

lr=tf.Variable(1.0)
batch_size=tf.placeholder(tf.int32,[])
keep_prob=tf.placeholder(tf.float32,[])

input_size=num_step=28

hidden_size=256
layer_num=2
class_num=10

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,class_num])
X=tf.reshape(x, [-1, num_step, input_size])

def get_lstm_cell():
    lstmcell=rnn.BasicLSTMCell(hidden_size,reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(lstmcell,input_keep_prob=keep_prob,output_keep_prob=keep_prob)
rnncell=rnn.MultiRNNCell([get_lstm_cell() for _ in range(layer_num)])

init_state=rnncell.zero_state(batch_size,dtype=tf.float32)

inputs=tf.unstack(X,axis=1)
# with tf.variable_scope('rnn',reuse=tf.AUTO_REUSE) as scope:

outputs,state=tf.nn.static_rnn(rnncell,inputs,initial_state=init_state)

# output=tf.reshape(tf.concat(outputs,1),[-1,num_step,hidden_size])
# h_state=output[:,-1,:]
h_state=outputs[-1]

w=tf.Variable(tf.truncated_normal([hidden_size,class_num],stddev=0.1),dtype=tf.float32)
b=tf.Variable(tf.constant(0.1,shape=[class_num]),dtype=tf.float32)
pre=tf.nn.softmax(tf.nn.xw_plus_b(h_state,w,b))

# loss=tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y,logits=pre))
loss=-tf.reduce_mean(tf.multiply(y,tf.log(pre)))
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),10)
optimizer = tf.train.GradientDescentOptimizer(lr)
train_op=optimizer.apply_gradients(zip(grads,tvars),global_step=tf.train.get_or_create_global_step())
# train_op=tf.train.AdamOptimizer(0.001).minimize(loss)

correct=tf.equal(tf.argmax(pre,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        _batch_size = 128
        batch = mnist.train.next_batch(_batch_size)
        if (i + 1) % 200 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                x: batch[0], y: batch[1], keep_prob: 1.0, batch_size: _batch_size})
            print("Iter%d, step %d, training accuracy %g" % (mnist.train.epochs_completed, (i + 1), train_accuracy))
        sess.run(train_op, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5, batch_size: _batch_size})

    testdata=mnist.test.next_batch(500)
    print("test accuracy %g" % sess.run(accuracy, feed_dict={
        x: testdata[0], y: testdata[1], keep_prob: 1.0, batch_size:500}))


