import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np

fx=open('/root/PycharmProjects/dataset/x1000.txt')
fy=open('/root/PycharmProjects/dataset/y1000.txt')


xraw=np.array([[float(i) for i in line.split() ] for line in fx.readlines()])
yraw=[ [float(line)] for line in fy.readlines()]
xdata=np.matrix(xraw,dtype=np.float32)
ydata=np.matrix(yraw,dtype=np.int32)


batch_size=10
num_iters=20000
epsilon=1e-10

num_features=xdata[0].size
print('num_features: %d' %num_features)

print('trainset size: %d' %len(xdata))

x=tf.placeholder(tf.float32,[None,num_features])
y=tf.placeholder(tf.int32,[None,1])
y2=tf.placeholder(tf.int32,[None,1])


params=contrib.tensor_forest.python.tensor_forest.ForestHParams(num_classes=2,num_features=num_features,num_trees=1000,max_nodes=100)

classifier=contrib.learn.SKCompat(contrib.tensor_forest.client.random_forest.TensorForestEstimator(params))

fit=classifier.fit(xdata[:700],ydata[:700])
predict=classifier.predict(xdata[700:])
# accuracy=contrib.metrics.accuracy(tf.convert_to_tensor(predict['classes'],dtype=tf.int32),ydata[700:])
accuracy=contrib.metrics.accuracy(y,y2)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
acc=sess.run(accuracy,feed_dict={y:np.reshape(predict['classes'],[-1,1]),y2:ydata[700:]})
print(acc)
# ct.tensor_forest.random_forest()