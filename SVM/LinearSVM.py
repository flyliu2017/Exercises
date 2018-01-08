import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

def input_fn():
    return {
        'example_id':
            tf.constant([str(i) for i in range(700)]),

        'score1':
            tf.constant(xdata[:700,0]),

        'score2':
            tf.constant(xdata[:700,1])
    },tf.constant(ydata[:700])

def input_fn1():
    return {
        'example_id':
            tf.constant([str(i) for i in range(700,1000)]),

        'score1':
            tf.constant(xdata[700:,0]),

        'score2':
            tf.constant(xdata[700:,1])
    }

score1=contrib.layers.real_valued_column('score1')
score2=contrib.layers.real_valued_column('score2')


classifier=contrib.learn.SVM(example_id_column='example_id',
                             feature_columns=[score1,score2],
                             l2_regularization=1.0)


fit=classifier.fit(input_fn=input_fn,steps=10000)
predict=classifier.predict(input_fn=input_fn1,as_iterable=False)



accuracy=contrib.metrics.accuracy(y,y2)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
acc=sess.run(accuracy,feed_dict={y:np.reshape(predict['classes'],[-1,1]),y2:ydata[700:]})
print(acc)

# classifier=SVC(C=1,max_iter=1000)
# fit=classifier.fit(xdata[:700],ydata[:700])
# predict=classifier.predict(xdata[700:])
# print(accuracy_score(ydata[700:],predict))