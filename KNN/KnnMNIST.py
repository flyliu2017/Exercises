from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

def main():
    k=10
    batch_size=500
    test_size=batch_size//10
    data=input_data.read_data_sets('KnnMNIST/')
    xtrain, ytrain = data.test.next_batch(batch_size)
    xtest, ytest = data.test.next_batch(test_size)
    num_feature=len(xtest[0])

    s1=tf.reduce_sum(tf.square(xtrain),1,keep_dims=True)
    s2=tf.reduce_sum(tf.square(xtest),1,keep_dims=True)
    m=tf.multiply(xtest,tf.transpose(xtrain))
    distance=tf.subtract(tf.add(tf.transpose(s1),s2),2*m)
    # sub=tf.subtract(tf.expand_dims(xtrain,0),tf.expand_dims(xtest,1))
    # distance=tf.sqrt(tf.reduce_sum(tf.square(sub),-1))
    top_k=tf.nn.top_k(tf.negative(distance),k)[1]



    with tf.Session() as sess:
        topk=sess.run(top_k)
        frequency=np.zeros([test_size,num_feature])
        for i in range(len(topk)):
            for j in topk[i]:
                frequency[i][ytrain[j]]+=1
        predict=tf.convert_to_tensor(np.argmax(frequency,1),dtype=ytest.dtype)
        TP = tf.equal(ytest, predict)
        accuracy = tf.reduce_mean(tf.cast(TP, tf.float32))

        accuracyResult = sess.run( accuracy)
        print('accuracy:%f'%accuracyResult)

if __name__=='__main__':
    main()