from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def main():
    k=5
    data=input_data.read_data_sets('Single_layer_Perceptron/')
    xtrain, ytrain = data.test.next_batch(2000)

    xtest, ytest = data.test.next_batch(200)
    sub=tf.subtract(tf.expand_dims(xtrain,0),tf.expand_dims(xtest,1))
    distance=tf.sqrt(tf.reduce_sum(tf.square(sub),-1))
    top_k=tf.nn.top_k(tf.negative(distance),k)[1]



    with tf.Session() as sess:
        topk=sess.run(top_k)
        frequency=[{j:0 for j in range(10)} for _ in range(200)]
        for i in topk:
            for j in i:
                frequency[i][ytrain[j]]+=1

        predict=[max(i,lambda  key:i[key])[0] for i in frequency]
        TP = tf.equal(ytest, predict)
        accuracy = tf.reduce_mean(tf.cast(TP, tf.float32))

        accuracyResult = sess.run( accuracy)
        print('accuracy:%f'%accuracyResult)

if __name__=='__main__':
    main()