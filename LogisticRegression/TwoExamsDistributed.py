import tensorflow as tf
import numpy as np
import time
import pickle
from  sklearn.metrics import roc_auc_score

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate',1,'Initial learning rate.')
tf.app.flags.DEFINE_integer('steps_to_validate',1000,'Steps to validate and print message.')
tf.app.flags.DEFINE_string('ps_hosts','','Comma-separated list of parameter servers : (ip:port) pairs')
tf.app.flags.DEFINE_string('worker_hosts','','Comma-separated list of work servers : (ip:port) pairs')
tf.app.flags.DEFINE_string('job_name','',"'ps' or 'worker'")
tf.app.flags.DEFINE_integer('task_index',0,"Index of task in the job")
tf.app.flags.DEFINE_integer('sync',0,"0 for Asynchronous, 1 for Synchronize")

learning_rate=FLAGS.learning_rate
steps_to_validate=FLAGS.steps_to_validate
sync=FLAGS.sync
num_iters=20000


fx = open('x1000.data', 'rb')
fy = open('y1000.data', 'rb')

xdata=pickle.load(fx)
ydata=pickle.load(fy)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))


def main(_):
    hosts = {}
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    if ps_hosts[0] != '':
        hosts['ps'] = ps_hosts
    if worker_hosts[0] != '':
        hosts['worker'] = worker_hosts
    cluster=tf.train.ClusterSpec(hosts)
    server=tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

    total_time = 0
    num_features=len(xdata[0])
    is_chief=FLAGS.task_index==0
    batch_size=len(xdata)//len(worker_hosts)


    print('num_features: %d' %num_features)
    print('trainset size: %d' %len(xdata))
    
    if FLAGS.job_name=='ps':
        server.join()
    elif FLAGS.job_name=='worker':
        with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:%d'%FLAGS.task_index,
                                                      cluster=cluster)):


            w=init_weights([num_features,1])
            global_step=tf.Variable(0,trainable=False)
            opt=tf.train.AdagradOptimizer(learning_rate)

            x = tf.placeholder('float32', [None, num_features])
            y = tf.placeholder('float32', [None, 1])

            i=FLAGS.task_index
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                  (logits=tf.matmul(x,w),
                                   labels=y))
            grad_var = opt.compute_gradients(loss)


            hook=[]
            if sync==0:
                train_op=opt.apply_gradients(grad_var,global_step=global_step)
            else:
                spo_op=tf.train.SyncReplicasOptimizer(opt,replicas_to_aggregate=len(worker_hosts),
                                                      total_num_replicas=len(worker_hosts))
                train_op=spo_op.apply_gradients(grad_var,global_step=global_step)
                hook.append(spo_op.make_session_run_hook(is_chief=is_chief))

            h = tf.matmul(x,w)
            predict = tf.nn.sigmoid(h)
            total_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=h, labels=y))

            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=is_chief,
                                                   hooks=hook) as sess:

                for global_step in range(num_iters) :
                    if not sess.should_stop():
                        start = time.time()
                        sess.run(train_op,feed_dict={x:xdata[i * batch_size:(i + 1) * batch_size ],
                                                     y:ydata[i * batch_size:(i + 1) * batch_size ]})

                        t=time.time()-start
                        total_time+=t
                        if  global_step%steps_to_validate==0:
                            predictResult, lossResult= sess.run([predict, total_loss], feed_dict={x: xdata, y: ydata})
                            print('step ',global_step,' auc:', roc_auc_score(np.array(ydata), predictResult),' loss:', lossResult,' %.4fs/batch' %t)

                print('Parameters are:')
                print(sess.run(w))
                predictResult, lossResult = sess.run([predict, total_loss], feed_dict={x: xdata, y: ydata})
                print('auc :%f  loss:%f'%(roc_auc_score(np.array(ydata), predictResult),lossResult))
                print('Total time:%ds'%total_time)

if __name__ =='__main__':
    tf.app.run()

