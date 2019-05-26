import tensorflow as tf

#在定义两个placeholder处加上
with tf.name_sope('input'):
    #定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 1],name='x_input')
    y = tf.placeholder(tf.float32, [None, 1],name='y_input')

with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([785,10]), name='W')
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x,W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

    # writer = tf.summary.FileWriter('logs/',sess.graph)
#tensorboard --logdir='路径'