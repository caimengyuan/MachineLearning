'''
Dropout
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

#创建一个简单的神经网络
# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))
W = tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
b = tf.Variable(tf.zeros([2000])+0.1)
# prediction = tf.nn.softmax(tf.matmul(x,W) + b)
L1 = tf.nn.tanh(tf.matmul(x,W) + b)
L1_drop = tf.nn.dropout(L1, keep_prob)

W1 = tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
b1 = tf.Variable(tf.zeros([2000])+0.1)
# prediction = tf.nn.softmax(tf.matmul(x,W) + b)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W1) + b1)
L2_drop = tf.nn.dropout(L2, keep_prob)

W2 = tf.Variable(tf.truncated_normal([2000,10],stddev=0.1))
b2= tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L2_drop,W2) + b2)


#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
#交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})

        test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:10})
        train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:10})

        print("Iter" + str(epoch)+",Testing Acuracy"+str(test_acc)+"Training test"+str(train_acc))
