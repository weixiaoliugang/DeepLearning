# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.framework import ops
import resnets_utils

tf.reset_default_graph() 


class hand_classifier(object):
    
    def __init__(self, model_save_path='./model_saving/hand_classifier'):
        self.model_save_path = model_save_path
        
    
    def weight_variable(self, shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    
    def cost(self, logits, labels):
        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
        cross_entropy_cost = tf.reduce_mean(cross_entropy)
        return cross_entropy_cost
    
    
    def accuracy(self, logits, labels):
        with tf.name_scope("accuracy"):
            correctPredict = tf.equal(tf.argmax(logits, axis = 1), tf.argmax(labels, axis = 1))
        accuracy = tf.reduce_mean(tf.cast(correctPredict, "float32"))
        return accuracy
     
        
    def evaluate(self, test_features, test_labels, name='test'):
        tf.reset_default_graph()
        [m, n_H, n_W, n_C] = test_features.shape
        [_, classes_num] = test_labels.shape
        X = tf.placeholder(tf.float32, shape=[None, n_H, n_W, n_C], name="X")
        Y = tf.placeholder(tf.float32, shape=[None, classes_num], name="Y")
        
        logits, keep_prob, train_mode = self.deepnn_resnet18(X)
        accuracy = self.accuracy(logits, Y)
        
        saver = tf.train.Saver()    #模型保存？？？？？？？
        with tf.Session() as sess:
            saver.restore(sess, self.model_save_path)    #加载之前训练完成保存后的模型？？？？？？
            accu = sess.run(accuracy, feed_dict={X: test_features, Y: test_labels,keep_prob: 1.0, train_mode: False})
            print('%s accuracy %g' % (name, accu))
            
            
    #恒等块，它适用于输入和输出一致的情况
    def identity_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training):
        """
        Implementation of the identity block as defined in Figure 3
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test
        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """
        # defining name basis
        block_name = 'res' + str(stage) + block
        with tf.variable_scope(block_name):
            f1, f2 = out_filters    #这个含义是干什么的？？？？？
            x_shortcut = X_input

            #first
            W_conv1 = self.weight_variable([kernel_size, kernel_size, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding="SAME")
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)
            
            #second
            W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding="SAME")
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            
            #final
            add = tf.add(X, x_shortcut)
            add_result = tf.nn.relu(add)
            return add_result
            
        
    #这个卷积块是另一种类型的残差块，它适用于输入输出的维度不一致的情况？？？？？？？
    def convolutional_block(self, X_input, kernel_size, in_filter,
                    out_filters, stage, block, training, stride=2):
        """
        Implementation of the convolutional block as defined in Figure 4
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test
        stride -- Integer, specifying the stride to be used
        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """
        # defining name basis
        block_name = 'res' + str(stage) + block
        with tf.variable_scope(block_name):
            f1, f2 = out_filters    #这个含义是干什么的？？？？？
            
            x_shortcut = X_input
            #first
            W_conv1 = self.weight_variable([kernel_size, kernel_size, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1],padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)
            
            #second
            W_conv2 = self.weight_variable([1, 1, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1],padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            
            #shortcut path
            W_shortcut = self.weight_variable([1, 1, in_filter, f2])
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')
            
            #final
            add = tf.add(X, x_shortcut)
            add_result = tf.nn.relu(add)
            return add_result
        
        
    def deepnn_resnet18(self, x_input, classes=6):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK
        -> CONVBLOCK -> IDBLOCK -> CONVBLOCK -> IDBLOCK -> AVGPOOL -> TOPLAYER
        Arguments:
        Returns:
        """
        #这边为什么需要padding？？？？？？？？？
        x = tf.pad(x_input, tf.constant([[0, 0], [3, 3, ], [3, 3], [0, 0]]), "CONSTANT")
        #tf.variable_scope('reference')？？？？？？？？
        with tf.variable_scope('reference'):
            training = tf.placeholder(tf.bool, name='training')   #这个train有什么用??????
            
            #stage 1
            #这边参数初始化可以使用xaiver
            w_conv1 = self.weight_variable([7, 7, 3, 64])
            x = tf.nn.conv2d(x, w_conv1, strides=[1, 2, 2, 1], padding='VALID')   #为什么这个卷积padding设置的为VALID？？？？？？
            x = tf.layers.batch_normalization(x, axis=3, training=training)       #这个batch_normalization？？？？？
            x = tf.nn.relu(x)
            #Max pool : 过滤器大小：3x3，步伐：2x2，填充方式：“VALID”
            x = tf.nn.max_pool(x, [1,3,3,1], [1,2,2,1], padding='VALID')          #为什么这个池化padding设置的为VALID？？？？？？
            assert (x.get_shape() == (x.get_shape()[0], 15, 15, 64))

            #stage 2
            x = self.identity_block(x, 3, 64,  [64, 64], stage=2, block='a', training=training)                  #恒等块？？？？？？
            x = self.identity_block(x, 3, 64, [64, 64], stage=2, block='b', training=training)
            
            #stage 3
            x = self.convolutional_block(x, 3, 64, [128, 128], stage=3, block='a', training=training, stride=2)
            x = self.identity_block(x, 3, 128, [128, 128], stage=3, block='b', training=training) 
            
            #stage 4
            x = self.convolutional_block(x, 3, 128, [256, 256], stage=4, block='a', training=training, stride=2)
            x = self.identity_block(x, 3, 256, [256, 256], stage=4, block='b', training=training) 
            
            #stage 5
            x = self.convolutional_block(x, 3, 256, [512, 512], stage=5, block='a', training=training, stride=2)
            x = self.identity_block(x, 3, 512, [512, 512], stage=5, block='b', training=training)
            
            
            #平均池化层
            x = tf.nn.avg_pool(x, [1, 2, 2, 1], strides=[1,1,1,1], padding='VALID')
            
            #展开操作？？？？？？？？？？？？
            flatten = tf.layers.flatten(x)
            x = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)     #全连接层操作？？？？？？
            
            # Dropout - controls the complexity of the model, prevents co-adaptation of
            with tf.name_scope("dropout"):
                keep_prob = tf.placeholder(tf.float32)
                x = tf.nn.dropout(x, keep_prob)
                
            logits = tf.layers.dense(x, units=classes, activation=tf.nn.softmax)   #全连接层操作？？？？？？
            
        return logits, keep_prob, training    #这个返回keep_prob是什么意思？？？？？
    
    
    def train(self, X_train, Y_train):
        
        #1、创建占位符号
        [m, n_H, n_W, n_C] = X_train.shape
        [_, classes_num] = Y_train.shape
        X = tf.placeholder(tf.float32, shape=[None, n_H, n_W, n_C], name="X")
        Y = tf.placeholder(tf.float32, shape=[None, classes_num], name="Y")
        
        #2、resnet模型结构定义，也就是前向传播
        logits, keep_prob, train_mode = self.deepnn_resnet18(X)    #这个返回keep_prob是什么意思？？？？？
    
        #3、计算损失函数
        cross_entropy = self.cost(logits, Y)
        
        #4、设置优化器
        with tf.name_scope("adam_optimizer"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
#        #这一块有什么用？？？？？？？
#        graph_location = tempfile.mkdtemp()
#        print('Saving graph to: %s' % graph_location)
#        train_writer = tf.summary.FileWriter(graph_location)
#        train_writer.add_graph(tf.get_default_graph())
    
        #5、随机打乱，并划分mini_batch
#        mini_batches = resnets_utils.random_mini_batches(X_train, Y_train, mini_batch_size=8, seed=None)
        
        seed = 0
        miniBatchSize = 16
        saver = tf.train.Saver()    #模型保存？？？？？？？
        with tf.Session() as sess:
            costs = []
            sess.run(tf.global_variables_initializer())
            for epoch in range(100):
                miniBatchNum = np.floor(m/miniBatchSize)
                seed = seed + 1
                epochCost = 0
                mini_batches = resnets_utils.random_mini_batches(X_train, Y_train, mini_batch_size=miniBatchSize, seed=seed)
                for miniBatch in mini_batches:
                    (X_mini_batch, Y_mini_batch) = miniBatch
#                    X_mini_batch, Y_mini_batch = mini_batches[np.random.randint(0, len(mini_batches))]
                    train_step.run(feed_dict={X: X_mini_batch, Y: Y_mini_batch, keep_prob: 0.5, train_mode: True})
                    miniBatchCost = sess.run(cross_entropy, feed_dict={X: X_mini_batch,
                                          Y: Y_mini_batch, keep_prob: 1.0, train_mode: False})
                    epochCost = epochCost + miniBatchCost/miniBatchNum


                if epoch % 1 == 0:
                    costs.append(epochCost)
                    print('step %d, training cost %g' % (epoch, epochCost))
            
            saver.save(sess, self.model_save_path) #模型保存？？？？？？？
            
            #打印训练集的准确率？？？？？？？？
            accuracy = self.accuracy(logits, Y)
            accu = sess.run(accuracy, feed_dict={X: X_train, Y: Y_train, keep_prob: 1.0, train_mode: False})
            print("训练集的准确率为：" + str(accu))

    
#主函数
if __name__ == "__main__":
    
    #加载数据集，这边数据集合要重换
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = resnets_utils.load_dataset()
    
    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.
    
    # Convert training and test labels to one hot matrices
    #转化为独热码
    Y_train = resnets_utils.convert_to_one_hot(Y_train_orig, 6).T
    Y_test = resnets_utils.convert_to_one_hot(Y_test_orig, 6).T
    
    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))
    
    
    #模型加载
    model = hand_classifier()
    model.train(X_train, Y_train)
    model.evaluate(X_test, Y_test)    #测试训练集
    model.evaluate(X_train, Y_train, 'training data')   #测试验证集
    print("sbbbb")
    