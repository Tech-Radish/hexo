---
title: Mnist手写数字体识别(tensorflow)
date: 2018-03-20 21:35:05
categories:
- tensorflow学习
- tensorflow-Demo
tags: 
- Mnist
- tensorflow
---
# Tensorflow #


> 首先，简单的说下，tensorflow的基本架构。
>使用 TensorFlow, 你必须明白 TensorFlow:

- 使用图 (graph) 来表示计算任务.
- 在被称之为 会话 (Session) 的上下文 (context) 中执行图.
- 使用 tensor 表示数据.
- 通过 变量 (Variable) 维护状态.
- 使用 feed 和 fetch 可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据.

<!--more -->

# Tensor #

> TensorFlow 是一个编程系统, 使用图来表示计算任务. 图中的节点被称之为 op (operation 的缩写). 一个 op 获得 0 个或多个 Tensor, 执行计算, 产生 0 个或多个 Tensor. 每个 Tensor 是一个类型化的多维数组. 例如, 你可以将一小组图像集表示为一个四维浮点数数组, 这四个维度分别是 [batch, height, width, channels].

> 一个 TensorFlow 图描述了计算的过程. 为了进行计算, 图必须在 会话 里被启动. 会话 将图的 op 分发到诸如 CPU 或 GPU 之类的 设备 上, 同时提供执行 op 的方法. 这些方法执行后, 将产生的 tensor 返回. 在 Python 语言中, 返回的 tensor 是 numpy ndarray 对象; 在 C 和 C++ 语言中, 返回的 tensor 是tensorflow::Tensor 实例.

> Tensor是tensorflow中非常重要且非常基础的概念，可以说数据的呈现形式都是用tensor表示的。输入输出都是tensor，tensor的中文含义，就是张量，可以简单的理解为线性代数里面的向量或者矩阵。

# Graph #


> TensorFlow 程序通常被组织成一个构建阶段和一个执行阶段. 在构建阶段, op 的执行步骤 被描述成一个图. 在执行阶段, 使用会话执行执行图中的 op.



> 例如, 通常在构建阶段创建一个图来表示和训练神经网络, 然后在执行阶段反复执行图中的训练 op. 下面这个图，就是一个比较形象的说明，图中的每一个节点，就是一个op，各个op透过tensor数据流向形成边的连接，构成了一个图。

![](https://images2015.cnblogs.com/blog/844237/201703/844237-20170330093311608-2056024255.gif)
> 构建图的第一步, 是创建源 op (source op). 源 op 不需要任何输入, 例如 常量 (Constant). 源 op 的输出被传递给其它 op 做运算. Python 库中, op 构造器的返回值代表被构造出的 op 的输出, 这些返回值可以传递给其它 op 构造器作为输入.


> TensorFlow Python 库有一个默认图 (default graph), op 构造器可以为其增加节点. 这个默认图对 许多程序来说已经足够用了.

# Session #
> 当图构建好后，需要创建一个Session来运行构建好的图，来实现逻辑，创建session的时候，若无任何参数，tensorflow将启用默认的session。session.run(xxx)是比较典型的使用方案, session运行结束后，返回值是一个tensor。



> tensorflow中的session，有两大类，一种就是普通的session，即tensorflow.Session(),还有一种是交互式session，即tensorflow.InteractiveSession(). 使用Tensor.eval() 和Operation.run()方法代替Session.run(). 这样可以避免使用一个变量来持有会话, 为程序架构的设计添加了灵活性.


# 数据载体 #
> Tensorflow体系下，变量（Variable）是用来维护图计算过程中的中间状态信息，是一种常见高频使用的数据载体，还有一种特殊的数据载体，那就是常量（Constant），主要是用作图处理过程的输入量。这些数据载体，也都是以Tensor的形式体现。变量定义和常量定义上，比较好理解：
   
	# 创建一个变量, 初始化为标量0.没有指定数据类型（dtype）
	state = tf.Variable(0, name="counter")

	# 创建一个常量，其值为1，没有指定数据类型（dtype）
	one = tf.constant(1)



> 针对上面的变量和常量，看看Tensorflow里面的函数定义：
>
    class Variable(object):　
	def __init__(self,
		initial_value=None,
		trainable=True,
		collections=None,
		validate_shape=True,
		caching_device=None,
		name=None,
		variable_def=None,
		dtype=None,
		expected_shape=None,
		import_scope=None)：

>
	def constant(value, dtype=None, shape=None, name="Const", verify_shape=False)：

> 从上面的源码可以看出，定义变量，其实就是定义了一个Variable的实例，而定义常量，其实就是调用了一下常量函数，创建了一个常量Tensor。

> 还有一个很重要的概念，那就是占位符placeholder，这个在Tensorflow中进行Feed数据灌入时，很有用。所谓的数据灌入，指的是在创建Tensorflow的图时，节点的输入部分，就是一个placeholder，后续在执行session操作的前，将实际数据Feed到图中，进行执行即可。
>
	input1 = tf.placeholder(tf.types.float32)
	input2 = tf.placeholder(tf.types.float32)
	output = tf.mul(input1, input2)
>	
	with tf.Session() as sess:
	  print sess.run([output], feed_dict={input1:[7.], input2:[2.]})
>	
	# 输出:
	# [array([ 14.], dtype=float32)]

> 占位符的定义原型，也是一个函数：
>
	def placeholder(dtype, shape=None, name=None)：



> 到此，Tensorflow的入门级的基本知识介绍完了。下面，将结合一个MNIST的手写识别的例子，从代码上简单分析一下，源代码分成4个文件：


----------

> main.py驱动程序
 
    #!/usr/bin/env python
	# -*- coding: utf-8 -*-
	# @Time    : 2018/2/21 20:41
	# @Author  : Jasontang
	# @Site    : 
	# @File    : main.py
	# @ToDo    : 驱动程序
	
	import _thread
	
	from neural_network_learning.hand_writting_refactor import mnist_train, mnist_eval
	
	
	if __name__ == '__main__':
	    _thread.start_new_thread(mnist_train.main, (None,))
	    _thread.start_new_thread(mnist_eval.main, (None,))
	
	    # 这个不能删除，当做主线程
	    while 1:
	        pass

> mnist_inference.py计算前向传播的过程及定义了神经网络的参数
 
	#!/usr/bin/env python
	# -*- coding: utf-8 -*-
	# @Time    : 2018/2/20 19:43
	# @Author  : Jasontang
	# @Site    : 
	# @File    : mnist_inference.py
	# @ToDo    : 定义了前向传播的过程及神经网络的参数
	
	
	import tensorflow as tf
	
	# 定义神经网络结构相关的参数
	INPUT_NODE = 784
	OUTPUT_NODE = 10
	LAYER1_NODE = 500
	
	
	# 训练时会创建这些变量，测试时会通过保存的模型加载这些变量的取值
	def get_weight_variable(shape, regularizer):
	    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
	
	    # 当使用正则化生成函数时,当前变量的正则化损失加入名字为losses的集合.
	    # 自定义集合
	    if regularizer:
	        tf.add_to_collection("losses", regularizer(weights))
	    return weights
	
	
	# 前向传播过程
	def inference(input_tensor, regularizer):
	    # 声明第一层神经网络的变量并完成前向传播过程
	    with tf.variable_scope("layer1"):
	        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
	        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
	        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
	
	    # 声明第二层圣经网络变量并完成前向传播过程
	    with tf.variable_scope("layer2"):
	        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
	        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
	        layer2 = tf.matmul(layer1, weights) + biases
	    # 返回最后前向传播的结果
	    return layer2

> mnist_train.py定义了神经网络的训练过程

	#!/usr/bin/env python
	# -*- coding: utf-8 -*-
	# @Time    : 2018/2/21 16:08
	# @Author  : Jasontang
	# @Site    : 
	# @File    : mnist_train.py
	# @ToDo    : 定义了神经网络的训练过程
	
	import os
	
	import tensorflow as tf
	from tensorflow.examples.tutorials.mnist import input_data
	
	import neural_network_learning.hand_writting_refactor.mnist_inference as mnist_inference
	
	# 配置神经网络的参数
	BATCH_SIZE = 100
	LEARNING_REATE_BASE = 0.8
	LEARNING_RATE_DECAY = 0.99
	REGULARAZTION_RATE = 0.0001
	TRAING_STEPS = 2000
	MOVING_AVERAGE_DECAY = 0.99
	# 模型保存的路径和文件名
	MODEL_SAVE_PATH = "./model/"
	MODEL_NAME = "model.ckpt"
	
	
	def train(mnist):
	    # 定义输入输出placeholder
	    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name="input-x")
	    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="input-y")
	
	    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
	    y = mnist_inference.inference(x, regularizer)
	    global_step = tf.Variable(0, trainable=False)
	
	    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	    variables_average_op = variable_averages.apply(tf.trainable_variables())
	    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
	    cross_entropy_mean = tf.reduce_mean(cross_entropy)
	    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
	    learing_rate = tf.train.exponential_decay(LEARNING_REATE_BASE,
	                                              global_step,
	                                              mnist.train.num_examples / BATCH_SIZE,
	                                              LEARNING_RATE_DECAY)
	    train_step = tf.train.GradientDescentOptimizer(learing_rate).minimize(loss, global_step)
	
	    with tf.control_dependencies([train_step, variables_average_op]):
	        train_op = tf.no_op(name="train")
	
	    # 初始化持久化类
	    saver = tf.train.Saver()
	    with tf.Session() as sess:
	        tf.global_variables_initializer().run()
	
	        for i in range(TRAING_STEPS):
	            xs, ys = mnist.train.next_batch(BATCH_SIZE)
	            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
	
	            if i % 1000 == 0:
	                print("After %d training step(s), loss on training batch is %g." % (i, loss_value))
	
	                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
	
	
	def main(argv=None):
	    mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
	    train(mnist)
	
	
	if __name__ == '__main__':
	    tf.app.run()

> mnist_eval.py测试过程
 
	#!/usr/bin/env python
	# -*- coding: utf-8 -*-
	# @Time    : 2018/2/21 16:32
	# @Author  : Jasontang
	# @Site    : 
	# @File    : mnist_eval.py
	# @ToDo    : 测试过程
	
	
	import time
	import tensorflow as tf
	from tensorflow.examples.tutorials.mnist import input_data
	
	import neural_network_learning.hand_writting_refactor.mnist_inference as mnist_inference
	import neural_network_learning.hand_writting_refactor.mnist_train as mnist_train
	
	# 每10s加载一次最新模型，并在测试数据上测试最新模型的正确率
	EVAL_INTERVAL_SECS = 10
	
	
	def evaluate(mnist):
	    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name="input-x")
	    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="input-y")
	
	    validate_feed = {x: mnist.validation.images,
	                     y_: mnist.validation.labels}
	
	    y = mnist_inference.inference(x, None)
	
	    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	    variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
	    variables_to_restore = variable_averages.variables_to_restore()
	    saver = tf.train.Saver(variables_to_restore)
	
	    # 每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化
	    stop_count = 0
	    while True:
	        with tf.Session() as sess:
	            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
	            # 停止条件 #
	            stop_count += EVAL_INTERVAL_SECS
	            if stop_count == mnist_train.TRAING_STEPS:
	                return
	            # 停止条件 #
	            if ckpt and ckpt.model_checkpoint_path:
	                saver.restore(sess, ckpt.model_checkpoint_path)
	                # 通过文件名得到模型保存时迭代的轮数
	                # 输出./model/model.ckpt-29001
	                print(ckpt.model_checkpoint_path)
	                global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
	                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
	                print("After %s training step(s), validation accuracy is %g" % (global_step, accuracy_score))
	            else:
	                print("No checkpoint file found")
	                return
	        time.sleep(EVAL_INTERVAL_SECS)
	
	
	def main(argv=None):
	    mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
	    evaluate(mnist)
	
	
	if __name__ == '__main__':
	    tf.app.run()

# 参考文章 #
[https://www.cnblogs.com/shihuc/p/6648130.html](https://www.cnblogs.com/shihuc/p/6648130.html "Tensorflow之基于MNIST手写识别的入门介绍")











