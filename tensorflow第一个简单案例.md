使用tensorflow
---
  陆陆续续看深度学习也有几天了，基本上还是云里雾里的状态，主要还是数学不好以及一些概念不清楚。选择`tensorflow`的原因很简单，Google官方以及使用的人还是蛮多的，所以就先从这个入手。用什么工具并不重要，主要还是想学习一下一些思想。	
	
`tensorflow`的官方文档还是写的很完善的，当然其中的识别数字的案例还是很有名的，有兴趣的可以都看看。
![](https://cdn.rawgit.com/monkeytest15/BlogPNG/89e7646f/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202017-01-20%20%E4%B8%8B%E5%8D%884.37.38.png)


简单案例
---
  我个人感觉，像我这样的初学者一下子看各种案例就会晕头转向，简单的案例还是能够帮助我们更好的去学习这门已经火过3次却2次跌入谷底的技术。
	`tensorflow`提供了一套可以利用`cpu`和`gpu`的算法，同时也提供了一套可以展现的`dashboard`，这一切都可以在代码中进行实现。先看这个简单的代码吧，详细的我全部写在注释里。
	
```python
# coding:utf-8

# 调用tensorflow
import tensorflow as tf
import numpy as np

# 这里生成了100对数字，作为整个神经网络的input
x_data = np.random.rand(100).astype("float32")

# 使用with，让我们的数据以节点的方式落在tensorflow的报告上。
with tf.name_scope('y_data'):
	y_data = x_data * 2.5 + 0.8 #权重2.5，偏移设置2.5
    tf.histogram_summary("method_demo"+"/y_data",y_data) #可视化观看变量y_data


# 指定W和b变量的取值范围，随机在[-200,200]
with tf.name_scope('W'):
    W = tf.Variable(tf.random_uniform([1], -200.0, 200.0))
    tf.histogram_summary("method_demo"+"/W",W) #可视化观看变量

# 指定偏移值b，同时shape等于1
with tf.name_scope('b'):
    b = tf.Variable(tf.zeros([1]))
    tf.histogram_summary("method_demo"+"/b",b) #可视化观看变量

with tf.name_scope('y'):
    y = W * x_data + b #sigmoid神经元
    tf.histogram_summary("method_demo"+"/y",y) #可视化观看变量

# 最小化均方
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y - y_data))
    tf.histogram_summary("method_demo"+"/loss",loss) #可视化观看变量
    tf.scalar_summary("method_demo"+'loss',loss) #可视化观看常量

# 定义学习率，我们先使用0.7来看看效果
optimizer = tf.train.GradientDescentOptimizer(0.7)
with tf.name_scope('train'):
    train = optimizer.minimize(loss)

# 初始化TensorFlow参数
init = tf.initialize_all_variables()

# 运行数据流图
sess = tf.Session()
#合并到Summary中
merged = tf.merge_all_summaries()
#选定可视化存储目录
writer = tf.train.SummaryWriter(LOG_PATH,sess.graph)

sess.run(init)

# 开始计算
for step in xrange(500):
    sess.run(train)
    if step % 5 == 0:
        print(step, "W:",sess.run(W),"b:", sess.run(b))
        result = sess.run(merged) #merged也是需要run的
        writer.add_summary(result,step) #result是summary类型的

```
