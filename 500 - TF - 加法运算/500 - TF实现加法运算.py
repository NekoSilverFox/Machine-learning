# ------*------ coding: utf-8 ------*------
# @Time    : 2022/6/14 14:43
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : Machine-learning
# @File    : 500 - TF实现加法运算.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------
import tensorflow as tf
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭警告


def tensorflow_demo():
    """
    使用 TensorFlow 实现加法运算
    :return:
    """
    # Python 原生实现加法
    a = 6
    b = 7
    c = a + b
    print('Python 原生方法 a+b=', c)

    # 使用 TensorFlow 实现加法运算
    tf_a = tf.constant(6)  # 定义了一个常量（张量 - tensor）
    tf_b = tf.constant(7)
    print('tf_a: ', tf_a)

    tf_c = tf_a + tf_b  # 定义了一个操作（Operation）
    print('TensorFlow 方法调用 tf_a + tf_b :', tf_c)
    print('TensorFlow 输出数值 tf_c.numpy():', tf_c.numpy())

    tf_c = tf.add(tf_a, tf_b)
    print('TensorFlow 方法调用 tf.add(tf_a, tf_b):', tf_c)

    # 【在 TF2 中已经不适用】开启一个会话。注意在 TensorFlow 中定义了一个操作后需要使用会话去调用
    # with tf.Session() as sess:
    #     print('TensorFlow 方法调用 tf_a + tf_b = ', sess.run(tf_c))


def graph_demo():
    """
    图的操作
    :return:
    """
    print('>>' * 50)

    # 使用 TensorFlow 实现加法运算
    tf_a = tf.constant(6)  # 定义了一个常量（张量 - tensor）
    tf_b = tf.constant(7)
    tf_c = tf_a + tf_b
    print('tf_c: ', tf_c)

    # 查看默认图
    # 方法 1：调用方法
    default_graph = tf.compat.v1.get_default_graph()
    print('tf.get_default_graph(): ', default_graph)

    # 方法 2：查看属性
    print('tf_a.graph: ', tf_a.graph)
    print('tf_b.graph: ', tf_b.graph)



if __name__ == '__main__':
    print(tf.__version__)
    # tensorflow_demo()

    graph_demo()
    pass

