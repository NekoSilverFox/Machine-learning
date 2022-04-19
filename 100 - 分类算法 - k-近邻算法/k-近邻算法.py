# ------*------ coding: utf-8 ------*------
# @Time    : 2022/4/19 18:39
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : Scikit-learn
# @File    : k-近邻算法.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import neighbors

if __name__ == '__main__':
    """
    目的：使用 k-近邻算法预测和评估鸢尾花分类
    """

    # 1. 获取数据集合
    iris = datasets.load_iris()

    # 2. 数据集划分，因为这个数据集优化的较好，所以不需要特征降维了
    x_train, x_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, random_state=233)  # 设置一个随机数种子便于对比

    # 3. 无量纲化（标准化）
    transfer = preprocessing.StandardScaler()
    x_train = transfer.fit_transform(X=x_train, y=y_train)
    """ 【重点】
    这里为测试集的特征值（x_test）调用的是 transform 而不是 fit_transform！因为测试集应按照训练集的 fit 进行缩放！！！
    至于为什么需要了解 fit 和 transform 之间的区别和他们各自的原理 """
    x_test = transfer.transform(X=x_test)  # 用 x_train 的 fit 进行 transform

    # 4. KNN 算法预估器
    estimator = neighbors.KNeighborsClassifier(n_neighbors=5)
    estimator.fit(X=x_train, y=y_train)

    # 5. 获得预测值结果，进行模型评估

    #   1) 方法 1：直接对于实际值和预测值
    y_predict = estimator.predict(X=x_test)  # 根据测试集的特征值（x_test）预测目标值
    print('方法 1：直接对于实际值和预测值\ny_predict:\n', y_predict)
    print('结果：\n', y_predict == y_test)

    #   2) 方法 2：计算准确率
    score = estimator.score(X=x_test, y=y_test)  # 注意：这里是把之前划分的测试集丢进去，score 方法会直接进行评估准确率
    print('\n方法 2：计算准确率\n', score)

    pass

