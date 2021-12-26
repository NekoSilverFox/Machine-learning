import numpy as np

# sklearn 提供的数据库（集）
from sklearn import datasets
from sklearn.model_selection import train_test_split

# KNeighborsClassifier 综合临近的两个点来预测值，KNN 算法
from sklearn.neighbors import KNeighborsClassifier

# iris 是一种花，可以 Google
iris = datasets.load_iris()

# iris 的【属性】全部储存在 data 当中
iris_X = iris.data
print('print(iris_X[:2, :]) 获取的【属性】：')
print(iris_X[:2, :])  # 打印一下【属性】，这里的 2 代表 2 个例子
print('\n')

# iris 的【分类】全部储存在 target 当中，在这个例子中有 4 个属性
iris_y = iris.target
print('iris_y 获取的【分类】，一种数字代表一个分类：')
print(iris_y)  # 打印一下【分类】
print('\n')

# 将数据分为用于训练的（train）和用于测试的（test）                         ↓ 用于测试的比例为 30% 于总数据，也就为 X_test 和 y_test 的和占了总数据的  30%
# 这里也同时将用于训练的数据给打乱了，【重点】在机器学习中乱的数据总比整齐的数据好！
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
print('打印用于训练的数据 y_train，可以发现数据被打乱了：')
print(y_train)
print('\n')

# 定义用 sklearn 哪一种模式
knn = KNeighborsClassifier()
# 【重点】进行训练 -> 把我们用于训练的数据放进去，也就是把我们的属性和分类放进去
knn.fit(X_train, y_train)
# 训练结束后，训练完成后，用它来预测一下是哪种花
print('预测值：')
print(knn.predict(X_test))  # 预测
print('真实值：')
print(y_test)  # 真实的数据







