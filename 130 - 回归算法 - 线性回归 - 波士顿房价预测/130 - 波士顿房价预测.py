# ------*------ coding: utf-8 ------*------
# @Time    : 2022/5/4 12:56
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : Scikit-learn
# @File    : 130 - 波士顿房价预测.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model


def linear_LinearRegression():
    """
    使用【正规方程】对波士顿房价的预测
    :return:
    """
    # 1. 获取数据集
    data = datasets.load_boston()

    # 2. 划分数据集
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data.data, data.target, random_state=233)

    # 3. 标准化
    transfer = preprocessing.StandardScaler()
    x_train = transfer.fit_transform(X=x_train)
    x_test = transfer.transform(x_test)

    # 4. 获取正规方程预估器
    estimator = linear_model.LinearRegression()

    # 5. 模型训练
    estimator.fit(X=x_train, y=y_train)
    print('正规方程权重：', estimator.coef_)
    print('正规方程偏置：', estimator.intercept_)

    # 6. 模型评估


def linear_SGDRegressor():
    """
    使用【梯度下降】对波士顿房价的预测
    :return:
    """
    # 1. 获取数据集
    data = datasets.load_boston()

    # 2. 划分数据集
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data.data, data.target, random_state=233)

    # 3. 标准化
    transfer = preprocessing.StandardScaler()
    x_train = transfer.fit_transform(X=x_train)
    x_test = transfer.transform(x_test)

    # 4. 获取正规方程预估器
    estimator = linear_model.SGDRegressor()

    # 5. 模型训练
    estimator.fit(X=x_train, y=y_train)
    print('梯度下降权重：', estimator.coef_)
    print('梯度下降偏置：', estimator.intercept_)

    # 6. 模型评估


if __name__ == '__main__':
    print('>>' * 50)
    linear_LinearRegression()

    print('>>' * 50)
    linear_SGDRegressor()



    pass
