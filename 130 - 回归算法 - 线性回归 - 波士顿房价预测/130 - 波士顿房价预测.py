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
from sklearn import metrics
import joblib


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
    predict = estimator.predict(X=x_test)
    error_mse = metrics.mean_squared_error(y_true=y_test, y_pred=predict)  # 均方误差(Mean Squared Error - MSE)评价机制：
    print('梯度下降的均方误差为：', error_mse)


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
    estimator = linear_model.SGDRegressor(learning_rate='constant', eta0=0.001, max_iter=10000)

    # 5. 模型训练
    estimator.fit(X=x_train, y=y_train)
    print('梯度下降权重：', estimator.coef_)
    print('梯度下降偏置：', estimator.intercept_)

    # 6. 模型评估
    predict = estimator.predict(X=x_test)
    error_mse = metrics.mean_squared_error(y_true=y_test, y_pred=predict)  # 均方误差(Mean Squared Error - MSE)评价机制：
    print('梯度下降的均方误差为：', error_mse)


def linear_Ridger():
    """
    使用 岭回归（Ridger）进行对波士顿放假的预测
    :return:
    """
    # 1. 获取数据集
    data = datasets.load_boston()

    # 2. 划分数据集
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data.data, data.target, random_state=233)

    # 3. 标准化
    transfer = preprocessing.StandardScaler()
    x_train = transfer.fit_transform(X=x_train)
    x_test = transfer.transform(X=x_test)

    # 4. 获取岭回归预估器
    # estimator = linear_model.Ridge(alpha=0.5, random_state=233, max_iter=10000)
    # estimator.fit(X=x_train, y=y_train)
    # print('Ridge 岭回归权重：', estimator.coef_)
    # print('Ridge 岭回归偏置：', estimator.intercept_)

    # 保存模型
    # joblib.dump(estimator, 'estimator_ridge.pkl')

    # 模型加载
    estimator = joblib.load('estimator_ridge.pkl')


    # 5. 模型评估
    predict = estimator.predict(X=x_test)
    error_mse = metrics.mean_squared_error(y_true=y_test, y_pred=predict)
    print('Ridge 岭回归的均方误差为：', error_mse)


if __name__ == '__main__':
    print('>>' * 50)
    linear_LinearRegression()

    print('>>' * 50)
    linear_SGDRegressor()

    print('>>' * 50)
    linear_Ridger()



    pass
