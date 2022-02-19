# ------*------ coding: utf-8 ------*------
# @Time    : 2022/1/25 19:41
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : Scikit-learn
# @File    : 归一化与标准化.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------
from sklearn import preprocessing
import pandas as pd


def min_max():
    """
    归一化
    :return:
    """
    # 1. 获取数据
    data = pd.read_csv('dating.txt')

    # 1. 利用 Pandas 截取数据（注意，这里只需要对特征值进行归一化，而目标值不需要）
    data = data.iloc[:, :3]
    print(data)

    # 3. 获取转换器
    transfer = preprocessing.MinMaxScaler(feature_range=(0, 1))

    # 4. 进行归一化
    data_minmax = transfer.fit_transform(data)
    print(data_minmax)


def standar():
    """
    标准化
    标准化适用于现代嘈杂的大数据场景
    :return:
    """
    # 1. 获取数据
    data = pd.read_csv('dating.txt')

    # 2. 截取数据
    data = data.iloc[:, :3]

    # 3.标准化
    transfer = preprocessing.StandardScaler()
    data_new = transfer.fit_transform(data)

    print(data_new)


if __name__ == '__main__':
    # min_max()
    standar()

    pass
