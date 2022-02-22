# ------*------ coding: utf-8 ------*------
# @Time    : 2022/2/22 17:39
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : Scikit-learn
# @File    : 特征降维.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------
import pandas as pd
from sklearn import feature_selection
from scipy.stats import pearsonr


def varince():
    """
    特征工程 - 特征降维 - 特征选择 - 低方差过滤
    :return:
    """
    # 1. 获取数据
    data = pd.read_csv('factor_returns.csv')
    print(data.head(), data.shape)

    # 2. 裁切数据
    data = data.iloc[:, 1: -2]
    print(data.head(), data.shape)


    # 3. 获取转换器对象
    transfer = feature_selection.VarianceThreshold(threshold=0.0)

    # 4. 调用 fit_transform
    data_new = transfer.fit_transform(data)
    print(data_new, data_new.shape)

    # 计算皮尔逊相关系数
    r = pearsonr(x=data['pe_ratio'], y=data['pb_ratio'])
    print('相关系数：', r)


if __name__ == '__main__':
    varince()
    pass
