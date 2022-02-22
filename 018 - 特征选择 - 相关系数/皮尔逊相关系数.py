# ------*------ coding: utf-8 ------*------
# @Time    : 2022/2/22 18:36
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : Scikit-learn
# @File    : 皮尔逊相关系数.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------

from scipy.stats import pearsonr  # 导包

def pearson_correlation_coefficient():
    """
    皮尔逊相关系数
    :return:
    """
    x1 = [12.5, 15.3, 23.2, 26.4, 33.5, 34.4, 39.4, 45.2, 55.4, 60.9]
    x2 = [21.2, 23.9, 32.9, 34.1, 42.5, 43.2, 49.0, 52.8, 59.4, 63.5]

    r = pearsonr(x=x1, y=x2)
    print(r)  # 输出：(0.9941983762371884, 4.922089955456964e-09)


if __name__ == '__main__':
    pearson_correlation_coefficient()
    pass
