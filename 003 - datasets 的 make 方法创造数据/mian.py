from sklearn import datasets

"""一个图像化的工具"""
import matplotlib.pyplot as plt

"""这里用了制造一个线性回归的函数"""
X, y = datasets.make_regression(
    n_samples=100,  # 数据数量
    n_features=1,  # 特征数量
    n_targets=1,  # 标签数量
    noise=1  # 噪声
)

"""用点的形式输出结果"""
plt.scatter(X, y)
plt.show()
