from sklearn import datasets
from sklearn.linear_model import LinearRegression

"""从 sklearn 的 datasets 中导入数据"""
loaded_data = datasets.load_boston()

"""导入属性和标签"""
data_X = loaded_data.data
data_y = loaded_data.target

"""导入模型"""
model = LinearRegression()

"""让他去学习"""
model.fit(data_X, data_y)

print('用前四个数据预测一下值：')
print(model.predict(data_X[:4, :]))

print('前四个数据的真实值：')
print(data_y[:4])
