from sklearn import datasets
from sklearn import model_selection  # 模型选择，内包含数据集划分 API（train_test_split）


def datasets_load_demo():
    """sklearn datasets 数据集 load 方法的使用
    :return: None
    """
    # 获取数据集
    iris = datasets.load_iris()
    print('\n鸢尾花数据集：\n', iris)
    print('\n查看数据集描述：\n', iris['DESCR'])  # 因为数据集是一个字典，所以可以通过 dicr['key'] 取值
    print('\n查看特征值名字：\n', iris.feature_names)
    # 因为 data 返回的是 array，所以可以通过 shape 方法查看几行几列，这里的行就是有几个样本（sample），几列就是每个样本有几个属性
    print('\n查看特征值\n', iris.data, iris.data.shape)

    return None


def datasets_split():
    """ 数据集的划分
    :return:
    """
    # 获取数据集
    iris = datasets.load_iris()

    # 数据集划分
    x_train, x_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.2)
    print('训练集的特征值：\n', x_test, x_test.shape)


if __name__ == '__main__':
    datasets_load_demo()
    datasets_split()
    
