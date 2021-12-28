from sklearn import datasets


def datasets_load_demo():
    """sklearn datasets 数据集 load 方法的使用
    :return:
    """
    # 获取数据集
    iris = datasets.load_iris()
    print('\n鸢尾花数据集：\n', iris)
    print('\n查看数据集描述：\n', iris['DESCR'])  # 因为数据集是一个字典，所以可以通过 dicr['key'] 取值
    print('\n查看特征值名字：\n', iris.feature_names)

    # 因为 data 返回的是 array，所以可以通过 shape 方法查看几行几列，这里的行就是有几个样本（sample），几列就是每个样本有几个属性
    print('\n查看特征值\n', iris.data, iris.data.shape)

    return None


if __name__ == '__main__':
    datasets_load_demo()
