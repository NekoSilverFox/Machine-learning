# ------*------ coding: utf-8 ------*------
# @Time    : 2022/4/29 15:49
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : Scikit-learn
# @File    : 决策树对鸢尾花分类.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------
from sklearn import datasets
from sklearn import model_selection
from sklearn import tree


def decision_iris():
    """
    用决策树对鸢尾花进行分类
    :return:
    """
    # 1. 获取数据集
    data_iris = datasets.load_iris()

    # 2. 数据集划分（【重点】决策树不太需要进行无量纲化，因为决策树不像 KNN 算法一样是计算距离的）
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data_iris.data, data_iris.target, random_state=233)

    # 3. 获取估计器，并训练
    estimator = tree.DecisionTreeClassifier(criterion='entropy')
    estimator.fit(X=x_train, y=y_train)

    # 4. 模型评估
    print('准确度为：', estimator.score(X=x_test, y=y_test))

    # 5. 决策树可视化
    # http://webgraphviz.com/
    tree.export_graphviz(estimator, out_file='./result/iris_tree.dot', feature_names=data_iris.feature_names)


if __name__ == '__main__':
    decision_iris()
    pass
