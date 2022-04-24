# ------*------ coding: utf-8 ------*------
# @Time    : 2022/4/24 23:11
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : Scikit-learn
# @File    : 105 - 20类文章分类.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------
from sklearn import datasets
from sklearn import model_selection
from sklearn import feature_extraction
from sklearn import naive_bayes


def demo():
    """
    使用朴素贝叶斯算法对文章进行分类
    :return:
    """
    # 1. 获取数据集
    data_news = datasets.fetch_20newsgroups(data_home='./data/', subset='all')  # subset='all' 意思为加载全部数据集

    # 2. 划分数据集
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data_news.data, data_news.target)

    # 3. 使用 Tf-IDF 提取关键词
    transfer = feature_extraction.text.TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4. 朴素贝叶斯估计器
    estimator = naive_bayes.MultinomialNB(alpha=1.0)
    estimator.fit(x_train, y_train)

    # 5. 模型评估
    score = estimator.score(X=x_test, y=y_test)
    print('预测准确率为：\n', score)

    pass


if __name__ == '__main__':
    demo()
    pass
