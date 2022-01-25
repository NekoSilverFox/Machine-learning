# ------*------ coding: utf-8 ------*------
# @Time    : 2022/1/25 14:08
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : Scikit-learn
# @File    : 借助 jieba 对中文分词及特征提取.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------

import jieba
from sklearn import feature_extraction


def cut_chinese_str(text):
    """
    利用 jieba 对中文进行分词
    :param text: 需要分词的字符串
    :return: 分词结束的字符串
    """

    return ' '.join(list(jieba.cut(text)))


def count_cn_text_jieba_feature_extraction():
    """
    中文文本特征提取，借助 jieba 进行分词
    :return:
    """
    data = ['一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。',
            '我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。',
            '如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取決于如何将其与我们所了解的事物相联系。']

    data_cut = []

    # 1. 利用 jieba 进行分词
    for str in data:
        data_cut.append(cut_chinese_str(str))
    print(data_cut)

    # 2. 实例化一个转换器类
    transfer = feature_extraction.text.CountVectorizer()

    # 3. 调用 fit_transform
    data_fit = transfer.fit_transform(data_cut)
    print('data_fit: \n', data_fit.toarray())  # 【重点】对于 sparse 矩阵，内部的 `.toarray()` 可以返回一个对应的二维数据
    print('特征名字:\n', transfer.get_feature_names_out())
    pass


if __name__ == '__main__':
    count_cn_text_jieba_feature_extraction()
