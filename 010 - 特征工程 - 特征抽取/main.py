from sklearn import feature_extraction  # 特征工程提取 API


def dic_feature_extraction():
    """
    字典特征提取
    :return:
    """
    data = [
        {'city': '北京', 'temperature': 100},
        {'city': '上海', 'temperature': 60},
        {'city': '深圳', 'temperature': 30}
    ]

    # 1. 实例化一个转换器对象
    transfer = feature_extraction.DictVectorizer(sparse=False)

    # 2. 调用 fit_transform(data)
    data_fit = transfer.fit_transform(data)
    print('data_fit:\n', data_fit)
    print('特征名字:\n', transfer.get_feature_names_out())


def count_en_text_feature_extraction():
    """
    英文文本特征提取
    :return:
    """
    data = ['life is short, i like python',
            'life is too long, i dislike python']

    # 1. 实例化转换器类
    transfer = feature_extraction.text.CountVectorizer()

    # 2. 调用 fit_transform
    data_fit = transfer.fit_transform(data)
    print('data_fit:\n', data_fit.toarray(), type(data_fit.toarray()))  # 【重点】对于 sparse 矩阵，内部的 `.toarray()` 可以返回一个对应的二维数据 numpy.ndarray
    print('特征名字:\n', transfer.get_feature_names_out())


def count_cn_text_feature_extraction():
    """
    中文文本特征提取，注意，这个 API 只能对英文有较好的分析，因为是用空格作为词与词之间的分隔，所以除非中文的各个词用空格分开，否则无法分析！
    :return:
    """
    data = ['我 爱 北京 天安门',
            '天安门 上 太阳 升']

    # 1. 实例化转换器类
    transfer = feature_extraction.text.CountVectorizer()

    # 2. 调用 fit_transform
    data_fit = transfer.fit_transform(data)
    print('data_fit:\n', data_fit.toarray())  # 【重点】对于 sparse 矩阵，内部的 `.toarray()` 可以返回一个对应的二维数据
    print('特征名字:\n', transfer.get_feature_names_out())


if __name__ == '__main__':
    dic_feature_extraction()
    # count_en_text_feature_extraction()
    # count_cn_text_feature_extraction()
