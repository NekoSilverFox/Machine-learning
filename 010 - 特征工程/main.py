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


if __name__ == '__main__':
    dic_feature_extraction()
