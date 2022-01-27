<p align="center">
 <img width="100px" src="https://raw.githubusercontent.com/NekoSilverFox/NekoSilverfox/403ab045b7d9adeaaf8186c451af7243f5d8f46d/icons/silverfox.svg" align="center" alt="NekoSilverfox" />
 <h1 align="center">机器学习</h2>
 <p align="center"><b>基于 scikit-learn 库</b></p>
</p>

<div align=center>


[![License](https://img.shields.io/badge/license-Apache%202.0-brightgreen)](LICENSE)
![Library](https://img.shields.io/badge/Library-Scikit--learn-orange)
![Python](https://img.shields.io/badge/Python-3.8+-blue)


<div align=left>
<!-- 顶部至此截止 -->



[toc]



# 概述

## 人工智能、机器学习、深度学习的关系

![image-20211228165701734](doc/pic/README/image-20211228165701734.png)

- 机器学习和人工智能，深度学习的关系

    机器学习是人工智能的一个实现途径

    深度学习是机器学习的一个方法（神经网络）发展而来

![image-20211228171230823](doc/pic/README/image-20211228171230823.png)



## 什么是机器学习

机器学习是从数据中自动分析获得模型，并利用模型对未知数据进行预测。

也就是说，通过**数据**来得到一个**模型**，然后再拿着这个模型去**预测**

![image-20211228171519821](doc/pic/README/image-20211228171519821.png)

 

## 机器学习应用场景

![image-20211228170732036](doc/pic/README/image-20211228170732036.png)

- 用在挖掘、预测领域：

    ​	应用场景：店铺销量预测、量化投资、广告推荐、企业客户分类、SQL语句安全检测分类...

- 用在图像领域：

    ​	应用场景：街道交通标志检测、人脸识别等等

- 用在自然语言处理领域：

    ​	应用场景：文本分类、情感分析、自动聊天、文本检测等等



## 数据集

在明确了机器学习的相关概念后，我们知道机器学习需要有数据来训练模型，那么我们的数据又是如何构成的呢？格式又是如何？

- **结构：特征值 + 目标值**

    - 对于**每一行数据**我们可以称之为**样本**
    - 有些数据集可以没有目标值

    

    比如下图的房屋数据：

    **特征值**就可以看作是房屋的面积、位置、楼层、朝向

    **目标值**就可以看做是房屋的价格

    ![image-20211228172647016](doc/pic/README/image-20211228172647016.png)





# 机器学习的算法

## 算法的大致分类

| 特征值                     | 目标值                           | 问题描述                 | 举例                                               |
| -------------------------- | -------------------------------- | ------------------------ | -------------------------------------------------- |
| （比如：猫狗的图片）       | **类别**（比如：猫还是狗）       | **分类问题**（监督学习） | k-临近算法、贝叶斯分类、决策树与随机森林、逻辑回归 |
| （比如：房屋面积、位置等） | **连续型数据**（比如：房屋价格） | **回归问题**（监督学习） | 线性回归、岭回归                                   |
|                            | **没有目标值**                   | **聚类**（无监督学习）   | k-means                                            |
|                            |                                  |                          |                                                    |

![image-20211228175206865](doc/pic/README/image-20211228175206865.png)



**举例：**

- 预测明天的气温是多少度？

    ​	回归问题

- 预测明天是阴、晴还是雨？

    ​	分类问题

- 人脸年龄预测？

    ​	回归或分类，取决如如何定义年龄

- 人脸识别？

    ​	分类





## 机器学习类型（算法）的具体描述

| 类型       | 特点                               |
| ---------- | ---------------------------------- |
| 监督学习   | 有数据、有标签                     |
| 非监督学习 | 有数据、无标签                     |
| 半监督学习 | 结合监督学习和非监督学习           |
| 强化学习   | 从经验中总结提升                   |
| 遗传算法   | 和强化学习类似，适者生存不适者淘汰 |



### 监督学习（Supervised learning）

**监督学习**（英语：Supervised learning），又叫有监督学习，监督式学习，是[机器学习](https://zh.wikipedia.org/wiki/机器学习)的一种方法，可以由训练资料中学到或建立一个模式（函数 / learning model），并依此模式推测新的实例。[[1\]](https://zh.wikipedia.org/wiki/监督学习#cite_note-1)[训练资料](https://zh.wikipedia.org/wiki/訓練資料)是由输入物件（通常是向量）和预期输出所组成。函数的输出可以是一个连续的值（称为[回归分析](https://zh.wikipedia.org/wiki/迴歸分析)），或是预测一个分类标签（称作[分类](https://zh.wikipedia.org/wiki/分类)）。

监督式学习有两种形态的模型。最一般的，监督式学习产生一个全域模型，会将输入物件对应到预期输出。而另一种，则是将这种对应实作在一个区域模型。（如[案例推论](https://zh.wikipedia.org/wiki/案例推论)及[最近邻居法](https://zh.wikipedia.org/wiki/最近鄰居法)）。为了解决一个给定的监督式学习的问题（[手写辨识](https://zh.wikipedia.org/wiki/手写识别)），必须考虑以下步骤：

1. 决定训练资料的范例的形态。在做其它事前，工程师应决定要使用哪种资料为范例。譬如，可能是一个手写字符，或一整个手写的辞汇，或一行手写文字。
2. 搜集训练资料。这资料须要具有真实世界的特征。所以，可以由人类专家或（机器或感测器的）测量中得到输入物件和其相对应输出。
3. 决定学习函数的输入特征的表示法。学习函数的准确度与输入的物件如何表示是有很大的关联度。传统上，输入的物件会被转成一个特征向量，包含了许多关于描述物件的特征。因为[维数灾难](https://zh.wikipedia.org/wiki/维数灾难)的关系，特征的个数不宜太多，但也要足够大，才能准确的预测输出。
4. 决定要学习的函数和其对应的学习演算法所使用的资料结构。譬如，工程师可能选择[人工神经网路](https://zh.wikipedia.org/wiki/人工神经网络)和[决策树](https://zh.wikipedia.org/wiki/决策树)。
5. 完成设计。工程师接著在搜集到的资料上跑学习演算法。可以借由将资料跑在资料的子集（称为*验证集*）或[交叉验证](https://zh.wikipedia.org/wiki/交叉驗證)（cross-validation）上来调整学习演算法的参数。参数调整后，演算法可以运行在不同于训练集的测试集上

另外对于监督式学习所使用的辞汇则是分类。现著有著各式的分类器，各自都有强项或弱项。分类器的表现很大程度上地跟要被分类的资料特性有关。并没有某一单一分类器可以在所有给定的问题上都表现最好，这被称为‘天下没有白吃的午餐理论’。各式的经验法则被用来比较分类器的表现及寻找会决定分类器表现的资料特性。决定适合某一问题的分类器仍旧是一项艺术，而非科学。

目前最广泛被使用的分类器有[人工神经网路](https://zh.wikipedia.org/wiki/人工神经网络)、[支持向量机](https://zh.wikipedia.org/wiki/支持向量机)、[最近邻居法](https://zh.wikipedia.org/wiki/最近鄰居法)、[高斯混合模型](https://zh.wikipedia.org/wiki/高斯混合模型)、[朴素贝叶斯方法](https://zh.wikipedia.org/wiki/朴素贝叶斯分类器)、[决策树](https://zh.wikipedia.org/wiki/决策树)和[径向基函数分类](https://zh.wikipedia.org/w/index.php?title=径向基函数分类&action=edit&redlink=1)。



**比如：**

我们给计算机一堆图片，并告诉他们那些是猫、哪些是狗。让计算机去分辨猫或狗，通过这种指引的方式让计算机学习我们是如何把这些图片数据对应到图片上所代表的的物体，并赋予这些图片猫或狗的标签。预测房屋的价格，股票的涨停同样可以用监督学期来实现。

神经网络同样是一种监督学习的方式

---



### 无监督学习（unsupervised learning）

**无监督学习**（英语：unsupervised learning）是[机器学习](https://zh.wikipedia.org/wiki/机器学习)的一种方法，没有给定事先标记过的训练范例，自动对输入的资料进行分类或分群。无监督学习的主要运用包含：[聚类分析](https://zh.wikipedia.org/wiki/聚类分析)（cluster analysis）、[关联规则](https://zh.wikipedia.org/wiki/关联规则学习)（association rule）、维度缩减（dimensionality reduce）。它是[监督式学习](https://zh.wikipedia.org/wiki/監督式學習)和[强化学习](https://zh.wikipedia.org/wiki/强化学习)等策略之外的一种选择。

一个常见的无监督学习是[数据聚类](https://zh.wikipedia.org/wiki/数据聚类)。在[人工神经网路](https://zh.wikipedia.org/wiki/人工神经网络)中，[生成对抗网络](https://zh.wikipedia.org/wiki/生成对抗网络)（GAN）、[自组织映射](https://zh.wikipedia.org/wiki/自组织映射)（SOM）和[适应性共振理论](https://zh.wikipedia.org/w/index.php?title=適應性共振理論&action=edit&redlink=1)（ART）则是最常用的非监督式学习。

ART模型允许丛集的个数可随著问题的大小而变动，并让使用者控制成员和同一个丛集之间的相似度分数，其方式为透过一个由使用者自定而被称为[警觉参数](https://zh.wikipedia.org/w/index.php?title=警覺參數&action=edit&redlink=1)的常数。ART也用于[模式识别](https://zh.wikipedia.org/wiki/模式识别)，如[自动目标辨识](https://zh.wikipedia.org/w/index.php?title=自動目標辨識&action=edit&redlink=1)和[数位信号处理](https://zh.wikipedia.org/wiki/數位信號處理)。第一个版本为"ART1"，是由卡本特和葛罗斯柏格所发展的。



**比如：**

我们给计算机一堆图片，但并**不**告诉他们那些是猫、哪些是狗。让计算机自己去分辨这些图片中的不同之处，自己去判断或分类。在这一种学习过程中，我们可以不用提供数据所对应的标签信息，计算机通过观察各种数据之间的特性，会发现这些特性背后的规律

---



### 半监督学习（Semi-supervised learning）

**半监督学习**是一种[机器学习](https://en.wikipedia.org/wiki/Machine_learning)方法，它在训练过程中将少量[标记数据](https://en.wikipedia.org/wiki/Labeled_data)与大量未标记数据相结合。半监督学习介于[无监督学习](https://en.wikipedia.org/wiki/Unsupervised_learning)（没有标记的训练数据）和[监督学习](https://en.wikipedia.org/wiki/Supervised_learning)（只有标记的训练数据）之间。它是[弱监督的](https://en.wikipedia.org/wiki/Weak_supervision)一个特例。

未标记数据与少量标记数据结合使用时，可以显着提高学习准确性。为学习问题获取标记数据通常需要熟练的人类代理（例如转录音频片段）或物理实验（例如确定蛋白质的 3D 结构或确定特定位置是否有油）。因此，与标记过程相关的成本可能会使大型、完全标记的训练集变得不可行，而未标记数据的获取相对便宜。在这种情况下，半监督学习具有很大的实用价值。半监督学习在机器学习和作为人类学习的模型方面也具有理论意义。

**它主要考虑如何利用少量有标签的样本和大量的没有标签样本进行训练和分类**

---



### 强化学习（Reinforcement learning）

**强化学习**（英语：Reinforcement learning，简称RL）是[机器学习](https://zh.wikipedia.org/wiki/机器学习)中的一个领域，强调如何基于[环境](https://zh.wikipedia.org/wiki/环境)而行动，以取得最大化的预期利益[[1\]](https://zh.wikipedia.org/wiki/强化学习#cite_note-1)。强化学习是除了[监督学习](https://zh.wikipedia.org/wiki/监督学习)和[非监督学习](https://zh.wikipedia.org/w/index.php?title=非监督学习&action=edit&redlink=1)之外的第三种基本的机器学习方法。与监督学习不同的是，强化学习不需要带标签的输入输出对，同时也无需对非最优解的精确地纠正。其关注点在于寻找探索（对未知领域的）和利用（对已有知识的）的平衡[[2\]](https://zh.wikipedia.org/wiki/强化学习#cite_note-kaelbling-2)，强化学习中的“探索-利用”的交换，在[多臂老虎机](https://zh.wikipedia.org/w/index.php?title=多臂老虎机&action=edit&redlink=1)问题和有限MDP中研究得最多。。

其灵感来源于心理学中的[行为主义](https://zh.wikipedia.org/wiki/行为主义)理论，即有机体如何在环境给予的奖励或惩罚的刺激下，逐步形成对刺激的预期，产生能获得最大利益的习惯性行为。这个方法具有普适性，因此在其他许多领域都有研究，例如[博弈论](https://zh.wikipedia.org/wiki/博弈论)、[控制论](https://zh.wikipedia.org/wiki/控制论)、[运筹学](https://zh.wikipedia.org/wiki/运筹学)、[信息论](https://zh.wikipedia.org/wiki/信息论)、仿真优化、[多智能体系统](https://zh.wikipedia.org/wiki/多智能体系统)、[群体智能](https://zh.wikipedia.org/wiki/群体智能)、[统计学](https://zh.wikipedia.org/wiki/统计学)以及[遗传算法](https://zh.wikipedia.org/wiki/遗传算法)。在运筹学和控制理论研究的语境下，强化学习被称作“近似动态规划”（approximate dynamic programming，ADP）。在[最优控制](https://zh.wikipedia.org/wiki/最优控制)理论中也有研究这个问题，虽然大部分的研究是关于最优解的存在和特性，并非是学习或者近似方面。在[经济学](https://zh.wikipedia.org/wiki/经济学)和[博弈论](https://zh.wikipedia.org/wiki/博弈论)中，强化学习被用来解释在[有限理性](https://zh.wikipedia.org/wiki/有限理性)的条件下如何出现平衡。

在机器学习问题中，环境通常被抽象为[马尔可夫决策过程](https://zh.wikipedia.org/wiki/马尔可夫决策过程)（Markov decision processes，MDP），因为很多强化学习算法在这种假设下才能使用[动态规划](https://zh.wikipedia.org/wiki/动态规划)的方法[[3\]](https://zh.wikipedia.org/wiki/强化学习#cite_note-3)。传统的动态规划方法和强化学习算法的主要区别是，后者不需要关于MDP的知识，而且针对无法找到确切方法的大规模MDP。[[4\]](https://zh.wikipedia.org/wiki/强化学习#cite_note-4)

比如：

在规划机器人的行为准则方面，(REINFORCEMENTLEARNING)  这种机器学习方法叫作“强化学习。也就是把计算机丢到了一个对于它完全陌生的环境或者让它完成一项从未接触过的任务，它会去尝试各种手段，最后让自己成功适应这个陌生环境，或者学会完成这件任务的方法途经。

比如我想训练机器人去投篮，我只需要给他一个球。并告诉他，你投进了我给你记一分，让它自己去尝试各种各样的投篮方法。在开始阶段，他的命中率可能非常低，不过他会像人类一样自己去总结投篮失败或成功的经验，最后达到很高的命中率。Google 的 AlphaGo 也就是应用了这一种学习的方式

---



### 遗传算法（Genetic Algorithm）

**遗传算法**（英语：Genetic Algorithm, **GA** ）是[计算数学](https://zh.wikipedia.org/wiki/计算数学)中用于解决[最佳化](https://zh.wikipedia.org/wiki/最佳化)的搜索[算法](https://zh.wikipedia.org/wiki/算法)，是[进化算法](https://zh.wikipedia.org/wiki/进化算法)的一种。进化算法最初是借鉴了[进化生物学](https://zh.wikipedia.org/wiki/进化生物学)中的一些现象而发展起来的，这些现象包括[遗传](https://zh.wikipedia.org/wiki/遗传)、[突变](https://zh.wikipedia.org/wiki/突变)、[自然选择](https://zh.wikipedia.org/wiki/自然选择)以及[杂交](https://zh.wikipedia.org/wiki/杂交)等等。

遗传算法通常实现方式为一种[计算机模拟](https://zh.wikipedia.org/wiki/计算机模拟)。对于一个最优化问题，一定数量的[候选解](https://zh.wikipedia.org/w/index.php?title=候选解&action=edit&redlink=1)（称为个体）可抽象表示为[染色体](https://zh.wikipedia.org/wiki/染色體_(遺傳演算法))，使[种群](https://zh.wikipedia.org/wiki/种群)向更好的解进化。传统上，解用[二进制](https://zh.wikipedia.org/wiki/二进制)表示（即0和1的串），但也可以用其他表示方法。进化从完全[随机](https://zh.wikipedia.org/wiki/随机)个体的种群开始，之后一代一代发生。在每一代中评价整个种群的[适应度](https://zh.wikipedia.org/wiki/适应度)，从当前种群中随机地选择多个个体（基于它们的适应度），通过自然选择和突变产生新的生命种群，该种群在算法的下一次迭代中成为当前种群。

遗传算法和强化学习类似，这一种方法是模拟我们熟知的进化理论，淘汰弱者；适者生存。通过这样的淘汰机制去选择最优的设计或模型，比如这位开发者所开发的计算机学会玩超级玛丽最开始的马里奥1代可能不久就牺牲了，不过系统会基于1代的马里奥随机盖成2代，淘汰掉比较弱的马利奥然后再次基于强者“繁衍和变异”

---



# 机器学习的流程

## 开发流程

![image-20211228220833060](doc/pic/README/image-20211228220833060.png)

1. 获取数据
2. 数据处理
3. 特征工程（将数据处理为算法能够使用的数据；特征就是特征值）
4. 机器学习算法进行训练（fit） --> 得到模型
5. 模型评估，如果模型不好返回第 2 步继续循环，直到模型评估比较好



## 机器框架和资料

明确：**机器学习算法是核心，数据和计算是基础**



**书籍：**

- 机器学习 - “西瓜书" - 周志华
- 统计学习方法 - 季航
- 深度学习 - “花书"



**库和框架：**

- 传统机器学习框架
    - sklearn	
- 深度学习框架
    - [TensorFlow](https://www.tensorflow.org/?hl=zh-cn)
    - theano
    - Caffe2
    - Pytorch
    - Chainer

---



# 特征工程（概念）

## 数据集

### 可用数据集

- Sklearn
    - Python语言的机器学习工具
    - Scikit-learn 包括许多知名的机器学习算法的实现
    - Scikit-learn文档完善，容易上手，丰富的API
- Kaggle
- UCI

![image-20211228230707442](doc/pic/README/image-20211228230707442.png)





## 特征工程介绍

特征工程是对数据的特征进行处理，能使一个算法得到最好的发挥，会直接影响到机器学习效果。

数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已

**工具：**

- `pandas` - 数据清洗，数据处理
- `sklearn` - 特征工程



**特征工程主要包括：**

- 特征抽取（提取）
- 特征预处理
- 特征降维



### 特征抽取（提取）

特征值化是为了让计算机更好的去理解数据；将任意数据（图像或文本）转换为可用于机器学习的数字特征

机器学习算法 --> 统计方法 --> 数学公式

- 字典特征提取（特征离散化）
- 文本特征提取
- 图像特征提取（深度学习）



### 特征预处理

**目标：**

- 了解数值型数据、类别型数据特点
- 应用 MinMaxScaler 实现对特征数据进行**归一化**
- 应用 StandardScaler 实现对特征数据进行**标准化**

![image-20220125164821285](doc/pic/README/image-20220125164821285.png)

可见，**特征预处理**就是通过一些转换函数将特征数据转换成更加适合算法模型的特征图数据过程



**特征预处理包含：**

- **归一化**

    通过对原始数据进行变换把数据映射到（默认[0, 1]）之间

- **标准化**



**为什么要进行归一化/标准化？**

- 简而言之就是要**统一数量级**，比如在做物理或数学题时我们要把数值的数量级或单位进行统一**使数据无量纲化**。
- 特征的单位或者大小相差较大，或者某特征的方差相比其他的特征要大出几个数量级，容易影响（支配）目标结果，使得一些算法无法学习到其它的特征

# sklearn

## 选择 sklearn 的机器学习算法

![ml_map](doc/pic/README/ml_map.png)



## sklean 数据集（datasets）

### 数据集 API 介绍

> 官方网址：
>
> https://scikit-learn.org/stable/modules/classes.html?highlight=dataset#module-sklearn.datasets

- `sklearn.datasets` - 数据集 API 
    - 加载流行数据集
    - `datasets.load_*()` - **获取（load）小**规模数据集，数据包在 datasets 中
    - `datasets.fetch_*(data_home=None)` - **加载（fetch）大**规模数据集。因为数据集很大，需要从网上下载。函数的第一个参数为 `data_home`，表示数据集下载的目录，默认是 `~/sklearn_learn_data`



### skleaen 小数据集

举例：

- `sklearn.datasets.load_iris()`

    加载并返回鸢尾花数据集

    | 名称         | 数量 |
    | ------------ | ---- |
    | 类别         | 3    |
    | 特征         | 4    |
    | 样本数量     | 150  |
    | 每个类别数量 | 50   |

    

- `sklearn.datasets.load_boston()`

    加载并返回波士顿数据集

    | 名称     | 数量 |
    | -------- | ---- |
    | 目标类别 | 5-50 |
    | 特征     | 13   |
    | 样本数量 | 506  |



### skleaen 大数据集

举例：

- `sklearn.datasets.fetch_20newsgroups(data_home=None, subset='train')`
    - `subset`： `train` 或 `test` 或 `all`，可选择要加载的数据集。
        -  `train` 表示仅加载训练集
        -  `test` 仅加载测试集
        -  `all` 加载训练集和测试集



### sklearn 数据集返回值

`load` 和 `fetch` 返回的数据类型是 `datasets.base.Bunch` 也就是**字典格式（带有键值对 key-value）**

- `data` - **特征数据数组**，是 `[n_sample * n_sample]` 的二维 numpy.ndarray 数组
- `target` - **标签数组**，是 n_samples 的一维 numpy.ndarray 数组
- `DESCR` - **数据描述**
- `feature_namees` - **特征名**；比如：手写数字、新闻数据、回归数据集
- `target_names` - **标签名**

```python
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
```



### 数据集的划分

Q：为什么要对数据进行划分？

A：因为我们在模型建立结束之后需要**对模型进行评估**，评估这些模型需要真实的不同数据，也就是说我们不能拿用于建立模型的数据进行模型的测试

**机器学习一般的数据集会划分为 2 个部分：**

- 训练数据：用于训练，构建**模型**
- 测试数据：在模型检验使用，用于**评估模型是否有效**

**划分比例：**

- 训练集：70% ~ 80%
- 测试集：20% ~ 30%

**数据集划分 API：**

`sklearn.model_selection.train_test_split(array, *options)`

- `X` 数据集的特征值

- `y` 数据集的目标值

- `test_size` 测试集的大小，一般为 float

- `random_state` 随机数种子，不同的种子会造成不同的随机采样结果。相同的种子得到的采样结果相同

- `return` 返回值：

    **`X` 代表特征值，`y` 代表目标值**

    **`train` 代表训练集，`test` 代表测试集**

    - 训练集的特征值 - `X_train`
    - 测试集的特征值 - `X_test`
    - 训练集的目标值 - `y_train`
    - 测试集的目标值 - `y_test`

**我们通常取数据的 20% 对模型进行评估（用作测试集）**，sklearn 对模型进行划分时默认是取总数据的 25% 用作测试（训练集）



### 借助 make 方法创造数据



- `datasets.make_XXXX` - 这用以 `make` 开头的代表可以自定义让它生成一些模拟数据，至于这些模拟的数据有多少个属性和多少个分类可以在方法的参数中指定



---



## 特征工程 - 特征提取

### sparse 稀疏矩阵

**sparse 稀疏矩阵将非零值按位置表示出来**

在类别很多的情况（比如特征值是`city`，它的类别就会有北京、上海、武汉、广东...） **one-hot 编码会出现 0 非常多的情况**；

而 **sparse 稀疏矩阵将将非零值按位置表示出来可以极大的节省内存、提高加载效率**

![image-20220124160937196](doc/pic/README/image-20220124160937196.png)

*左为稀疏矩阵，右为 one-hot 矩阵*

**sparse 稀疏矩阵的方法：**

- `sparse稀疏矩阵.toarray()` **返回**该稀疏矩阵转换为的二维数组





### 特征提取 API

```python
sklearn.feature_extraction
```



### 字典特征提取

**作用：对字典数据进行特征值化，对于字典中的类别可以转换为 one-hot 编码**

**应用场景：**

- **当特征值比较多的时候**
    1. 将数据集的特征转换为字典类型
    2. DicVectorizer 转换
- **本身拿到的数据就是字典类型**



> **sparse 稀疏矩阵：**
>
> 将非零值按位置表示出来
>
> 在类别很多的情况（比如特征值是`city`，它的类别就会有北京、上海、武汉、广东...） **one-hot 编码会出现 0 非常多的情况**；
>
> 而 **sparse 稀疏矩阵将将非零值按位置表示出来可以极大的节省内存、提高加载效率**
>
> ![image-20220124160937196](doc/pic/README/image-20220124160937196.png)
>
> *左为稀疏矩阵，右为 one-hot 矩阵*



**转换器类 API：**

**`sklearn.feature_extraction.DicVectorizer(sparse=True, ...)`** 其中 sparse 代表 sparse 稀疏矩阵

​	**实例化对象后，可以调用以下方法：**

- `DicVectorizer.fit_transform(字典或包含字典的迭代器)` ***返回值：sparse 矩阵***
- `DicVectorizer.inverse_transform(array 数组或 sparse 矩阵)`  ***返回值：转换之前的数据格式***
- `DicVectorizer.get_feature_names_out()` ***返回值：类别名称***



**比如：**

我们对一下数据进行特征提取：

```python
    data = [
        {'city': '北京', 'temperature': 100},
        {'city': '上海', 'temperature': 60},
        {'city': '深圳', 'temperature': 30}
    ]
```



当以 one-hot 编码表示时（`feature_extraction.DictVectorizer(sparse=False)`）：

```python
data_fit:
 [[  0.   1.   0. 100.]
 [  1.   0.   0.  60.]
 [  0.   0.   1.  30.]]
```



当以 sparse 矩阵表示时（`feature_extraction.DictVectorizer(sparse=True)`）：

```python
data_fit:
   (0, 1)	1.0
  (0, 3)	100.0
  (1, 0)	1.0
  (1, 3)	60.0
  (2, 2)	1.0
  (2, 3)	30.0
```



---



### 文本特征提取

**作用：对文本数据进行特征值化**



#### CountVectorizer 统计样本特征词出现频率（个数）

**转换器类 API：**

**`sklearn.feature_extraction.text.CountVectorizer(stop_words=[])`** 返回词频矩阵，`stop_words` 是停用词列表，指的是哪些词不纳入统计

注意，这个 API 只能对英文有较好的分析，因为是用空格作为词与词之间的分隔，所以除非中文的各个词用空格分开，否则无法分析，并且这里不支持单个中文字！

​	**实例化对象后，可以调用以下方法：**

- `CountVectorizer.fit_transform(文本或包含文本字符串的可迭代对象)` ***返回值：sparse 矩阵***
- `CountVectorizer.inverse_transform(array 数组或 sparse 矩阵)`  ***返回值：转换之前的数据格式***
- `CountVectorizer.get_feature_names_out()` ***返回值：单词列表***



例子：

- 对英文进行特征词词频提取

```python
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
    print('data_fit:\n', data_fit.toarray())  # 【重点】对于 sparse 矩阵，内部的 `.toarray()` 可以返回一个对应的二维数据
    print('特征名字:\n', transfer.get_feature_names_out())
    
>>> 输出
data_fit:
 [[0 1 1 1 0 1 1 0]
 [1 1 1 0 1 1 0 1]]
特征名字:
 ['dislike' 'is' 'life' 'like' 'long' 'python' 'short' 'too']
```



- 对中文进行特征词词频提取

```python

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
    
>>> 输出
data_fit:
 [[1 1 0]
 [0 1 1]]
特征名字:
 ['北京' '天安门' '太阳']
```



#### 统计中文样本中特征词频率（借助 jieba）

由于中文的特殊性，需要先对样本进行语义分析，进行分词，然后才能进行特征词频率的分析

```python
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

```



#### TF-IDF 文本特征提取（关键词）

TF-IDF 指的就是重要程度，TF-IDF 的主要思想是：**如果某个词或短语在一篇文章中出现的概率高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力**，适合用来分类

TF-IDF 作用：用以评估一字词对于个文件集或一个语料库中的其中一份文件的重要程度

简而言之，TF-IDF 就查找关键词，TF-IDF 的值越大，关键词的可能性越高

**计算方法：**

- `TF` - 词频（Term Frequency）指的是**某一个给定的词语在该文件中出现的频率**

- `IDF` - 逆向文档频率（Inverse Document Frequency）是个词语普遍重要性的度量。

    **某一特定词语的 IDF，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取以10为底的对数得到**

 **公式**： $ tfidf_i,_j = tf_i,_j * idf_i$，最终得到的结果就可以理解为重要程度

例子：

```apl
比如对两个词：“经济” 和 “非常”
条件：
	语料库中有 1000 篇文章
	100 篇文章中有 “非常”
	10 篇文章中有 “经济”
	
那么：
【一】
	对于一个有 100 个词语的 文章A，出现了 10 次 “经济”
	TF：10 / 100 = 0.1
	IDF：lg(1000 / 10) = 2
	TF-IDF = TF * IDF = 0.1 * 2 = 0.2
	
【二】
	对于一个有 100 个词语的 文章A，出现了 10 次 “非常”
	TF：10 / 100 = 0.1
	IDF：lg(1000 / 100) = 1
	TF-IDF = TF * IDF = 0.1 * 1 = 0.1
	
得到：
	“经济”的 TF-IDF = 0.2
	“非常”的 TF-IDF = 0.1
	因为 0.2 > 0.1 所以说 “经济” 这个词对最终的分类有比较大的影响
```



**转换器类 API：**

**`sklearn.feature_extraction.text.TfidfVectorizer(stop_words=[])`** 返回词频矩阵，`stop_words` 是停用词列表，指的是哪些词不纳入统计

注意，这个 API 只能对英文有较好的分析，因为是用空格作为词与词之间的分隔，所以除非中文的各个词用空格分开，否则无法分析，并且这里不支持单个中文字！

​	**实例化对象后，可以调用以下方法：**

- `TfidfVectorizer.fit_transform(文本或包含文本字符串的可迭代对象)` ***返回值：sparse 矩阵***
- `TfidfVectorizer.inverse_transform(array 数组或 sparse 矩阵)`  ***返回值：转换之前的数据格式***
- `TfidfVectorizer.get_feature_names_out()` ***返回值：单词列表***



## 特征工程 - 特征预处理

### 特征预处理 API

```python
sklearn.perprocessing
```



### 归一化

**定义：**

​	通过对原始数据进行变换把数据映射到固定区间（默认[0, 1]）之内

**公式：**

![image-20220125171413298](doc/pic/README/image-20220125171413298.png)

- 对于每一列， *max* 为一列的最大值，*min* 为一列的最小值
- *mx, mi* 分别指定映射到区间的最小值或最大值（*默认 mx=1, mi=0*）
- *X''* 为最终结果



**缩放器 API：**

`sklearn.perprocessing.MinMaxScaler(feature_range=(0, 1), ...)`

​	**实例化后可以调用：**

- `MinMaxScaler.fit_transform(numpy-array 格式的二维数组数据[n_samples 类别, n_features 特征值])` ***返回值：归一化后形状相同的 array***

    



### 标准化

























---

---











