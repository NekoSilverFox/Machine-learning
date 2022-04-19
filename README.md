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

<img src="doc/pic/README/image-20211228165701734.png" alt="image-20211228165701734" style="zoom:50%;" />

- 机器学习和人工智能，深度学习的关系

    机器学习是人工智能的一个实现途径

    深度学习是机器学习的一个方法（神经网络）发展而来

<img src="doc/pic/README/image-20211228171230823.png" alt="image-20211228171230823" style="zoom:50%;" />



## 什么是机器学习

机器学习是从数据中自动分析获得模型，并利用模型对未知数据进行预测。

也就是说，通过**数据**来得到一个**模型**，然后再拿着这个模型去**预测**

<img src="doc/pic/README/image-20211228171519821.png" alt="image-20211228171519821" style="zoom:50%;" />

 

## 机器学习应用场景

<img src="doc/pic/README/image-20211228170732036.png" alt="image-20211228170732036" style="zoom:50%;" />

- 用在挖掘、预测领域：

    ​	应用场景：店铺销量预测、量化投资、广告推荐、企业客户分类、SQL语句安全检测分类...

    

- 用在图像领域：

    ​	应用场景：街道交通标志检测、人脸识别等等

    

- 用在自然语言处理领域：

    ​	应用场景：文本分类、情感分析、自动聊天、文本检测等等



---



# 机器学习的算法

## 算法的大致分类

| 特征值                     | 目标值                           | 问题描述                 | 举例                                               |
| -------------------------- | -------------------------------- | ------------------------ | -------------------------------------------------- |
| （比如：猫狗的图片）       | **类别**（比如：猫还是狗）       | **分类问题**（监督学习） | k-临近算法、贝叶斯分类、决策树与随机森林、逻辑回归 |
| （比如：房屋面积、位置等） | **连续型数据**（比如：房屋价格） | **回归问题**（监督学习） | 线性回归、岭回归                                   |
|                            | **没有目标值**                   | **聚类**（无监督学习）   | k-means                                            |
|                            |                                  |                          |                                                    |

<img src="doc/pic/README/image-20211228175206865.png" alt="image-20211228175206865" style="zoom:50%;" />



**举例：**

- 预测明天的气温是多少度？

    ​	回归问题

    

- 预测明天是阴、晴还是雨？

    ​	分类问题

    

- 人脸年龄预测？

    ​	回归或分类，取决如如何定义年龄

    

- 人脸识别？

    ​	分类





## 机器学习类型（算法）描述

| 类型       | 特点                               | 定义或是由                                                   | 代表                                                         |
| ---------- | ---------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 监督学习   | 有数据、有标签                     | **输入的数据是由特征值和目标值**所构成。**函数的输出可以是一个连续的值（称为回归）**或是**输出是有限个离散值（称作分类）** | **分类：**<br />k-临近算法、贝叶斯分类、决策树与随机森林、逻辑回归<br /><br />**回归：**<br />线性回归、岭回归 |
| 非监督学习 | 有数据、无标签                     | **输入的数据是由特征值和目标值**所构成                       | 聚类：<br />k-means                                          |
| 半监督学习 | 结合监督学习和非监督学习           |                                                              |                                                              |
| 强化学习   | 从经验中总结提升                   |                                                              |                                                              |
| 遗传算法   | 和强化学习类似，适者生存不适者淘汰 |                                                              |                                                              |



### 监督学习

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



### 无监督学习

**无监督学习**（英语：unsupervised learning）是[机器学习](https://zh.wikipedia.org/wiki/机器学习)的一种方法，没有给定事先标记过的训练范例，自动对输入的资料进行分类或分群。无监督学习的主要运用包含：[聚类分析](https://zh.wikipedia.org/wiki/聚类分析)（cluster analysis）、[关联规则](https://zh.wikipedia.org/wiki/关联规则学习)（association rule）、维度缩减（dimensionality reduce）。它是[监督式学习](https://zh.wikipedia.org/wiki/監督式學習)和[强化学习](https://zh.wikipedia.org/wiki/强化学习)等策略之外的一种选择。

一个常见的无监督学习是[数据聚类](https://zh.wikipedia.org/wiki/数据聚类)。在[人工神经网路](https://zh.wikipedia.org/wiki/人工神经网络)中，[生成对抗网络](https://zh.wikipedia.org/wiki/生成对抗网络)（GAN）、[自组织映射](https://zh.wikipedia.org/wiki/自组织映射)（SOM）和[适应性共振理论](https://zh.wikipedia.org/w/index.php?title=適應性共振理論&action=edit&redlink=1)（ART）则是最常用的非监督式学习。

ART模型允许丛集的个数可随著问题的大小而变动，并让使用者控制成员和同一个丛集之间的相似度分数，其方式为透过一个由使用者自定而被称为[警觉参数](https://zh.wikipedia.org/w/index.php?title=警覺參數&action=edit&redlink=1)的常数。ART也用于[模式识别](https://zh.wikipedia.org/wiki/模式识别)，如[自动目标辨识](https://zh.wikipedia.org/w/index.php?title=自動目標辨識&action=edit&redlink=1)和[数位信号处理](https://zh.wikipedia.org/wiki/數位信號處理)。第一个版本为"ART1"，是由卡本特和葛罗斯柏格所发展的。



**比如：**

我们给计算机一堆图片，但并**不**告诉他们那些是猫、哪些是狗。让计算机自己去分辨这些图片中的不同之处，自己去判断或分类。在这一种学习过程中，我们可以不用提供数据所对应的标签信息，计算机通过观察各种数据之间的特性，会发现这些特性背后的规律

---



### 半监督学习

**半监督学习**是一种[机器学习](https://en.wikipedia.org/wiki/Machine_learning)方法，它在训练过程中将少量[标记数据](https://en.wikipedia.org/wiki/Labeled_data)与大量未标记数据相结合。半监督学习介于[无监督学习](https://en.wikipedia.org/wiki/Unsupervised_learning)（没有标记的训练数据）和[监督学习](https://en.wikipedia.org/wiki/Supervised_learning)（只有标记的训练数据）之间。它是[弱监督的](https://en.wikipedia.org/wiki/Weak_supervision)一个特例。

未标记数据与少量标记数据结合使用时，可以显着提高学习准确性。为学习问题获取标记数据通常需要熟练的人类代理（例如转录音频片段）或物理实验（例如确定蛋白质的 3D 结构或确定特定位置是否有油）。因此，与标记过程相关的成本可能会使大型、完全标记的训练集变得不可行，而未标记数据的获取相对便宜。在这种情况下，半监督学习具有很大的实用价值。半监督学习在机器学习和作为人类学习的模型方面也具有理论意义。

**它主要考虑如何利用少量有标签的样本和大量的没有标签样本进行训练和分类**

---



### 强化学习

**强化学习**（英语：Reinforcement learning，简称RL）是[机器学习](https://zh.wikipedia.org/wiki/机器学习)中的一个领域，强调如何基于[环境](https://zh.wikipedia.org/wiki/环境)而行动，以取得最大化的预期利益[[1\]](https://zh.wikipedia.org/wiki/强化学习#cite_note-1)。强化学习是除了[监督学习](https://zh.wikipedia.org/wiki/监督学习)和[非监督学习](https://zh.wikipedia.org/w/index.php?title=非监督学习&action=edit&redlink=1)之外的第三种基本的机器学习方法。与监督学习不同的是，强化学习不需要带标签的输入输出对，同时也无需对非最优解的精确地纠正。其关注点在于寻找探索（对未知领域的）和利用（对已有知识的）的平衡[[2\]](https://zh.wikipedia.org/wiki/强化学习#cite_note-kaelbling-2)，强化学习中的“探索-利用”的交换，在[多臂老虎机](https://zh.wikipedia.org/w/index.php?title=多臂老虎机&action=edit&redlink=1)问题和有限MDP中研究得最多。。

其灵感来源于心理学中的[行为主义](https://zh.wikipedia.org/wiki/行为主义)理论，即有机体如何在环境给予的奖励或惩罚的刺激下，逐步形成对刺激的预期，产生能获得最大利益的习惯性行为。这个方法具有普适性，因此在其他许多领域都有研究，例如[博弈论](https://zh.wikipedia.org/wiki/博弈论)、[控制论](https://zh.wikipedia.org/wiki/控制论)、[运筹学](https://zh.wikipedia.org/wiki/运筹学)、[信息论](https://zh.wikipedia.org/wiki/信息论)、仿真优化、[多智能体系统](https://zh.wikipedia.org/wiki/多智能体系统)、[群体智能](https://zh.wikipedia.org/wiki/群体智能)、[统计学](https://zh.wikipedia.org/wiki/统计学)以及[遗传算法](https://zh.wikipedia.org/wiki/遗传算法)。在运筹学和控制理论研究的语境下，强化学习被称作“近似动态规划”（approximate dynamic programming，ADP）。在[最优控制](https://zh.wikipedia.org/wiki/最优控制)理论中也有研究这个问题，虽然大部分的研究是关于最优解的存在和特性，并非是学习或者近似方面。在[经济学](https://zh.wikipedia.org/wiki/经济学)和[博弈论](https://zh.wikipedia.org/wiki/博弈论)中，强化学习被用来解释在[有限理性](https://zh.wikipedia.org/wiki/有限理性)的条件下如何出现平衡。

在机器学习问题中，环境通常被抽象为[马尔可夫决策过程](https://zh.wikipedia.org/wiki/马尔可夫决策过程)（Markov decision processes，MDP），因为很多强化学习算法在这种假设下才能使用[动态规划](https://zh.wikipedia.org/wiki/动态规划)的方法[[3\]](https://zh.wikipedia.org/wiki/强化学习#cite_note-3)。传统的动态规划方法和强化学习算法的主要区别是，后者不需要关于MDP的知识，而且针对无法找到确切方法的大规模MDP。[[4\]](https://zh.wikipedia.org/wiki/强化学习#cite_note-4)

比如：

在规划机器人的行为准则方面，(REINFORCEMENTLEARNING)  这种机器学习方法叫作“强化学习。也就是把计算机丢到了一个对于它完全陌生的环境或者让它完成一项从未接触过的任务，它会去尝试各种手段，最后让自己成功适应这个陌生环境，或者学会完成这件任务的方法途经。

比如我想训练机器人去投篮，我只需要给他一个球。并告诉他，你投进了我给你记一分，让它自己去尝试各种各样的投篮方法。在开始阶段，他的命中率可能非常低，不过他会像人类一样自己去总结投篮失败或成功的经验，最后达到很高的命中率。Google 的 AlphaGo 也就是应用了这一种学习的方式

---



### 遗传算法

**遗传算法**（英语：Genetic Algorithm, **GA** ）是[计算数学](https://zh.wikipedia.org/wiki/计算数学)中用于解决[最佳化](https://zh.wikipedia.org/wiki/最佳化)的搜索[算法](https://zh.wikipedia.org/wiki/算法)，是[进化算法](https://zh.wikipedia.org/wiki/进化算法)的一种。进化算法最初是借鉴了[进化生物学](https://zh.wikipedia.org/wiki/进化生物学)中的一些现象而发展起来的，这些现象包括[遗传](https://zh.wikipedia.org/wiki/遗传)、[突变](https://zh.wikipedia.org/wiki/突变)、[自然选择](https://zh.wikipedia.org/wiki/自然选择)以及[杂交](https://zh.wikipedia.org/wiki/杂交)等等。

遗传算法通常实现方式为一种[计算机模拟](https://zh.wikipedia.org/wiki/计算机模拟)。对于一个最优化问题，一定数量的[候选解](https://zh.wikipedia.org/w/index.php?title=候选解&action=edit&redlink=1)（称为个体）可抽象表示为[染色体](https://zh.wikipedia.org/wiki/染色體_(遺傳演算法))，使[种群](https://zh.wikipedia.org/wiki/种群)向更好的解进化。传统上，解用[二进制](https://zh.wikipedia.org/wiki/二进制)表示（即0和1的串），但也可以用其他表示方法。进化从完全[随机](https://zh.wikipedia.org/wiki/随机)个体的种群开始，之后一代一代发生。在每一代中评价整个种群的[适应度](https://zh.wikipedia.org/wiki/适应度)，从当前种群中随机地选择多个个体（基于它们的适应度），通过自然选择和突变产生新的生命种群，该种群在算法的下一次迭代中成为当前种群。

遗传算法和强化学习类似，这一种方法是模拟我们熟知的进化理论，淘汰弱者；适者生存。通过这样的淘汰机制去选择最优的设计或模型，比如这位开发者所开发的计算机学会玩超级玛丽最开始的马里奥1代可能不久就牺牲了，不过系统会基于1代的马里奥随机盖成2代，淘汰掉比较弱的马利奥然后再次基于强者“繁衍和变异”

---



# 机器学习的流程

## 开发流程

<img src="doc/pic/README/image-20211228220833060.png" alt="image-20211228220833060" style="zoom:50%;" />

1. 获取数据
2. 数据处理（处理缺失值等）
3. 特征工程（将数据处理为算法能够使用的数据；特征就是特征值）
4. 使用机器学习算法进行训练（fit） --> 得到模型
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

在明确了机器学习的相关概念后，我们知道机器学习需要有数据来训练模型，那么我们的数据又是如何构成的呢？格式又是如何？

- **结构：特征值 + 目标值**

    - 对于**每一行数据**我们可以称之为**样本**
    - 有些数据集可以没有目标值

    

    比如下图的房屋数据：

    **特征值**就可以看作是房屋的面积、位置、楼层、朝向

    **目标值**就可以看做是房屋的价格

    <img src="doc/pic/README/image-20211228172647016.png" alt="image-20211228172647016" style="zoom:50%;" />



- **可用数据集**

    - Sklearn
        - Python语言的机器学习工具
        - Scikit-learn 包括许多知名的机器学习算法的实现
        - Scikit-learn文档完善，容易上手，丰富的API

    - Kaggle

    - UCI


<img src="doc/pic/README/image-20211228230707442.png" alt="image-20211228230707442" style="zoom:50%;" />





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



---

### 特征抽取（提取）

特征值化是为了让计算机更好的去理解数据；将任意数据（图像或文本）转换为可用于机器学习的数字特征

机器学习算法 --> 统计方法 --> 数学公式

- 字典特征提取（特征离散化）
- 文本特征提取
- 图像特征提取（深度学习）

---

### 特征预处理

**目标：**

- 了解数值型数据、类别型数据特点
- 应用 `MinMaxScaler` 实现对特征数据进行**归一化**
- 应用 `StandardScaler` 实现对特征数据进行**标准化**

<img src="doc/pic/README/image-20220125164821285.png" alt="image-20220125164821285" style="zoom:50%;" />

可见，**特征预处理**就是通过一些转换函数将特征数据转换成更加适合算法模型的特征图数据过程



**特征预处理包含：**

- **归一化**

    通过对原始数据进行变换把数据映射到（默认[0, 1]）之间

    

- **标准化**



**为什么要进行归一化/标准化？**

- 简而言之就是要**统一数量级**，比如在做物理或数学题时我们要把数值的数量级或单位进行统一**使数据无量纲化**。
- 特征的单位或者大小相差较大，或者某特征的方差相比其他的特征要大出几个数量级，容易影响（支配）目标结果，使得一些算法无法学习到其它的特征



---





# sklearn

## 选择 sklearn 的机器学习算法

![ml_map](doc/pic/README/ml_map.png)

<img src="doc/pic/README/image-20220217203643291.png" alt="image-20220217203643291" style="zoom:50%;" />

---



## sklearn-API 汇总

**数据集部分：**

- **【数据结构】`database.base.Bunch`**

    `load` 和 `fetch` **返回**的数据类型是 `datasets.base.Bunch` 也就是一种**基于字典的格式（带有键值对 key-value）** ，可以使用`dict['key']` 和 `bunch.key` 两种方式获得数值

    - `data` - **特征数据数组**，是 `[n_sample * n_sample]` 的二维 **numpy.ndarray** 数组
    - `target` - **目标值数组**，是 n_samples 的一维 **numpy.ndarray** 数组
    - `DESCR` - **数据描述**
    - `feature_namees` - **特征名**；比如：手写数字、新闻数据、回归数据集
    - `target_names` - **目标值名**

    

- **数据集加载/生成**

    ```pytohn
    sklearn.database
    ```

    | 函数/方法                                                    | 描述                                                         |
    | ------------------------------------------------------------ | ------------------------------------------------------------ |
    | `database.load_*()`                                          | **返回**小数据集，**返回**的数据类型是 `datasets.base.Bunch` |
    | `database.fetch_*(datahome=数据集保存到, subset='all')`      | 先从网络下载，再**返回**大数据集；<br />`data_home=`数据集保存路径，默认 `~/sklearn_learn_data`<br />`subset=`加载哪些集合：`all`所有；`train` 训练集；`test`测试集 |
    | `dataset.make_*(n_samples=数据量, n_features=特征数, n_targets=目标值数, noise=噪声)` | **返回**生成的随机的模拟数据                                 |

    

- **数据集划分：**

    ```python
    sklearn.model_selection
    ```

    `x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x=特征值, y=目标值, test_size=0.25, random_state=随机种子)`

    **务必要注意返回值的顺序**

    - `X` 特征值

    - `y` 目标值

    - `test_size` 测试集的占比，一般为 float，默认为 0.25

    - `random_state` 随机数种子，不同的种子会造成不同的随机采样结果。相同的种子得到的采样结果相同

    - `return` 返回值：

        **`X` 代表特征值，`y` 代表目标值**
    
        **`train` 代表训练集，`test` 代表测试集**
    
        1. 特征值（训练集） - `X_train`
        2. 特征值（测试集） - `X_test`
        3. 目标值（训练集） - `y_train`
        4. 目标值（测试集） - `y_test`
    
    **我们通常取数据的 20% 对模型进行评估（用作测试集）**，sklearn 对模型进行划分时默认是取总数据的 25% 用作测试（训练集）



---



**特征工程部分：**

- **sparse** 稀疏矩阵

    | 方法               | 描述                                             |
    | ------------------ | ------------------------------------------------ |
    | `sparse.toarray()` | **返回** sparse 稀疏矩阵对应的 **numpy.ndarray** |
    |                    |                                                  |
    |                    |                                                  |



- **特征提取**

    ```python
    sklearn.feature_extraction
    ```

    1. **根据数据的格式调用对应的 API，得到相应的转换器对象**

        | 转换器                                                       | 描述                                                         |
        | ------------------------------------------------------------ | ------------------------------------------------------------ |
        | `dic_transfer =`<br />` sklearn.feature_extraction.DicVectorizer(sparse=True, ...)` | **返回** ***字典特征提取***的转换器对象<br />`sparse`是否为稀疏矩阵，默认为 True |
        | `count_transfer = `<br />`sklearn.feature_extraction.text.CountVectorizer(stop_words=[停用词列表])` | **返回**词频对应的离散化后的矩阵（词频矩阵）                 |
        | `tfidf_transfer = `<br />`sklearn.feature_extraction.text.TfidfVectorizer(stop_words=[停用词列表])` | **返回**文本中关键词对应的离散化后的矩阵                     |

        

    2. **调用转换器中的方法**

        | 方法                                                         | 描述                             |
        | ------------------------------------------------------------ | -------------------------------- |
        | `*_transfer.fit_transform(对应数据或包含对应数据的可迭代对象)` | **返回**字典对应的离散化后的矩阵 |
        | `*_transfer.inverse_transform(ndarray 数组或 sparse 矩阵)`   | **返回**转换之前的数据格式       |
        | `*_transfer.get_feature_names_out()`                         | **返回**特征值名称列表           |


    

- **特征预处理**

    ```python
    sklearn.preprocessing
    ```

    | 方法 | 描述 |
    | ---- | ---- |
    |      |      |
    |      |      |
    |      |      |

    

## sklean 数据集

### 数据集 API 介绍

> 官方网址：
>
> https://scikit-learn.org/stable/modules/classes.html?highlight=dataset#module-sklearn.datasets

- `sklearn.datasets` - 数据集 API 
  
    **加载流行数据集：**
    
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

`load` 和 `fetch` **返回**的数据类型是 `datasets.base.Bunch` 也就是一种**基于字典的格式（带有键值对 key-value）** ，可以使用`dict['key']` 和 `bunch.key` 两种方式获得数值

- `data` - **特征数据数组**，是 `[n_sample * n_sample]` 的二维 **numpy.ndarray** 数组
- `target` - **目标值数组**，是 n_samples 的一维 **numpy.ndarray** 数组
- `DESCR` - **数据描述**
- `feature_namees` - **特征名**；比如：手写数字、新闻数据、回归数据集
- `target_names` - **目标值名**

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

`x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x=特征值, y=目标值, test_size=0.25, random_state=随机种子)`

- `X` 数据集的特征值

- `y` 数据集的目标值

- `test_size` 测试集的大小，一般为 float，默认为 0.25

- `random_state` 随机数种子，不同的种子会造成不同的随机采样结果。相同的种子得到的采样结果相同

- `return` 返回值：

    **`X` 代表特征值，`y` 代表目标值**

    **`train` 代表训练集，`test` 代表测试集**

    1. 特征值（训练集） - `X_train`
    2. 特征值（测试集） - `X_test`
    3. 目标值（训练集） - `y_train`
    4. 目标值（测试集） - `y_test`

**我们通常取数据的 20% 对模型进行评估（用作测试集）**，sklearn 对模型进行划分时默认是取总数据的 25% 用作测试（训练集）



### 借助 make 方法创造数据



- `datasets.make_XXXX` - 这用以 `make` 开头的代表可以自定义让它生成一些模拟数据，至于这些模拟的数据有多少个属性和多少个分类可以在方法的参数中指定



---



## 特征工程 - 特征提取

### sparse 稀疏矩阵

**sparse 稀疏矩阵将非零值按位置表示出来**

在类别很多的情况（比如特征值是`city`，它的类别就会有北京、上海、武汉、广东...） **one-hot 编码会出现 0 非常多的情况**；

而 **sparse 稀疏矩阵将将非零值按位置表示出来可以极大的节省内存、提高加载效率**

<img src="doc/pic/README/image-20220124160937196.png" alt="image-20220124160937196" style="zoom:50%;" />

*左为稀疏矩阵，右为 one-hot 矩阵*



**sparse 稀疏矩阵的方法：**

- `sparse稀疏矩阵.toarray()` **返回**该稀疏矩阵转换为的二维数组



---



### 特征提取 API

```python
sklearn.feature_extraction
```



### 字典特征提取

**作用：对字典数据进行特征值化，对于字典中的类别可以转换为 one-hot 编码**（数据离散化）



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
> <img src="doc/pic/README/image-20220124160937196.png" alt="image-20220124160937196" style="zoom:50%;" />
>
> *左为稀疏矩阵，右为 one-hot 矩阵*



**转换器类 API：**

**`transfer = sklearn.feature_extraction.DicVectorizer(sparse=True, ...)`** 其中 sparse 代表 sparse 稀疏矩阵

​	**实例化对象后，可以调用以下方法：**

- `DicVectorizer.fit_transform(字典或包含字典的迭代器)` **返回**sparse 矩阵
- `DicVectorizer.inverse_transform(array 数组或 sparse 矩阵)`  **返回**转换之前的数据格式
- `DicVectorizer.get_feature_names_out()` **返回**特征值名称



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

**`sklearn.feature_extraction.text.TfidfVectorizer(stop_words=[])`** **返回**词频矩阵

- `stop_words` 是停用词列表，指的是哪些词不纳入统计



注意，这个 API 只能对英文有较好的分析，因为是用空格作为词与词之间的分隔，所以除非中文的各个词用空格分开，否则无法分析，并且这里不支持单个中文字！

​	**实例化对象后，可以调用以下方法：**

- `TfidfVectorizer.fit_transform(文本或包含文本字符串的可迭代对象)` ***返回值：sparse 矩阵***
- `TfidfVectorizer.inverse_transform(array 数组或 sparse 矩阵)`  ***返回值：转换之前的数据格式***
- `TfidfVectorizer.get_feature_names_out()` ***返回值：单词列表***



## 特征工程 - 特征预处理

特征工程是使用专业背景知识和技巧处理数据，使得特征能在机器学习算法上发挥更好的作用的过程。会直接影响机器学习的效果



### 特征预处理 API

```python
sklearn.preprocessing
```



### 归一化

**定义：**

​	通过对原始数据进行变换把数据映射到固定区间（默认[0, 1]）之内



**公式：**

<img src="doc/pic/README/image-20220125171413298.png" alt="image-20220125171413298" style="zoom:33%;" />

- 对于每一**列**， *max* 为一列的最大值，*min* 为一列的最小值
- *mx, mi* 分别指定**映射到区间的最小值或最大值**（*默认 mx=1, mi=0*）
- *X''* 为最终结果



**缩放器 API：**

`sklearn.perprocessing.MinMaxScaler(feature_range=(0, 1), ...)`

​	**实例化后可以调用：**

- `MinMaxScaler.fit_transform(ndarray格式的二维数组数据[n_samples 类别, n_features 特征值])` **返回**归一化后形状相同的 ndarray

    

**问题：**

如果出现缺失值或者异常值（最大值和最小值是一个非常大的数），归一化会非常收到异常点的影响。所以这种方法的鲁棒性较差



---

### 标准化

**定义：**

​	通过对原始数据进行变换把数据变换到**均值为 0，标准差为 1 的范围**内。**适合用于现代嘈杂的大数据场景**



**公式：**

$$X ^ { \prime } = \frac { x - m e an } { \sigma }  $$

- mean 平均值
- $\sigma$ 标准差



- 对于归一化来说：如果出现异常点，影响了最大值和最小值，那么结果当然会发生变化
- 对于标准化来说：如果出现异常点，由于具有一定的数据来量，少量的异常点对于平均值的影响并不大，从而方差改变较小



**缩放器 API：**

`sklearn.perprocessing.StandardScaler(feature_range=(0, 1), ...)`

​	**实例化后可以调用：**

- `StandardScaler.fit_transform(ndarray格式的二维数组数据[n_samples 类别, n_features 特征值])` **返回**归一化后形状相同的 ndarray

  ​    









---

---



## 特征工程 - 特征降维

**降维**指的是在某些限定条件下，**降低随机变量（特征的个数）**，得到**一组”不相关“主变量的过程**。

在进行训练的时候，我们都是使用特征进行学习。如果特征本身存在问题或者两者的相关性较强，对于算法学习的预测会影响较大



**降维的对象**：二维数组

**降低的维度是什么**：特征的个数（列数）



Q:什么是相关特征

A:比如相对湿度与降雨量之间的关系，**特征降维就是要减少这种相关性**



---

### 相关系数

主要实现方式有：

- 皮尔逊相关系数
- 斯皮尔曼相关系数



---



#### 皮尔逊相关系数 (Rank IC)

英语：Pearson Correlation Coefficient

**作用：**反应变量与变量之间相关关系密切程度的统计指标



**公式：**
$$
r=\frac{n \sum x y-\sum x \sum y}{\sqrt{n \sum x^{2}-\left(\sum x\right)^{2}} \sqrt{n \sum y^{2}-\left(\sum y\right)^{2}}}
$$
**特点**：

**相关系数的值介于 [–1, +1]**。其性质如下：

- **当 $r>0$ 时，表示两变量【正】相关；**
- **当 $r<0$ 时，表示两变量【负】相关**
- 当 $|r|=1$ 时，表示两变量为**完全相关**
- 当 $r=0$ 时，表示两变量间**无相关关系**
- **当 $0<|r|<1$ 时，表示两变量存在一定程度的相关。且 $|r|$ 越接近1，两变量间线性关系越密切；$|r|$ 越接近于 0，表示两变量的线性相关越弱**
- **一般可按三级划分：**
    - **$|r|<0.4$ 为低度相关；**
    - **$0.4≤|r|<0.7$ 为显著性相关；**
    - **$0.7≤|r|<1$ 为高度线性相关**




**API**

`from scipy.stats import pearsonr`

- x : (N,) array_like
- y : (N,) array_like 
- Returns: (Pearson’s correlation coefficient, p-value)

```python
from scipy.stats import pearsonr  # 导包

def pearson_correlation_coefficient():
    """
    皮尔逊相关系数
    :return:
    """
    x1 = [12.5, 15.3, 23.2, 26.4, 33.5, 34.4, 39.4, 45.2, 55.4, 60.9]
    x2 = [21.2, 23.9, 32.9, 34.1, 42.5, 43.2, 49.0, 52.8, 59.4, 63.5]

    r = pearsonr(x=x1, y=x2)
    print(r)  # 输出：(0.9941983762371884, 4.922089955456964e-09)
```





**计算举例**：比如说我们计算年广告费投入与月均销售额

<img src="file:///Users/fox/%E9%9B%AA%E7%8B%B8%E7%9A%84%E6%96%87%E4%BB%B6/%E3%80%90%E8%B5%84%E6%96%99%E3%80%91%E5%AD%A6%E4%B9%A0%E4%B8%AD/%E3%80%90%E4%BC%98%E8%B4%A8%E3%80%91%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95%E8%AE%B2%E8%A7%A3%E8%AE%B2%E4%B9%89%EF%BC%88%E5%81%8F%E6%95%B0%E5%AD%A6%EF%BC%89/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E7%AE%97%E6%B3%95%E7%AF%87%EF%BC%89/%E8%81%9A%E7%B1%BB%E7%AE%97%E6%B3%95/images/111.png" alt="img" style="zoom:50%;" />

那么之间的相关系数怎么计算：

<img src="file:///Users/fox/%E9%9B%AA%E7%8B%B8%E7%9A%84%E6%96%87%E4%BB%B6/%E3%80%90%E8%B5%84%E6%96%99%E3%80%91%E5%AD%A6%E4%B9%A0%E4%B8%AD/%E3%80%90%E4%BC%98%E8%B4%A8%E3%80%91%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95%E8%AE%B2%E8%A7%A3%E8%AE%B2%E4%B9%89%EF%BC%88%E5%81%8F%E6%95%B0%E5%AD%A6%EF%BC%89/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E7%AE%97%E6%B3%95%E7%AF%87%EF%BC%89/%E8%81%9A%E7%B1%BB%E7%AE%97%E6%B3%95/images/%E7%9B%B8%E5%85%B3%E7%B3%BB%E6%95%B01.png" alt="img" style="zoom:50%;" />

最终计算：
$$
\frac{10 \times 16679.09-346.2 \times 422.5}{\sqrt{10 \times 14304.52-346.2^{2}} \sqrt{10 \times 19687.81-422.5^{2}}} =0.9942
$$
所以我们最终得出结论是广告投入费与月平均销售额之间有高度的正相关关系



---







### 特征选择

**定义：**数据中包含**冗余或相关变量（或称特征、属性、指标等）**，旨在从原有特征中找出主要特征



**方法：**

- **Filter（过滤式）**

    - **方差选择法**：将低方差的数据过滤掉，因为低方差说明没什么相关性

    - **相关系数法**：特征与特征之间的相关程度

        

- **Embedded（嵌入式）**

    - **决策树**：信息熵、信息增益
    - **正则化**：L1、L2
    - **深度学习**：卷积等



**如果两个特征的相关项很强**，那么我们可以：

1. 选取其中一个
2. 加权求和
3. 主成分分析

---



#### API

```python
sklearn.feature_selection
```



#### 低方差特征过滤

删除低方差的一些特征

- 特征方差**小**：某个特征大多数样本的值都比较接近；意味着这组特征是比较相近的，都选入是没有太大意义的
- 特征方差**大**：某个特征很多样本的值都有区别；这组特征适合保留下来



**获取转换器对象：**

`tranfer = sklearn.feature_selection.VarianceThreshold(threahold=0.0)`

- `threahold` 是临界值，低于或等于临界值的数据将会被删除

    

    **实例化后可调用：**

    - `VarianceThreshold.fit_transform(X=numpy.ndarry)` **返回**新特征值数组：训练集差异低于 `threahold` 的特征将会被删除。默认值是保留所有非零方差特征，即删除所有样本中具有相同值的特征
        - `X` numpy.ndarry 格式的二维数据



---

#### 主成分分析 (PCA)

主成分分析可以理解为一种特征提取的方式

**定义**：**高维数据转化为低维数据的过程**，在此过程中**可能会舍弃原有数据、创造新的变量**

**作用**：**是数据维数压缩，尽可能降低原数据的维数（复杂度），损失少量信息。**

**应用**：回归分析或者聚类分析当中

> 对于信息一词，在决策树中会进行介绍

那么更好的理解这个过程呢？我们来看一张图

<img src="doc/pic/README/PCA解释图.png" alt="PCAè§£é‡Šå›¾" style="zoom:50%;" />

假如对二维平面的 5 个点进行降维：

<img src="doc/pic/README/image-20220222190150894.png" alt="image-20220222190150894" style="zoom:30%;" />

将这个二维降为一维的方法，并且损失最少的信息：

<img src="doc/pic/README/image-20220222190244174.png" alt="image-20220222190244174" style="zoom:30%;" />

可以通过矩阵运算得到：
$$
Y=\left(\begin{array}{ll}
1 / \sqrt{2} & 1 / \sqrt{2}
\end{array}\right)\left(\begin{array}{ccccc}
-1 & -1 & 0 & 2 & 0 \\
-2 & 0 & 0 & 1 & 1
\end{array}\right)=\left(\begin{array}{cccc}
-3 / \sqrt{2} & -1 / \sqrt{2} & 0 & 3 / \sqrt{2} & -1 / \sqrt{2}
\end{array}\right)
$$

**API：**

**获取转换器：**

`transfer = sklearn.decomposition.PCA(n_components=None)` 将数据分解为较低维数的空间

**【注意】：这里的 `n_components` 传递整数和小数的效果是不一样的！** 

- **小数**：表示保留百分之多少的信息
- **整数：**表示减少到多少特征



**获取到转换器之后可以调用：**

`transfer.fir_transform(numpy.ndarry)` **返回**转换后的指定维数的数据



---

**【案例】探究用户对物品类别的喜好细分**

应用 PCA 和 K-means 实现用户对物品类别的喜好细分划分

数据如下：

- order_products_prior.csv：订单与商品信息
    - 字段：**order_id**, **product_id**, add_to_cart_order, reordered
- products.csv：商品信息
    - 字段：**product_id**, product_name, **aisle_id**, department_id
- orders.csv：用户的订单信息
    - 字段：**order_id**,**user_id**,eval_set,order_number,….
- aisles.csv：商品所属具体物品类别
    - 字段： **aisle_id**, **aisle**



**分析：**

- 1.获取数据
- 2.数据基本处理
    - 2.1 合并表格
    - 2.2 交叉表合并
    - 2.3 数据截取
- 3.特征工程 — PCA
- 4.机器学习（k-means）
- 5.模型评估
    - sklearn.metrics.silhouette_score(X, labels)
        - 计算所有样本的平均轮廓系数
        - X：特征值
        - labels：被聚类标记的目标值



---



## 分类算法

Q:如何判定属于分类问题？

A:没有目标值





### 转换器（Transformer）

==**转换器是特征工程的父类**==

回顾之前特征工程那里的步骤：

1. 实例化 `transfer` 转换器对象（实例化的是一个转换器类队对象：==Transformer==）
2. 调用 `transfer` 的 `.fit_transform(data)` 对象



其实这里的 `fit_transform()` 可以拆分为：

1. `fit()` 通常进行了计算，平均值、标准差等
2. `transform()` 进行最终的转换



### 估计器（Estimator）

**在 sklearn 中，估计器（estimator）是一个重要的角色，==是一类实现了算法的 API==**

**估计器的使用步骤：**

1. 实例化一个 Estimator

2. 将**训练集的特征值（x_train）**和**训练集的特征值（y_train）**传入进行计算
    `estimator.fit(x_train, y_train)`，相当于开始训练，调用完 `fit` 方法相当于**运算完成，模型生成**

3. 评估模型

    - 比对真实值和预测值
        `y_predict = estimator.predict(x_test)` 将**测试集的特征值（x_test）**传入进行预测，获得预测的目标值（y_predict）
        `y_predict == y_test` 判断预测的目标值（y_predict）和测试集的目标（y_test）值是否一致

    - 直接调用 `estimator` 内部的方法进行评估准确率

        `estimator.score(x_test, y_test)` 传入测试集的特征值和目标值

    

**估计器工作流程：**
<img src="doc/pic/README/image-20220419162953249.png" alt="image-20220419162953249" style="zoom:50%;" />



- 用于**分类**的估计器

    | 估计器                                    | 描述             |
    | ----------------------------------------- | ---------------- |
    | `sklearn.neighbors`                       | k-临近算法       |
    | `sklearn.naive_bayes`                     | 贝叶斯算法       |
    | `sklearn.linear_model.LogisticRegression` | 逻辑回归         |
    | `sklearn.tree`                            | 决策树与回归森林 |

    

- 用于**回归**的估计器

    | 估计器                                  | 描述     |
    | --------------------------------------- | -------- |
    | `sklearn.linear_model.LinearRegression` | 线性回归 |
    | `sklearn.linear_model.Ridge`            | 岭回归   |

    

- 用于**无监督学习**的估计器

    | 估计器                   | 描述     |
    | ------------------------ | -------- |
    | `sklearn.cluster.KMeans` | 聚类算法 |

    



## k-近邻算法

根据你的“邻居”来推断出你的类别：

<img src="doc/pic/README/地图K紧邻算法.png" alt="img" style="zoom:50%;" />





k-近邻算法也是 KNN 算法（K Nearest Neighbor），这个算法是机器学习里面一个比较经典的算法，总体来说 KNN算法是相对比较容易理解的算法



**定义：**
如果一个样本在特征空间中的 k 个最相似（即特征空间中最邻近）的样本中的大多数属于某一个类别，则该样本也属于这个类别。

> KNN算法最早是由 Cover 和 Hart 提出的一种分类算法



**距离公式：**

两个样本的距离可以通过如下公式计算

- 欧式距离
    $$
    d_{3维}=\sqrt{\left(x_{1}-x_{2}\right)^{2}+\left(y_{1}-y_{2}\right)^{2}+\left(z_{1}-z_{2}\right)^{2}}
    $$

- 曼哈顿距离
    $$
    d_{3维}=\left|x_{1}-x_{2}\right|+\left|y_{1}-y_{2}\right|+\left|z_{1}-z_{2}\right|
    $$
    

- 明科夫斯基距离



**示例：电影类型分析**

<img src="doc/pic/README/image-20220419180439940.png" alt="image-20220419180439940" style="zoom:50%;" />

<img src="doc/pic/README/image-20220419180506258.png" alt="image-20220419180506258" style="zoom:50%;" />

如果 $k = 1$ => 爱情片

如果 $k = 2$ => 爱情片

...

如果 $k = 6$ => 无法确定

如果增加一行动作片则 $k = 7$ => 动作片



**问题：可见 k-近邻算法受到 k 值的影响**

- K值过小
    - 容易受到异常点的影响
    - 容易过拟合
- k值过大：
    - 受到样本均衡的问题
    - 容易欠拟合



**KNN算法流程总结：**

1）计算已知类别数据集中的点与当前点之间的距离

2）按距离递增次序排序

3）选取与当前点距离最小的 k 个点

4）统计前k个点所在的类别出现的频率

5）返回前k个点出现频率最高的类别作为当前点的预测分类



**K-近邻算法 API：**

`sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='auto')`

- `n_neighbors -> int`：k-近邻算法中的 k 值（邻居数），默认为 5
- `algorithm：{'auto'，'bal_tree'，'kd_tree'，'brute'}`，可选用于计算最近邻居的算法，默认为 'auto'
    - bal_tree 将会使用 BallTree
    - kd_tree 将使用 KDTree
    - auto 将尝试根据传递给 fit 方法的值来决定最合适的算法。（不同实现方式影响效率）



**总结：**

- 优点：
    - 简单，易于理解，易于实现，无需训练
- 缺点：
    - 懒惰算法，对测试样本分类时的计算量大，内存开销大
    - 必须指定K值，K值选择不当则分类精度不能保证

使用场景：小数据场景，几千~几万样本，具体场景具体业务去测试





































