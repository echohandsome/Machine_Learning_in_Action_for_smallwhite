#### 基于面向对象思路实现的代码若无基础需先补充面向对象知识

推荐面向对象知识补充路径：

https://www.liaoxuefeng.com/wiki/1016959663602400/1017495723838528


```python

# __future__模块，把下一个新版本的特性导入到当前版本，于是我们就可以在当前版本中测试一些新版本的特性，解决python2中运行pytho3兼容性问题
#　如果某个版本中出现了某个新的功能特性，而且这个特性和当前版本中使用的不兼容
# 也就是它在该版本中不是语言标准，那么我如果想要使用的话就需要从future模块导入
# division 表示精确除法
from __future__ import division, print_function
import numpy as np
# 显示完成的进度条
from progressbar import *


# 这段代码主要展示progressbar进度条的作用，在于展示任务的完成进度并显示出来，与本项目无任何关系
import time
from progressbar import *
 
total = 1000
 
def dosomework():
    time.sleep(0.01)
 
progress = ProgressBar()
for i in progress(range(1000)):
    dosomework()






# 导入辅助函数,这里的辅助函数全部在模块中，如果代码报错，需要自行对辅助函数的py文件和类进行整理

# 训练集和测试集划分函数、数据处理标准化函数、将整型的类别标签转为onehot编码函数
from utils.data_manipulation import train_test_split, standardize, to_categorical

# 导入均方误差函数、分类AUC评估函数
from utils.data_operation import mean_squared_error, accuracy_score

# GBDT需要用到决策树的回归树模块，这也是GBDT的核心基础算法之一，如果对决策树不熟悉，需要先学习决策树decision_tree库下面的代码
from utils.decision_tree.decision_tree_model import RegressionTree

# 导入进度条调度函数，方便展示模型训练进度和倒计时
from utils.misc import bar_widgets

# 导入平方损失函数、交叉熵损失函数（用于多分类损失评估）、softmax损失函数
from utils.loss_functions import SquareLoss, CrossEntropy, SoftMaxLoss






# 这里定义GBDT的核心算法父类，后面的分类和回归算法直接继承父类的函数方法
class GBDT(object):
    """使用一组回归树来训练预测梯度损失函数。
    参数:
    -----------
    n_estimators: int
        树的数量
        The number of classification trees that are used.
    learning_rate: float
        梯度下降的学习率
        The step length that will be taken when following the negative gradient during
        training.
    min_samples_split: int
        每棵子树的节点的最小数目（小于后不继续切割）
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        每棵子树的最小纯度（小于后不继续切割）
        The minimum impurity required to split the tree further.
    max_depth: int
        每棵子树的最大层数（大于后不继续切割）
        The maximum depth of a tree.
    regression: boolean
        是否为回归问题
        True or false depending on if we're doing regression or classification.
    """

    def __init__(self, n_estimators, learning_rate, min_samples_split,
                 min_impurity, max_depth, regression):
        
        # self表示实例本身，在__init__方法内部，就可以把各种属性绑定到self
        # 由于类可以起到模板的作用，因此，可以在创建实例的时候，把一些我们认为必须绑定的属性强制填写进去。
        # 通过定义一个特殊的__init__方法，在创建实例的时候，把上述的属性（参数）绑定到self
        
        # 树的棵数、梯度下降学习率α、每棵子树节点最小数量、每棵子树的最小纯度、每棵子树最大深度、是否为回归问题
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression

        # 进度条 processbar
        self.bar = progressbar.ProgressBar(widgets=bar_widgets)

        # 定义损失函数为平方损失
        self.loss = SquareLoss()
        
        # 如果是分类问题，则使用SoftMax损失
        if not self.regression:
            self.loss = SoftMaxLoss()

        # 分类问题也使用回归树，利用残差去学习概率
        self.trees = []
        
        # 遍历每一棵树，并对每一颗数调用回归树方法，并限制树的相关参数执行上述定义的参数
        for i in range(self.n_estimators):
            self.trees.append(RegressionTree(min_samples_split=self.min_samples_split,
                                             min_impurity=self.min_impurity,
                                             max_depth=self.max_depth))
    
    
    # GBDT的核心代码
    def fit(self, X, y):
        # 让第一棵树去拟合模型
        self.trees[0].fit(X, y)
        y_pred = self.trees[0].predict(X)
        for i in self.bar(range(1, self.n_estimators)):
            gradient = self.loss.gradient(y, y_pred)
            self.trees[i].fit(X, gradient)
            y_pred -= np.multiply(self.learning_rate, self.trees[i].predict(X))
    
    # np.multiply 矩阵乘法
    def predict(self, X):
        y_pred = self.trees[0].predict(X)
        for i in range(1, self.n_estimators):
            y_pred -= np.multiply(self.learning_rate, self.trees[i].predict(X))

        # 判断是分类问题则转换预测值的表达方式
        if not self.regression:
            # 如果是分类问题，则转换为概率分布
            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
            # 将标签设置为最大化概率的值
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred





# 对np.argmax的功能做一次演示,按行搜索最大值的索引，当前行存在多个并列最大值取第一个所在位置索引
import numpy as np
a = np.array([[1, 5, 5, 2],
              [9, 6, 2, 8],
              [3, 7, 9, 1]])

c=np.argmax(a, axis=1)
print(c)

[1 0 2]


# 对np.multiply功能做一次演示，向量乘法和矩阵乘法
A = np.array([[1, 2],
       [3, 4]])
B = np.array([[0, 1],
       [2, 3]])
C = np.multiply(A, B)

print(C)

print(np.multiply(np.mat(A),np.mat(B)))


[[ 0  2]
 [ 6 12]]
 
[[ 0  2]
 [ 6 12]]








# GBDT回归算法
class GBDTRegressor(GBDT):
    def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                 min_var_red=1e-7, max_depth=4, debug=False):
        super(GBDTRegressor, self).__init__(n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            min_samples_split=min_samples_split,
                                            min_impurity=min_var_red,
                                            max_depth=max_depth,
                                            regression=True)






# GBDT分类算法
# 在类中提前定义好训练的参数
class GBDTClassifier(GBDT):
    def __init__(self, n_estimators=200, learning_rate=.5, min_samples_split=2,
                 min_info_gain=1e-7, max_depth=2, debug=False):
        super(GBDTClassifier, self).__init__(n_estimators=n_estimators,
                                             learning_rate=learning_rate,
                                             min_samples_split=min_samples_split,
                                             min_impurity=min_info_gain,
                                             max_depth=max_depth,
                                             regression=False)
    def fit(self, X, y):
        
        # 对多分类label进行one_hot编码
        y = to_categorical(y)
        super(GBDTClassifier, self).fit(X, y)
        
        
        
        






# 分类算法的具体测试实例

from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

from utils.misc import Plot


def main():

    print ("-- Gradient Boosting Classification --")
    
    # 利用自带的鸢尾花数据集
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # 划分训练集和测试集，测试集比例40%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    print(y_train)

    clf = GBDTClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)


    Plot().plot_in_2d(X_test, y_pred,
        title="Gradient Boosting",
        accuracy=accuracy,
        legend_labels=data.target_names)



if __name__ == "__main__":
    main()
    
    
    







# 回归算法的具体测试实例，此处需要导入文件TempLinkoping2016.txt进行测试
# from __future__ import division, print_function
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import progressbar

# from utils import train_test_split, standardize, to_categorical
# from utils import mean_squared_error, accuracy_score, Plot
# from utils.loss_functions import SquareLoss
# from utils.misc import bar_widgets
# from gradient_boosting_decision_tree.gbdt_model import GBDTRegressor

def main():
    print ("-- Gradient Boosting Regression --")

    # Load temperature data
    data = pd.read_csv('D:\Git\Machine_Learning_in_Action_for_smallwhite\GBDT\GBDT_python3_code\TempLinkoping2016.txt', sep="\t")

    time = np.atleast_2d(data["time"].as_matrix()).T
    temp = np.atleast_2d(data["temp"].as_matrix()).T

    X = time.reshape((-1, 1))               # Time. Fraction of the year [0, 1]
    X = np.insert(X, 0, values=1, axis=1)   # Insert bias term
    y = temp[:, 0]                          # Temperature. Reduce to one-dim

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    model = GBDTRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred_line = model.predict(X)

    # Color map
    cmap = plt.get_cmap('viridis')

    mse = mean_squared_error(y_test, y_pred)

    print ("Mean Squared Error:", mse)

    # Plot the results
    m1 = plt.scatter(366 * X_train[:, 1], y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test[:, 1], y_test, color=cmap(0.5), s=10)
    m3 = plt.scatter(366 * X_test[:, 1], y_pred, color='black', s=10)
    plt.suptitle("Regression Tree")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2, m3), ("Training data", "Test data", "Prediction"), loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()








```