# 此部分代码在于理解本代码的部分np函数的用途，熟悉的可以忽略

# np.ones_like测试实例
test = np.array([[1,2,3],[4,5,6]])
test_one = np.ones_like(test)
print(test_one)


# reshape测试实例
K = np.arange(6)
print(K) 
T = K.reshape((3,2))
print(T)



# 理解np.shape的作用
K = np.arange(6)
print(K)
type(np.shape(K)[0]/2)


# 测试数组相减求和
def gain(y,y_pred):
    K = (y - y_pred).sum()
    return K
y = np.array([[1,2],[1,2]])
y_pred = np.array([[2,3],[2,9]])
gain(y,y_pred)


# 定义一个三元二次方程理解np.power的作用
def funciton_test(x_1,x_2,x_3):
    y = np.power(x_1,2)+3*x_2+x_3
    return y 
funciton_test(2,1,3)


##################################XGBoost代码从此处开始################################

# __future__模块，把下一个新版本的特性导入到当前版本，于是我们就可以在当前版本中测试一些新版本的特性，解决python2中运行pytho3兼容性问题
#　如果某个版本中出现了某个新的功能特性，而且这个特性和当前版本中使用的不兼容
# 也就是它在该版本中不是语言标准，那么我如果想要使用的话就需要从future模块导入
# division 表示精确除法
from __future__ import division, print_function
import numpy as np
# 显示完成的进度条
import progressbar

# xgboost算法也将决策树算法作为基函数进行使用
from utils.decision_tree.decision_tree_model import DecisionTree
# 导入进度条调度函数，方便展示模型训练进度和倒计时
from utils.misc import bar_widgets





# 最小二乘损失

class LeastSquaresLoss():
    """Least squares loss"""
    
    # 定义梯度，参数包括真实值和预测值
    def gradient(self, actual, predicted):
        return actual - predicted

    # 定义海塞矩阵，参数包括真实值和预测值,np.ones_like返回一个用1填充的跟输入形状和类型一致的数组
    def hess(self, actual, predicted):
        return np.ones_like(actual)





# XGBoost回归树，从父类决策树继承，是决策树的子类
class XGBoostRegressionTree(DecisionTree):
    """
    Regression tree for XGBoost
    - 参考文档 -
    http://xgboost.readthedocs.io/en/latest/model.html
    """
    # 有些时候，你会看到以一个下划线开头的实例变量名，比如_name，这样的实例变量外部是可以访问的
    # 但是，按照约定俗成的规定，当你看到这样的变量时，意思就是，“虽然我可以被访问，但是，请把我视为私有变量，不要随意访问”。
    def _split(self, y):
        """ y contains y_true in left half of the middle column and
        y_pred in the right half. Split and return the two matrices """
        # y输入是一个矩阵，np.shape是计算矩阵的行数和列数，此处代表返回矩阵的列数的一半用作划分点
        # 此处划分的目的在于将label划分为两部分
        col = int(np.shape(y)[1]/2)
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred
    
    # 定义基尼系数，注意此处的基尼系数和CART的基尼系数的区别
    def _gain(self, y, y_pred):
        
        # 假设这里的函数是平方误差，那么梯度就是残差，这里的结果就是对矩阵求元素对应位置相减，然后对所有元素求和，最后求平方
        nominator = np.power((self.loss.gradient(y, y_pred)).sum(), 2)
        # 返回一个以y为行列数的对角矩阵，对角线的元素均为1，并求和
        denominator = self.loss.hess(y, y_pred).sum()
        return 0.5 * (nominator / denominator)
    
    # 计算切分数据前后的基尼，并据此计算总的基尼系数
    def _gain_by_taylor(self, y, y1, y2):
        # Split
        y, y_pred = self._split(y)
        y1, y1_pred = self._split(y1)
        y2, y2_pred = self._split(y2)

        true_gain = self._gain(y1, y1_pred)
        false_gain = self._gain(y2, y2_pred)
        gain = self._gain(y, y_pred)
        return true_gain + false_gain - gain
    
    
    # 定义近似更新，这种近似更新的目的在于提供一种近似算法来拟合原有的切分算法
    def _approximate_update(self, y):
        # y split into y, y_pred
        y, y_pred = self._split(y)
        gradient = np.sum(self.loss.gradient(y, y_pred),axis=0)
        hessian = np.sum(self.loss.hess(y, y_pred), axis=0)
        update_approximation =  gradient / hessian
        return update_approximation


    def fit(self, X, y):
        self._impurity_calculation = self._gain_by_taylor
        self._leaf_value_calculation = self._approximate_update
        super(XGBoostRegressionTree, self).fit(X, y)








# 定义XGBoost分类树
class XGBoost(object):
    """The XGBoost classifier.

    参考文档: http://xgboost.readthedocs.io/en/latest/model.html

    Parameters:
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
    """

    def __init__(self, n_estimators=200, learning_rate=0.01, min_samples_split=2,
                 min_impurity=1e-7, max_depth=2):
        self.n_estimators = n_estimators  # Number of trees
        self.learning_rate = learning_rate  # Step size for weight update
        self.min_samples_split = min_samples_split  # The minimum n of sampels to justify split
        self.min_impurity = min_impurity  # Minimum variance reduction to continue
        self.max_depth = max_depth  # Maximum depth for tree

        self.bar = progressbar.ProgressBar(widgets=bar_widgets)

        # Log loss for classification
        self.loss = LeastSquaresLoss()

        # Initialize regression trees
        self.trees = []
        for _ in range(n_estimators):
            tree = XGBoostRegressionTree(
                min_samples_split=self.min_samples_split,
                min_impurity=min_impurity,
                max_depth=self.max_depth,
                loss=self.loss)

            self.trees.append(tree)

    def fit(self, X, y):
        # y = to_categorical(y)
        m = X.shape[0]
        y = np.reshape(y, (m, -1))
        y_pred = np.zeros(np.shape(y))
        for i in self.bar(range(self.n_estimators)):
            tree = self.trees[i]
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            tree.fit(X, y_and_pred)
            update_pred = tree.predict(X)
            update_pred = np.reshape(update_pred, (m, -1))
            y_pred += update_pred

    def predict(self, X):
        y_pred = None
        m = X.shape[0]
        # Make predictions
        for tree in self.trees:
            # Estimate gradient and update prediction
            update_pred = tree.predict(X)
            update_pred = np.reshape(update_pred, (m, -1))
            if y_pred is None:
                y_pred = np.zeros_like(update_pred)
            y_pred += update_pred

        return y_pred
        




# 开始使用实例进行测试，实例数据TempLinkoping2016.txt在GBDT文件夹可以找到

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from utils.data_manipulation import train_test_split, standardize, to_categorical, normalize
from utils.data_operation import mean_squared_error, accuracy_score

def main():
    print ("-- XGBoost --")

    # Load temperature data
    data = pd.read_csv('D:\jupyter_notebook\Machine-Learning-From-Scratch-master\TempLinkoping2016.txt', sep="\t")

    time = np.atleast_2d(data["time"].as_matrix()).T
    temp = np.atleast_2d(data["temp"].as_matrix()).T

    X = time.reshape((-1, 1))               # Time. Fraction of the year [0, 1]
    X = np.insert(X, 0, values=1, axis=1)   # Insert bias term
    y = temp[:, 0]                          # Temperature. Reduce to one-dim

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    #print(y_train)
    model = XGBoost()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred_line = model.predict(X)
    print(y_test[0:5])
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