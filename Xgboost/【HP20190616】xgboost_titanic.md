
## kaggle泰坦尼克号机器学习xgboost


```python
import numpy as np
import pandas as pd
import re 
import sklearn
import os
# 显示当前路径
os.getcwd()
```




    'D:\\jupyter_notebook'




```python
# 导入数据
train_ = pd.read_csv('D:/jupyter_notebook/titanic/train.csv')
test_ = pd.read_csv('D:/jupyter_notebook/titanic/test.csv')
```


```python
#根据原始特征的观察构建新特征
# 计算名字的长度
train_['Name_length'] = train_['Name'].apply(len)
# 将旅客是否住在头等舱二值化
train_['Has_Cabin'] = train_["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
# 构建新特征家庭总人数
train_['FamilySize'] = train_['SibSp'] + train_['Parch'] + 1
# 构建新特征是否独居
train_['IsAlone'] = 0
train_.loc[train_['FamilySize'] == 1, 'IsAlone'] = 1
# 查看乘客登船口岸存在缺失值
train_['Embarked'].isnull().value_counts() 
# 对乘客登船口岸进行固定值填充缺失值
train_['Embarked'] = train_['Embarked'].fillna('S')
# 对票价进行中位数填充缺失值
train_['Fare'] = train_['Fare'].fillna(train_['Fare'].median())
# 生成绝对票价分区，qcut是根据分区分位定义，将每一个值划为到具体的分区区间中去，此处定义为四分位值
train_['CategoricalFare'] = pd.qcut(train_['Fare'], 4)
# 生成新变量年龄平均值、年龄标准差
age_avg = train_['Age'].mean()
age_std = train_['Age'].std()
# 计算年龄是否有缺失值并统计
age_null_count = train_['Age'].isnull().sum()
# np.random.randint()产生离散均匀分布的整数,size是产生的元素数量，前面分别为最小值和最大值区间
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
# 对年龄用生成的一些新数值进行填充
train_['Age'][np.isnan(train_['Age'])] = age_null_random_list
# 转换变量类型为数值类型，便于后期计算
train_['Age'] = train_['Age'].astype(int)
# 对年龄生成新的分箱变量中来代替，即将年龄绝对值转换为离散类别
train_['CategoricalAge'] = pd.cut(train_['Age'], 5)

# 定义正则表达式函数导出旅客的Title
def get_title(name):
    # re.search()方法扫描整个字符串，并返回第一个成功的匹配。如果匹配失败，则返回None
    title_search = re.search('([A-Za-z]+)\.',name)
    if title_search:
        return title_search.group(1)
    return ''

# 取出姓名中尊称部分
train_['Title'] = train_['Name'].apply(get_title)

# 对姓名的称呼部分做统一
train_['Title'] = train_['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major'
                                           , 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_['Title'] = train_['Title'].replace('Mlle', 'Miss')
train_['Title'] = train_['Title'].replace('Ms', 'Miss')
train_['Title'] = train_['Title'].replace('Mme', 'Mrs')

# 对性别从离散型替换为数值型
train_['Sex'] = train_['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
# 对姓名的称呼部分做数值型变换
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
# 先定义一个字典，然后通过map函数传入字典进行替换
train_['Title'] = train_['Title'].map(title_mapping)
# 最后对缺失值替换为0
train_['Title'] = train_['Title'].fillna(0)
    
# 替换登船口岸
train_['Embarked'] = train_['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
# 替换票价的四分位数，该步骤应该有更好的办法做数据处理
# loc函数取出列中某类元素的数据集
train_.loc[ train_['Fare'] <= 7.91, 'Fare'] = 0
train_.loc[(train_['Fare'] > 7.91) & (train_['Fare'] <= 14.454), 'Fare'] = 1
train_.loc[(train_['Fare'] > 14.454) & (train_['Fare'] <= 31), 'Fare']   = 2
train_.loc[ train_['Fare'] > 31, 'Fare'] = 3
train_['Fare'] = train_['Fare'].astype(int)
    
# 对年龄进行分段
train_.loc[ train_['Age'] <= 16, 'Age'] = 0
train_.loc[(train_['Age'] > 16) & (train_['Age'] <= 32), 'Age'] = 1
train_.loc[(train_['Age'] > 32) & (train_['Age'] <= 48), 'Age'] = 2
train_.loc[(train_['Age'] > 48) & (train_['Age'] <= 64), 'Age'] = 3
train_.loc[train_['Age'] > 64, 'Age'] = 4


# 特征选择，先对处理过的不需要的特征进行删除，定义一个列表，然后批量删除
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train_ = train_.drop(drop_elements, axis = 1)
train_ = train_.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
# test_  = test_.drop(drop_elements, axis = 1)

train_.head()
```

    C:\Users\IBM\Anaconda3\lib\site-packages\ipykernel_launcher.py:27: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Name_length</th>
      <th>Has_Cabin</th>
      <th>FamilySize</th>
      <th>IsAlone</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>23</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>51</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>44</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>24</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#根据原始特征的观察构建新特征
# 计算名字的长度
test_['Name_length'] = test_['Name'].apply(len)
# 将旅客是否住在头等舱二值化
test_['Has_Cabin'] = test_["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
# 构建新特征家庭总人数
test_['FamilySize'] = test_['SibSp'] + test_['Parch'] + 1
# 构建新特征是否独居
test_['IsAlone'] = 0
test_.loc[test_['FamilySize'] == 1, 'IsAlone'] = 1
# 查看乘客登船口岸存在缺失值
test_['Embarked'].isnull().value_counts() 
# 对乘客登船口岸进行固定值填充缺失值
test_['Embarked'] = test_['Embarked'].fillna('S')
# 对票价进行中位数填充缺失值
test_['Fare'] = test_['Fare'].fillna(test_['Fare'].median())
# 生成绝对票价分区，qcut是根据分区分位定义，将每一个值划为到具体的分区区间中去，此处定义为四分位值
test_['CategoricalFare'] = pd.qcut(test_['Fare'], 4)
# 生成新变量年龄平均值、年龄标准差
age_avg = test_['Age'].mean()
age_std = test_['Age'].std()
# 计算年龄是否有缺失值并统计
age_null_count = test_['Age'].isnull().sum()
# np.random.randint()产生离散均匀分布的整数,size是产生的元素数量，前面分别为最小值和最大值区间
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
# 对年龄用生成的一些新数值进行填充
test_['Age'][np.isnan(test_['Age'])] = age_null_random_list
# 转换变量类型为数值类型，便于后期计算
test_['Age'] = test_['Age'].astype(int)
# 对年龄生成新的分箱变量中来代替，即将年龄绝对值转换为离散类别
test_['CategoricalAge'] = pd.cut(test_['Age'], 5)

# 定义正则表达式函数导出旅客的Title
def get_title(name):
    # re.search()方法扫描整个字符串，并返回第一个成功的匹配。如果匹配失败，则返回None
    title_search = re.search('([A-Za-z]+)\.',name)
    if title_search:
        return title_search.group(1)
    return ''

# 取出姓名中尊称部分
test_['Title'] = test_['Name'].apply(get_title)

# 对姓名的称呼部分做统一
test_['Title'] = test_['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major'
                                           , 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_['Title'] = test_['Title'].replace('Mlle', 'Miss')
test_['Title'] = test_['Title'].replace('Ms', 'Miss')
test_['Title'] = test_['Title'].replace('Mme', 'Mrs')

# 对性别从离散型替换为数值型
test_['Sex'] = test_['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
# 对姓名的称呼部分做数值型变换
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
# 先定义一个字典，然后通过map函数传入字典进行替换
test_['Title'] = test_['Title'].map(title_mapping)
# 最后对缺失值替换为0
test_['Title'] = test_['Title'].fillna(0)
    
# 替换登船口岸
test_['Embarked'] = test_['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
# 替换票价的四分位数，该步骤应该有更好的办法做数据处理
# loc函数取出列中某类元素的数据集
test_.loc[ test_['Fare'] <= 7.91, 'Fare'] = 0
test_.loc[(test_['Fare'] > 7.91) & (test_['Fare'] <= 14.454), 'Fare'] = 1
test_.loc[(test_['Fare'] > 14.454) & (test_['Fare'] <= 31), 'Fare']   = 2
test_.loc[ test_['Fare'] > 31, 'Fare'] = 3
test_['Fare'] = test_['Fare'].astype(int)
    
# 对年龄进行分段
test_.loc[ test_['Age'] <= 16, 'Age'] = 0
test_.loc[(test_['Age'] > 16) & (test_['Age'] <= 32), 'Age'] = 1
test_.loc[(test_['Age'] > 32) & (test_['Age'] <= 48), 'Age'] = 2
test_.loc[(test_['Age'] > 48) & (test_['Age'] <= 64), 'Age'] = 3
test_.loc[test_['Age'] > 64, 'Age'] = 4


# 特征选择，先对处理过的不需要的特征进行删除，定义一个列表，然后批量删除
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
test_ = test_.drop(drop_elements, axis = 1)
test_ = test_.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
# test_  = test_.drop(drop_elements, axis = 1)

test_.head()
```

    C:\Users\IBM\Anaconda3\lib\site-packages\ipykernel_launcher.py:27: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Name_length</th>
      <th>Has_Cabin</th>
      <th>FamilySize</th>
      <th>IsAlone</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>25</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>44</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
import xgboost as xgb
import pandas as pd 
import numpy as np
import sklearn 
import os 
from sklearn.model_selection import train_test_split  # 导入测试集和验证集划分函数
```


```python
X = train_.drop("Survived",axis= 1) # 提取不带标签的数据集
Y = train_["Survived"] # 提取数据集的标签，数据集的标签一般是指用于预测的label
```


```python
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 0)
```


```python
dtrain = xgb.DMatrix(X_train, label=Y_train)
dtest = xgb.DMatrix(X_test, label=Y_test)
```


```python
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'multi:softmax', 'num_class':3}
 
bst = xgb.train(param, dtrain, num_boost_round=10, evals=watchlist)
y_hat = bst.predict(dtest)
result = Y_test.values.reshape(1, -1) == y_hat
print('the accuracy:\t', float(np.sum(result)) / len(y_hat))
```

    [0]	eval-merror:0.201493	train-merror:0.160514
    [1]	eval-merror:0.16791	train-merror:0.157303
    [2]	eval-merror:0.175373	train-merror:0.144462
    [3]	eval-merror:0.182836	train-merror:0.136437
    [4]	eval-merror:0.171642	train-merror:0.138042
    [5]	eval-merror:0.160448	train-merror:0.133226
    [6]	eval-merror:0.160448	train-merror:0.126806
    [7]	eval-merror:0.171642	train-merror:0.125201
    [8]	eval-merror:0.164179	train-merror:0.11878
    [9]	eval-merror:0.164179	train-merror:0.117175
    the accuracy:	 0.835820895522388
    


```python

```
