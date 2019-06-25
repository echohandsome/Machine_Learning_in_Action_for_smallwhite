
## kaggle泰坦尼克号机器学习stacking模型融合


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
# 为方便进行数据处理，将训练集和测试集合并进行数据处理
train_['number'] = 1 
test_['number'] = 0
datamart = pd.concat([train_, test_], axis=0, join='outer')  
```

    C:\Users\IBM\Anaconda3\lib\site-packages\ipykernel_launcher.py:4: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
    of pandas will change to not sort by default.
    
    To accept the future behavior, pass 'sort=False'.
    
    To retain the current behavior and silence the warning, pass 'sort=True'.
    
      after removing the cwd from sys.path.
    

### 1.根据原始特征进行特征处理，训练集和测试集合并处理


```python
#根据原始特征的观察构建新特征
# 计算名字的长度
datamart['Name_length'] = datamart['Name'].apply(len)
# 将旅客是否住在头等舱二值化
datamart['Has_Cabin'] = datamart["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
# 构建新特征家庭总人数
datamart['FamilySize'] = datamart['SibSp'] + datamart['Parch'] + 1
# 构建新特征是否独居
datamart['IsAlone'] = 0
datamart.loc[datamart['FamilySize'] == 1, 'IsAlone'] = 1
# 查看乘客登船口岸存在缺失值
datamart['Embarked'].isnull().value_counts() 
# 对乘客登船口岸进行固定值填充缺失值
datamart['Embarked'] = datamart['Embarked'].fillna('S')
# 对票价进行中位数填充缺失值
datamart['Fare'] = datamart['Fare'].fillna(datamart['Fare'].median())
# 生成绝对票价分区，qcut是根据分区分位定义，将每一个值划为到具体的分区区间中去，此处定义为四分位值
datamart['CategoricalFare'] = pd.qcut(datamart['Fare'], 4)
# 生成新变量年龄平均值、年龄标准差
age_avg = datamart['Age'].mean()
age_std = datamart['Age'].std()
# 计算年龄是否有缺失值并统计
age_null_count = datamart['Age'].isnull().sum()
# np.random.randint()产生离散均匀分布的整数,size是产生的元素数量，前面分别为最小值和最大值区间
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
# 对年龄用生成的一些新数值进行填充
datamart['Age'][np.isnan(datamart['Age'])] = age_null_random_list
# 转换变量类型为数值类型，便于后期计算
datamart['Age'] = datamart['Age'].astype(int)
# 对年龄生成新的分箱变量中来代替，即将年龄绝对值转换为离散类别
datamart['CategoricalAge'] = pd.cut(datamart['Age'], 5)

# 定义正则表达式函数导出旅客的Title
def get_title(name):
    # re.search()方法扫描整个字符串，并返回第一个成功的匹配。如果匹配失败，则返回None
    title_search = re.search('([A-Za-z]+)\.',name)
    if title_search:
        return title_search.group(1)
    return ''

# 取出姓名中尊称部分
datamart['Title'] = datamart['Name'].apply(get_title)

# 对姓名的称呼部分做统一
datamart['Title'] = datamart['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major'
                                           , 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
datamart['Title'] = datamart['Title'].replace('Mlle', 'Miss')
datamart['Title'] = datamart['Title'].replace('Ms', 'Miss')
datamart['Title'] = datamart['Title'].replace('Mme', 'Mrs')

# 对性别从离散型替换为数值型
datamart['Sex'] = datamart['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
# 对姓名的称呼部分做数值型变换
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
# 先定义一个字典，然后通过map函数传入字典进行替换
datamart['Title'] = datamart['Title'].map(title_mapping)
# 最后对缺失值替换为0
datamart['Title'] = datamart['Title'].fillna(0)
    
# 替换登船口岸
datamart['Embarked'] = datamart['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
# 替换票价的四分位数，该步骤应该有更好的办法做数据处理
# loc函数取出列中某类元素的数据集
datamart.loc[ datamart['Fare'] <= 7.91, 'Fare'] = 0
datamart.loc[(datamart['Fare'] > 7.91) & (datamart['Fare'] <= 14.454), 'Fare'] = 1
datamart.loc[(datamart['Fare'] > 14.454) & (datamart['Fare'] <= 31), 'Fare']   = 2
datamart.loc[ datamart['Fare'] > 31, 'Fare'] = 3
datamart['Fare'] = datamart['Fare'].astype(int)
    
# 对年龄进行分段
datamart.loc[ datamart['Age'] <= 16, 'Age'] = 0
datamart.loc[(datamart['Age'] > 16) & (datamart['Age'] <= 32), 'Age'] = 1
datamart.loc[(datamart['Age'] > 32) & (datamart['Age'] <= 48), 'Age'] = 2
datamart.loc[(datamart['Age'] > 48) & (datamart['Age'] <= 64), 'Age'] = 3
datamart.loc[datamart['Age'] > 64, 'Age'] = 4


# 特征选择，先对处理过的不需要的特征进行删除，定义一个列表，然后批量删除
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
datamart = datamart.drop(drop_elements, axis = 1)
datamart = datamart.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
# test_  = test_.drop(drop_elements, axis = 1)

datamart.head()
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
      <th>Age</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Survived</th>
      <th>number</th>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>23</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
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
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>22</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>44</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>24</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### 2.对特征处理后的测试集和训练集分开


```python
# 通过loc方法选取训练集的数据
train_new = datamart.loc[datamart['number'] == 1]
# 对number列进行删除
train_new = train_new.drop(['number'],axis=1)
```


```python
train_new.head()
```




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
      <th>Age</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Survived</th>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0.0</td>
      <td>23</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>51</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1.0</td>
      <td>22</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>44</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0.0</td>
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
test_new = datamart.loc[datamart['number'] == 0]
drop_columns = ['number','Survived']
test_new = test_new.drop(drop_columns,axis=1)
```


```python
test_new.head()
```




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
      <th>Age</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
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
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>32</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>25</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
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



### 3.导入机器学习库对数据特征进行探索


```python
import sklearn
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
```


        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        requirejs.config({
            paths: {
                'plotly': ['https://cdn.plot.ly/plotly-latest.min']
            }
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        



```python

```
