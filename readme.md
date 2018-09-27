<!-- TOC -->

- [时间格式的转化](#时间格式的转化)
- [查看数据类型](#查看数据类型)
- [查看DataFrame的详细信息](#查看dataframe的详细信息)
- [填充缺失值](#填充缺失值)
- [category 数据类型转化](#category-数据类型转化)
- [模型参数设定](#模型参数设定)
- [结论](#结论)

<!-- /TOC -->

- 该项目是针对kaggle中的[homesite](https://www.kaggle.com/c/homesite-quote-conversion)进行的算法预测，使用xgboost的sklearn接口，进行数据建模，购买预测。

```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
```


```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>QuoteNumber</th>
      <th>Original_Quote_Date</th>
      <th>QuoteConversion_Flag</th>
      <th>Field6</th>
      <th>Field7</th>
      <th>Field8</th>
      <th>Field9</th>
      <th>Field10</th>
      <th>Field11</th>
      <th>Field12</th>
      <th>...</th>
      <th>GeographicField59A</th>
      <th>GeographicField59B</th>
      <th>GeographicField60A</th>
      <th>GeographicField60B</th>
      <th>GeographicField61A</th>
      <th>GeographicField61B</th>
      <th>GeographicField62A</th>
      <th>GeographicField62B</th>
      <th>GeographicField63</th>
      <th>GeographicField64</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2013-08-16</td>
      <td>0</td>
      <td>B</td>
      <td>23</td>
      <td>0.9403</td>
      <td>0.0006</td>
      <td>965</td>
      <td>1.0200</td>
      <td>N</td>
      <td>...</td>
      <td>9</td>
      <td>9</td>
      <td>-1</td>
      <td>8</td>
      <td>-1</td>
      <td>18</td>
      <td>-1</td>
      <td>10</td>
      <td>N</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2014-04-22</td>
      <td>0</td>
      <td>F</td>
      <td>7</td>
      <td>1.0006</td>
      <td>0.0040</td>
      <td>548</td>
      <td>1.2433</td>
      <td>N</td>
      <td>...</td>
      <td>10</td>
      <td>10</td>
      <td>-1</td>
      <td>11</td>
      <td>-1</td>
      <td>17</td>
      <td>-1</td>
      <td>20</td>
      <td>N</td>
      <td>NJ</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>2014-08-25</td>
      <td>0</td>
      <td>F</td>
      <td>7</td>
      <td>1.0006</td>
      <td>0.0040</td>
      <td>548</td>
      <td>1.2433</td>
      <td>N</td>
      <td>...</td>
      <td>15</td>
      <td>18</td>
      <td>-1</td>
      <td>21</td>
      <td>-1</td>
      <td>11</td>
      <td>-1</td>
      <td>8</td>
      <td>N</td>
      <td>NJ</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>2013-04-15</td>
      <td>0</td>
      <td>J</td>
      <td>10</td>
      <td>0.9769</td>
      <td>0.0004</td>
      <td>1,165</td>
      <td>1.2665</td>
      <td>N</td>
      <td>...</td>
      <td>6</td>
      <td>5</td>
      <td>-1</td>
      <td>10</td>
      <td>-1</td>
      <td>9</td>
      <td>-1</td>
      <td>21</td>
      <td>N</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>2014-01-25</td>
      <td>0</td>
      <td>E</td>
      <td>23</td>
      <td>0.9472</td>
      <td>0.0006</td>
      <td>1,487</td>
      <td>1.3045</td>
      <td>N</td>
      <td>...</td>
      <td>18</td>
      <td>22</td>
      <td>-1</td>
      <td>10</td>
      <td>-1</td>
      <td>11</td>
      <td>-1</td>
      <td>12</td>
      <td>N</td>
      <td>IL</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 299 columns</p>
</div>




```python
train=train.drop('QuoteNumber',axis=1)
```


```python
test = test.drop('QuoteNumber', axis=1)
```

####  时间格式的转化


```python
train['Date']=pd.to_datetime(train['Original_Quote_Date'])
train= train.drop('Original_Quote_Date',axis=1)
```


```python
test['Date']=pd.to_datetime(test['Original_Quote_Date'])
test= test.drop('Original_Quote_Date',axis=1)
```


```python
train['year']=train['Date'].dt.year
```


```python
train['month']=train['Date'].dt.month
train['weekday']=train['Date'].dt.weekday
```


```python
train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>QuoteConversion_Flag</th>
      <th>Field6</th>
      <th>Field7</th>
      <th>Field8</th>
      <th>Field9</th>
      <th>Field10</th>
      <th>Field11</th>
      <th>Field12</th>
      <th>CoverageField1A</th>
      <th>CoverageField1B</th>
      <th>...</th>
      <th>GeographicField61A</th>
      <th>GeographicField61B</th>
      <th>GeographicField62A</th>
      <th>GeographicField62B</th>
      <th>GeographicField63</th>
      <th>GeographicField64</th>
      <th>Date</th>
      <th>year</th>
      <th>month</th>
      <th>weekday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>B</td>
      <td>23</td>
      <td>0.9403</td>
      <td>0.0006</td>
      <td>965</td>
      <td>1.0200</td>
      <td>N</td>
      <td>17</td>
      <td>23</td>
      <td>...</td>
      <td>-1</td>
      <td>18</td>
      <td>-1</td>
      <td>10</td>
      <td>N</td>
      <td>CA</td>
      <td>2013-08-16</td>
      <td>2013</td>
      <td>8</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>F</td>
      <td>7</td>
      <td>1.0006</td>
      <td>0.0040</td>
      <td>548</td>
      <td>1.2433</td>
      <td>N</td>
      <td>6</td>
      <td>8</td>
      <td>...</td>
      <td>-1</td>
      <td>17</td>
      <td>-1</td>
      <td>20</td>
      <td>N</td>
      <td>NJ</td>
      <td>2014-04-22</td>
      <td>2014</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>F</td>
      <td>7</td>
      <td>1.0006</td>
      <td>0.0040</td>
      <td>548</td>
      <td>1.2433</td>
      <td>N</td>
      <td>7</td>
      <td>12</td>
      <td>...</td>
      <td>-1</td>
      <td>11</td>
      <td>-1</td>
      <td>8</td>
      <td>N</td>
      <td>NJ</td>
      <td>2014-08-25</td>
      <td>2014</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>J</td>
      <td>10</td>
      <td>0.9769</td>
      <td>0.0004</td>
      <td>1,165</td>
      <td>1.2665</td>
      <td>N</td>
      <td>3</td>
      <td>2</td>
      <td>...</td>
      <td>-1</td>
      <td>9</td>
      <td>-1</td>
      <td>21</td>
      <td>N</td>
      <td>TX</td>
      <td>2013-04-15</td>
      <td>2013</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>E</td>
      <td>23</td>
      <td>0.9472</td>
      <td>0.0006</td>
      <td>1,487</td>
      <td>1.3045</td>
      <td>N</td>
      <td>8</td>
      <td>13</td>
      <td>...</td>
      <td>-1</td>
      <td>11</td>
      <td>-1</td>
      <td>12</td>
      <td>N</td>
      <td>IL</td>
      <td>2014-01-25</td>
      <td>2014</td>
      <td>1</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 301 columns</p>
</div>




```python
test['year']=test['Date'].dt.year
test['month']=test['Date'].dt.month
test['weekday']=test['Date'].dt.weekday
```


```python
train = train.drop('Date', axis=1)  
test = test.drop('Date', axis=1)
```

#### 查看数据类型


```python
train.dtypes
```




    QuoteConversion_Flag      int64
    Field6                   object
    Field7                    int64
    Field8                  float64
    Field9                  float64
    Field10                  object
    Field11                 float64
    Field12                  object
    CoverageField1A           int64
    CoverageField1B           int64
    CoverageField2A           int64
    CoverageField2B           int64
    CoverageField3A           int64
    CoverageField3B           int64
    CoverageField4A           int64
    CoverageField4B           int64
    CoverageField5A           int64
    CoverageField5B           int64
    CoverageField6A           int64
    CoverageField6B           int64
    CoverageField8           object
    CoverageField9           object
    CoverageField11A          int64
    CoverageField11B          int64
    SalesField1A              int64
    SalesField1B              int64
    SalesField2A              int64
    SalesField2B              int64
    SalesField3               int64
    SalesField4               int64
                             ...   
    GeographicField50B        int64
    GeographicField51A        int64
    GeographicField51B        int64
    GeographicField52A        int64
    GeographicField52B        int64
    GeographicField53A        int64
    GeographicField53B        int64
    GeographicField54A        int64
    GeographicField54B        int64
    GeographicField55A        int64
    GeographicField55B        int64
    GeographicField56A        int64
    GeographicField56B        int64
    GeographicField57A        int64
    GeographicField57B        int64
    GeographicField58A        int64
    GeographicField58B        int64
    GeographicField59A        int64
    GeographicField59B        int64
    GeographicField60A        int64
    GeographicField60B        int64
    GeographicField61A        int64
    GeographicField61B        int64
    GeographicField62A        int64
    GeographicField62B        int64
    GeographicField63        object
    GeographicField64        object
    year                      int64
    month                     int64
    weekday                   int64
    Length: 300, dtype: object



#### 查看DataFrame的详细信息


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 260753 entries, 0 to 260752
    Columns: 300 entries, QuoteConversion_Flag to weekday
    dtypes: float64(6), int64(267), object(27)
    memory usage: 596.8+ MB
    

####  填充缺失值


```python
train = train.fillna(-999)
test = test.fillna(-999)
```

#### category 数据类型转化


```python
from sklearn import preprocessing
features = list(train.columns[1:])  
for i in features:
    if train[i].dtype=='object':
        le=preprocessing.LabelEncoder()
        le.fit(list(train[i].values)+list(test[i].values))
        train[i] = le.transform(list(train[i].values))
        test[i] = le.transform(list(test[i].values))
        
```

#### 模型参数设定


```python
#brute force scan for all parameters, here are the tricks
#usually max_depth is 6,7,8
#learning rate is around 0.05, but small changes may make big diff
#tuning min_child_weight subsample colsample_bytree can have 
#much fun of fighting against overfit 
#n_estimators is how many round of boosting
#finally, ensemble xgboost with multiple seeds may reduce variance

xgb_model = xgb.XGBClassifier()

parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.05,0.1], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [5], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}

```


```python
sfolder = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
clf= GridSearchCV(xgb_model,parameters,n_jobs=4,cv=sfolder.split(train[features], train["QuoteConversion_Flag"]),scoring='roc_auc',
                   verbose=2, refit=True,return_train_score=True)
clf.fit(train[features], train["QuoteConversion_Flag"])
```

    Fitting 5 folds for each of 2 candidates, totalling 10 fits
    

    [Parallel(n_jobs=4)]: Done  10 out of  10 | elapsed:  2.4min finished
    




    GridSearchCV(cv=<generator object _BaseKFold.split at 0x0000000018459888>,
           error_score='raise',
           estimator=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
           max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
           n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=True, subsample=1),
           fit_params=None, iid=True, n_jobs=4,
           param_grid={'nthread': [4], 'objective': ['binary:logistic'], 'learning_rate': [0.05, 0.1], 'max_depth': [6], 'min_child_weight': [11], 'silent': [1], 'subsample': [0.8], 'colsample_bytree': [0.7], 'n_estimators': [5], 'missing': [-999], 'seed': [1337]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring='roc_auc', verbose=2)




```python
clf.grid_scores_
```

    c:\anaconda3\envs\nlp\lib\site-packages\sklearn\model_selection\_search.py:761: DeprecationWarning: The grid_scores_ attribute was deprecated in version 0.18 in favor of the more elaborate cv_results_ attribute. The grid_scores_ attribute will not be available from 0.20
      DeprecationWarning)
    




    [mean: 0.94416, std: 0.00118, params: {'colsample_bytree': 0.7, 'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 11, 'missing': -999, 'n_estimators': 5, 'nthread': 4, 'objective': 'binary:logistic', 'seed': 1337, 'silent': 1, 'subsample': 0.8},
     mean: 0.94589, std: 0.00120, params: {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 11, 'missing': -999, 'n_estimators': 5, 'nthread': 4, 'objective': 'binary:logistic', 'seed': 1337, 'silent': 1, 'subsample': 0.8}]




```python
pd.DataFrame(clf.cv_results_['params'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>colsample_bytree</th>
      <th>learning_rate</th>
      <th>max_depth</th>
      <th>min_child_weight</th>
      <th>missing</th>
      <th>n_estimators</th>
      <th>nthread</th>
      <th>objective</th>
      <th>seed</th>
      <th>silent</th>
      <th>subsample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.7</td>
      <td>0.05</td>
      <td>6</td>
      <td>11</td>
      <td>-999</td>
      <td>5</td>
      <td>4</td>
      <td>binary:logistic</td>
      <td>1337</td>
      <td>1</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.7</td>
      <td>0.10</td>
      <td>6</td>
      <td>11</td>
      <td>-999</td>
      <td>5</td>
      <td>4</td>
      <td>binary:logistic</td>
      <td>1337</td>
      <td>1</td>
      <td>0.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
```

    Raw AUC score: 0.9458947562485674
    colsample_bytree: 0.7
    learning_rate: 0.1
    max_depth: 6
    min_child_weight: 11
    missing: -999
    n_estimators: 5
    nthread: 4
    objective: 'binary:logistic'
    seed: 1337
    silent: 1
    subsample: 0.8
    

    c:\anaconda3\envs\nlp\lib\site-packages\sklearn\model_selection\_search.py:761: DeprecationWarning: The grid_scores_ attribute was deprecated in version 0.18 in favor of the more elaborate cv_results_ attribute. The grid_scores_ attribute will not be available from 0.20
      DeprecationWarning)
    


```python
test_probs = clf.predict_proba(test[features])[:,1]

sample = pd.read_csv('sample_submission.csv')
sample.QuoteConversion_Flag = test_probs
sample.to_csv("xgboost_best_parameter_submission.csv", index=False)
```


```python
clf.best_estimator_.predict_proba(test[features])
```




    array([[0.6988076 , 0.3011924 ],
           [0.6787684 , 0.3212316 ],
           [0.6797658 , 0.32023418],
           ...,
           [0.5018287 , 0.4981713 ],
           [0.6988076 , 0.3011924 ],
           [0.62464744, 0.37535256]], dtype=float32)




```python
kears_result=pd.read_csv('keras_nn_test.csv')
result1=[1 if i>0.5 else 0 for i in kears_result['QuoteConversion_Flag']]
xgb_result=pd.read_csv('xgboost_best_parameter_submission.csv')
result2=[1 if i>0.5 else 0 for i in xgb_result['QuoteConversion_Flag']]
from sklearn import metrics
metrics.accuracy_score(result1,result2)
```




    0.8566004740099864




```python
metrics.confusion_matrix(result1,result2)
```




    array([[148836,  24862],
           [    66,     72]], dtype=int64)
#### 结论
- 对数据的时间进行了预处理
- 对数据中的category类型进行了label化，我觉得有必要对这个进行重新考虑，个人觉得应该使用one-hot进行category的处理，而不是LabelEncoder处理（疑虑）
- Label encoding在某些情况下很有用，但是场景限制很多。再举一例：比如有[dog,cat,dog,mouse,cat]，我们把其转换为[1,2,1,3,2]。这里就产生了一个奇怪的现象：dog和mouse的平均值是cat。所以目前还没有发现标签编码的广泛使用。
- 得到的模型对测试集进行处理，Raw AUC 0.94，而对应的准确率只有85%，实际上并没有实际的分类效果，对于实际上是0的，预测成1的太多了，也就是假阳性太高了，而漏检的也很多。
- 其实模型还有很多可以调整的参数都没有调整，如果对调参有兴趣的可以查看美团的文本分类项目中的例子。

