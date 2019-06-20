# -*-coding:utf-8-*-
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
# from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 字符转码 方便显示
# reload(sys)
# sys.setdefaultencoding('utf-8')

data = pd.read_csv('bankTraining.csv')

# x=data[["age","job","education","housing"]]
x = data
y = data["y"]
del x["y"]

x = x.to_dict(orient="records")

# train_test_split函数用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签。
# random_state：是随机数的种子。
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

# DictVectorizer的处理对象是符号化(非数字化)的但是具有一定结构的特征数据，如字典等，将符号转成数字0/1表示。
transfer = DictVectorizer()

x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# estimator = DecisionTreeClassifier(criterion="entropy",max_depth=8)
estimator = RandomForestClassifier()
# 加入网格搜索与交叉验证
# 参数准备
param_dict = {"n_estimators":[120,200,300,500,800,1200],"max_depth":[5,8,15,25,30]}
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
estimator.fit(x_train, y_train)

# 模型评估
# 方法1：直接比对真实值和预测值
y_predict = estimator.predict(x_test)

print("y_predict:",y_test == y_predict)

# 方法2：计算准确率
score = estimator.score(x_test, y_test)
print("准确率为：", score)

# 最佳参数：
print("最佳参数：", estimator.best_params_)
print("最佳结果：", estimator.best_score_)
print("最佳估计器：", estimator.best_estimator_)
print("交叉验证结果：", estimator.best_estimator_)

# export_graphviz(estimator, out_file="tree.dot",feature_names=transfer.get_feature_names())



