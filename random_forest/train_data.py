# -*- coding: utf-8 -*-

from datetime import datetime

import pandas as pd
import numpy as np
from matplotlib import pylab

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

from random_forest import smote
import matplotlib.pyplot as plt


def split_data(data):
    data_len = data['y'].count()
    split1 = int(data_len*0.80)
    split2 = int(data_len*0.90)
    train_data = data[:split1]
    cv_data = data[split1:split2]
    test_data = data[split2:]

    return train_data, cv_data, test_data


def resample_train_data(train_data, n, frac):
    numeric_attrs = ['age', 'duration', 'campaign', 'pdays', 'previous',
                 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                 'euribor3m', 'nr.employed',]
    #numeric_attrs = train_data.drop('y',axis=1).columns
    pos_train_data_original = train_data[train_data['y'] == 1]
    pos_train_data = train_data[train_data['y'] == 1]
    new_count = n * pos_train_data['y'].count()
    neg_train_data = train_data[train_data['y'] == 0].sample(frac=frac)
    train_list = []
    if n != 0:
        pos_train_X = pos_train_data[numeric_attrs]
        pos_train_X2 = pd.concat([pos_train_data.drop(numeric_attrs, axis=1)] * n)
        pos_train_X2.index = range(new_count)
        
        s = smote.Smote(pos_train_X.values, N=n, k=3)
        pos_train_X = s.over_sampling()
        pos_train_X = pd.DataFrame(pos_train_X, columns=numeric_attrs, 
                                   index=range(new_count))
        pos_train_data = pd.concat([pos_train_X, pos_train_X2], axis=1)
        pos_train_data = pd.DataFrame(pos_train_data, columns=pos_train_data_original.columns)
        train_list = [pos_train_data, neg_train_data, pos_train_data_original]
    else:
        train_list = [neg_train_data, pos_train_data_original]
    print("Size of positive train data: {} * {}".format(pos_train_data_original['y'].count(), n+1))
    print("Size of negative train data: {} * {}".format(neg_train_data['y'].count(), frac))
    train_data = pd.concat(train_list, axis=0)
    return shuffle(train_data)
    
    



def plot_pr(auc_score, precision, recall, label=None):  
    pylab.figure(num=None, figsize=(6, 5))  
    pylab.xlim([0.0, 1.0])  
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('Recall')  
    pylab.ylabel('Precision')  
    pylab.title('P/R (AUC=%0.2f) / %s' % (auc_score, label))  
    pylab.fill_between(recall, precision, alpha=0.2)  
    pylab.grid(True, linestyle='-', color='0.75')  
    pylab.plot(recall, precision, lw=1)      
    pylab.show()



def train_evaluate(train_data, test_data, classifier, n=1, frac=1.0, threshold = 0.5):  
    # train_data = resample_train_data(train_data, n, frac)
    train_X = train_data.drop('y',axis=1)
    train_y = train_data['y']
    test_X = test_data.drop('y', axis=1)
    test_y = test_data['y']
    
    classifier = classifier.fit(train_X, train_y)
    prodict_prob_y = classifier.predict_proba(test_X)[:,1]
    report = classification_report(test_y, prodict_prob_y > threshold,
                                   target_names = ['no', 'yes'])
    prodict_y = (prodict_prob_y > threshold).astype(int)
    accuracy = np.mean(test_y.values == prodict_y)
    print("Accuracy: {}".format(accuracy))    
    print(report)
    fpr, tpr, thresholds = metrics.roc_curve(test_y, prodict_prob_y)
    precision, recall, thresholds = metrics.precision_recall_curve(test_y, prodict_prob_y)  
    test_auc = metrics.auc(fpr, tpr)
    plot_pr(test_auc, precision, recall, "yes")
    
    return prodict_y
  #  print("AUC: {}".format(test_auc))


def select_model(train_data, cv_data):
    for i in range(1):
      #  print("n_estimators: {}".format(i))
      #  print("threshold: {}".format(i/50.0))
      #  print("n: {}".format(i))
        forest = RandomForestClassifier(n_estimators=400, oob_score=True)
        #lr = LogisticRegression(max_iter=100, C=1, random_state=0)
        train_evaluate(train_data, cv_data, forest, n=7, frac=1.0, threshold=0.4)

    
    
# processed_data = '/Users/tk/Code/custom_code/ML-master/data/processed_bankTraining.csv'
processed_data = '../data/processed_bankTraining.csv'
data = pd.read_csv(processed_data)
train_data, cv_data, test_data = split_data(data)

features_list = train_data.drop('y',axis=1).columns
select_model(train_data, cv_data)
start_time = datetime.now()

print('Training...')
forest = RandomForestClassifier(n_estimators=400, oob_score=True)
prodict_y = train_evaluate(train_data, test_data, forest, n=7, frac=1, threshold=0.40)

end_time = datetime.now()
delta_seconds = (end_time - start_time).seconds

print("Cost time: {}s".format(delta_seconds))

