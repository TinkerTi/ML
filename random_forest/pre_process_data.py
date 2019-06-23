import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle


def feature_scaling(data, numeric_attrs):
    for i in numeric_attrs:
        std = data[i].std()
        if std != 0:
            data[i] = (data[i]-data[i].mean()) / std
        else:
            data = data.drop(i, axis=1)
    return data


def encode_cate_attrs(data, cate_attrs):
    data = encode_edu_attrs(data)
    cate_attrs.remove('education')
    for i in cate_attrs:
        dummies_df = pd.get_dummies(data[i])
        dummies_df = dummies_df.rename(columns=lambda x: i+'_'+str(x))
        data = pd.concat([data,dummies_df],axis=1)
        data = data.drop(i, axis=1)
    return data


def encode_bin_attrs(data, bin_attrs):
    for i in bin_attrs:
        data.loc[data[i] == 'no', i] = 0
        data.loc[data[i] == 'yes', i] = 1
    return data


def encode_edu_attrs(data):
    values = ["illiterate", "basic.4y", "basic.6y", "basic.9y",
    "high.school",  "professional.course", "university.degree"]
    levels = range(1,len(values)+1)
    dict_levels = dict(zip(values, levels))
    for v in values:
        data.loc[data['education'] == v, 'education'] = dict_levels[v]
    return data


def trans_num_attrs(data, numeric_attrs):
    bining_num = 10
    bining_attr = 'age'
    data[bining_attr] = pd.qcut(data[bining_attr], bining_num)
    data[bining_attr] = pd.factorize(data[bining_attr])[0]+1
    return data


def fill_unknown(data, bin_attrs, cate_attrs, numeric_attrs):
    for i in bin_attrs+cate_attrs:
        if data[data[i] == 'unknown']['y'].count() <500:
            # delete col containing unknown
            data = data[data[i] != 'unknown']
    data = encode_cate_attrs(data, cate_attrs)
    data = encode_bin_attrs(data, bin_attrs)
    data['y'] = data['y'].map({'no': 0, 'yes': 1}).astype(int)

    return data


def train_predict_unknown(trainX, trainY, testX):
    forest = RandomForestClassifier(n_estimators=100)

    forest = forest.fit(trainX, trainY)
    test_predictY = forest.predict(testX).astype(int)
    return pd.DataFrame(test_predictY,index=testX.index)


def pre_process_data(input_data_path,processed_data_path):
    # input_data_path = "/Users/tk/Code/custom_code/ML-master/data/bankTraining.csv"
    # processed_data_path = '/Users/tk/Code/custom_code/ML-master/data/processed_bankTraining.csv'
    data = pd.read_csv(input_data_path)
    data.info()

    data.drop(['default','month','day_of_week'],axis=1)




    for col in data.columns:
        if type(data[col][0]) is str:
            print(col+"\t"+str(data[data[col] == 'unknown']['y'].count()))

    data = data.drop(['default','day_of_week','month'], axis=1)

    numeric_attrs = ['age', 'duration', 'campaign', 'pdays', 'previous',
                     'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                     'euribor3m', 'nr.employed']
    bin_attrs = [ 'housing', 'loan']
    cate_attrs = ['poutcome', 'education', 'job', 'marital',
                  'contact']

    data = shuffle(data)
    data = fill_unknown(data, bin_attrs, cate_attrs, numeric_attrs)
    data.to_csv(processed_data_path, index=False)

src_input_data_path = "../data/bankTraining.csv"
src_processed_data_path = '../data/src_processed_bankTraining.csv'
test_input_data_path = "../data/bankTest.csv"
test_processed_data_path='../data/test_processed_bankTraining.csv'
pre_process_data(src_input_data_path,src_processed_data_path)
pre_process_data(test_input_data_path,test_processed_data_path)
data = pd.read_csv(src_processed_data_path)
data.info()


