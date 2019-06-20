import pandas as pd
import sklearn.utils as su

data = pd.read_csv("/Users/tk/Private/MEM/201902/ML/bankTraining.csv")
data.info()

data.describe()

for col in data.columns:
    if type(data[col][0]) is str:
        print(col+"\t"+str(data[data[col] == 'unknown']['y'].count()))


def preprocess_data():
    input_data_path = "/Users/tk/Private/MEM/201902/ML/bankTraining.csv"
    processed_data_path = '/Users/tk/Private/MEM/201902/ML/processed_bankTraining.csv'
    print("Loading data...")
    data = pd.read_csv(input_data_path)
    print("Preprocessing data...")
    numeric_attrs = ['age', 'duration', 'campaign', 'pdays', 'previous',
                     'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                     'euribor3m', 'nr.employed']
    bin_attrs = ['default', 'housing', 'loan']
    cate_attrs = ['poutcome', 'education', 'job', 'marital',
                  'contact', 'month','day_of_week']

    data = su.shuffle(data)
    data = fill_unknown(data, bin_attrs, cate_attrs, numeric_attrs)
    data.to_csv(processed_data_path, index=False)


def fill_unknown(data, bin_attrs, cate_attrs, numeric_attrs):
    # fill_attrs = ['education', 'default', 'housing', 'loan']
    fill_attrs = []
    for i in bin_attrs+cate_attrs:
        if data[data[i] == 'unknown']['y'].count() < 500:
            # delete col containing unknown
            data = data[data[i] != 'unknown']
        else:
            fill_attrs.append(i)
    bining_attr=['duration'];
    data = encode_cate_attrs(data, cate_attrs)
    data = encode_bin_attrs(data, bin_attrs)
    data= trans_duration_attrs(data,bining_attr)
    data = trans_num_attrs(data, numeric_attrs)

    data['y'] = data['y'].map({'no': 0, 'yes': 1}).astype(int)
    for i in fill_attrs:
        test_data = data[data[i] == 'unknown']
        testX = test_data.drop(fill_attrs, axis=1)
        train_data = data[data[i] != 'unknown']
        trainY = train_data[i]
        trainX = train_data.drop(fill_attrs, axis=1)
        test_data[i] = train_predict_unknown(trainX, trainY, testX)
        data = pd.concat([train_data, test_data])

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

def encode_edu_attrs(data):
    values = ["illiterate", "basic.4y", "basic.6y", "basic.9y",
    "high.school",  "professional.course", "university.degree"]
    levels = range(1,len(values)+1)
    dict_levels = dict(zip(values, levels))
    for v in values:
        data.loc[data['education'] == v, 'education'] = dict_levels[v]
    return data

def encode_bin_attrs(data, bin_attrs):
    for i in bin_attrs:
        data.loc[data[i] == 'no', i] = 0
        data.loc[data[i] == 'yes', i] = 1
    return data



def trans_duration_attrs(data,bining_attr):
    data[bining_attr] = pd.qcut(data[bining_attr], 0.75)
    data[bining_attr] = pd.factorize(data[bining_attr])[0]+1

def trans_num_attrs(data, numeric_attrs):
    for i in numeric_attrs:
        scaler = preprocessing.StandardScaler()
        data[i] = scaler.fit_transform(data[i])



def split_data(data):
    data_len = data['y'].count()
    split1 = int(data_len*0.6)
    split2 = int(data_len*0.8)
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
    return su.shuffle(train_data)

def train_evaluate(train_data, test_data, classifier, n=1, frac=1.0, threshold = 0.5):
    train_data = resample_train_data(train_data, n, frac)
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





