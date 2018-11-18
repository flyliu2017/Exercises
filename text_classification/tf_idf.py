from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pickle, operator,os
from functools import reduce
from scipy import sparse
import pandas as pd
import numpy as np


class tf_idf(object):
    def __init__(self, dir, path, load=False, **kwargs):
        self.dir = dir

        if not load:

            file_names = ['train_token.txt', 'val_token.txt', 'test_token.txt']
            train_label, train_set = self.divide_text(self.dir + file_names[0])
            val_label, val_set = self.divide_text(self.dir + file_names[1])
            test_label, test_set = self.divide_text(self.dir + file_names[2])
            corpus_set = train_set + val_set + test_set
            self.tfidf, self.feature_names = self.cal_tfidf(corpus_set, **kwargs)

            encoder = preprocessing.LabelEncoder()
            self.train_label = encoder.fit_transform(train_label)
            self.val_label = encoder.fit_transform(val_label)
            self.test_label = encoder.fit_transform(test_label)

            self.save_params(path)
        else:
            self.load_params(path)

        self.train_tfidf = self.tfidf[:50000]
        self.val_tfidf = self.tfidf[50000:55000]
        self.test_tfidf = self.tfidf[55000:]
    
    def save_params(self,path):
        with open(path, 'wb') as f:
            pickle.dump((self.tfidf, self.feature_names, self.train_label, self.val_label, self.test_label), f)
            
    def load_params(self,path):
        with open(path, 'rb') as f:
            self.tfidf, self.feature_names, self.train_label, self.val_label, self.test_label = pickle.load(f)

    def divide_text(self, path):
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            label_list = [line.split('\t')[0] for line in lines]
            text_list = [line.split('\t')[1] for line in lines]
        return label_list, text_list

    def cal_tfidf(self, texts, **kwargs):
        vectorizer = TfidfVectorizer(**kwargs)
        tfidf = vectorizer.fit_transform(texts)
        return tfidf, vectorizer.get_feature_names()

    def rf_classify(self, **kwargs):
        rf = RandomForestClassifier(**kwargs)
        # if param_grid:
        #     cv=GridSearchCV(rf,param_grid,n_jobs=-1,cv=5,verbose=1)
        # else:
        #     cv=rf
        rf.fit(self.train_tfidf, self.train_label)
        score = rf.score(self.train_tfidf, self.train_label)
        print("train score:%s" % score)
        score = rf.score(self.val_tfidf, self.val_label)
        print("validation score:%s" % score)
        y = rf.predict(self.test_tfidf)
        print(classification_report(self.test_label, y))

        return rf.feature_importances_

    def lr_classify(self, **kwargs):
        lr = LogisticRegression(**kwargs)
        lr.fit(self.train_tfidf, self.train_label)
        score = lr.score(self.val_tfidf, self.val_label)
        print("score:%s" % score)
        y = lr.predict(self.test_tfidf)
        print(classification_report(self.test_label, y))
        
    def gb_classify(self, **kwargs):
        gb = GradientBoostingClassifier(**kwargs)
        gb.fit(self.train_tfidf, self.train_label)
        score = gb.score(self.val_tfidf, self.val_label)
        print("score:%s" % score)
        y = gb.predict(self.test_tfidf)
        print(classification_report(self.test_label, y))

    def param_grid(self,estimator, params, param_grid):
        total = reduce(operator.mul, [len(v) for v in param_grid.values()])

        for n in range(total):
            extra_params = get_params(param_grid, n)
            p = dict(params)
            p.update(extra_params)
            print(extra_params)
            print('=' * 20)
            estimator(**p)

    def feature_selection(self, path,feature_importance, length_list):
        if not os.path.exists(path):
            os.makedirs(path)
        coo=self.tfidf.tocoo()
        fn=[self.feature_names[i] for i in coo.col]
        df=pd.DataFrame({'row':coo.row,'col':fn,'data':coo.data})
        # df.to_csv(self.dir+'tfidf.csv')
        fs = pd.Series(feature_importance, self.feature_names)
        fs.to_csv(path + 'feature_importance.csv')
        sort_fs = fs.sort_values(ascending=False)
        for n in length_list:
            index = sort_fs[:n].keys()
            self.feature_names=list(index)
            selected=df[df['col'].isin(index)]
            selected.to_csv(path + 'selected_tfidf_%s.csv' % n)
            col=preprocessing.LabelEncoder().fit_transform(selected.col)
            newcoo = sparse.coo_matrix((selected.data, (selected.row,col )),
                                             shape=(coo.shape[0], n))
            self.tfidf=newcoo.tocsr()
            self.save_params(path+'selected_tfidf_%s.pickle' % n)

def get_params(param_dict, n):
    params = {}
    for key, value in param_dict.items():
        i = len(value)
        r = n % i
        n = n // i
        params[key] = value[r]
    return params


def main(dir, path, load=False, **kwargs):
    m = tf_idf(dir, path, load, **kwargs)
    # m.lr_classify()
    # params = dict(n_estimators=100,
    #               max_features=None,
    #               max_depth=None,
    #               min_impurity_decrease=0e-6,
    #               oob_score=True,
    #               random_state=1024,
    #               n_jobs=-1)
    params={}
    param_grid = {'max_features': [100,200],'max_depth':[30,40]}
    # param_grid = {'max_features': [200]}
    m.param_grid(m.gb_classify,params,param_grid)
    # fi = m.rf_classify(**params)
    # m.feature_selection(dir+'full\\',fi, [10000, 20000, 30000])


if __name__ == '__main__':
    dir = 'E:\corpus\cnews\\'
    # path = r'E:\corpus\cnews\tfidf_save.pickle'
    path=dir + 'selected_tfidf_10000.pickle'
    main(dir, path, load=True)
