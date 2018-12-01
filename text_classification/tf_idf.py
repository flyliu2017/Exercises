from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import pickle, operator,os,time
from functools import reduce
from scipy import sparse
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from mlxtend.classifier import StackingCVClassifier

class tf_idf(object):
    def __init__(self, out_dir, path, load=False, **kwargs):
        self.out_dir = out_dir

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(path)

        if not load:

            file_names = ['train_token.txt', 'val_token.txt', 'test_token.txt']
            train_label, train_set = self.divide_text(self.out_dir + file_names[0], shuffle=True)
            val_label, val_set = self.divide_text(self.out_dir + file_names[1], shuffle=True)
            test_label, test_set = self.divide_text(self.out_dir + file_names[2], shuffle=True)
            corpus_set = train_set + val_set + test_set
            self.tfidf, self.feature_names = self.cal_tfidf(corpus_set,token_pattern=r"(?u)\b\w+\b", **kwargs)

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
    

    def divide_text(self, path,shuffle=False):
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            if shuffle:
                np.random.shuffle(lines)
            label_list = [line.split('\t')[0] for line in lines]
            text_list = [line.split('\t')[1] for line in lines]
        return label_list, text_list

    def cal_tfidf(self, texts, **kwargs):
        vectorizer = TfidfVectorizer(**kwargs)
        tfidf = vectorizer.fit_transform(texts)
        return tfidf, vectorizer.get_feature_names()


    def classify(self, classifier,  **kwargs):
        clf = classifier(**kwargs)
        start = time.time()
        print(time.asctime())
        clf.fit(self.train_tfidf, self.train_label)
        score = clf.score(self.val_tfidf, self.val_label)
        print("score:%s" % score)
        y = clf.predict(self.test_tfidf)
        print(classification_report(self.test_label, y,digits=5))
        print(accuracy_score(self.test_label,y))
        cost=time.time()-start
        print('time cost:%.2f min'% (cost/60))
        if hasattr(clf,'feature_importances_'):
            return clf.feature_importances_
        return None

    def param_grid(self,classifier, params, param_grid):
        total = reduce(operator.mul, [len(v) for v in param_grid.values()])

        for n in range(total):
            extra_params = get_params(param_grid, n)
            p = dict(params)
            p.update(extra_params)
            print(extra_params)
            print('=' * 20)
            self.classify(classifier,**p)

    def stacking_param_grid(self, params, param_grid):
        total = reduce(operator.mul, [len(v) for v in param_grid.values()])

        for n in range(total):
            extra_params = get_params(param_grid, n)
            p = dict(params)
            p.update(extra_params)
            print(extra_params)
            print('=' * 20)
            self.stacking(**p)

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

    def stacking(self,meta_clf,**kwargs):
        rf=RandomForestClassifier(n_estimators=100,
                                  max_features=600,
                                  max_depth=None,
                                  min_impurity_decrease=0e-6,
                                  oob_score=True,
                                  random_state=1024,
                                  n_jobs=-1)
        lr=LogisticRegression(solver='lbfgs',
                              max_iter=200,
                              n_jobs=-1)
        gb=GradientBoostingClassifier(n_estimators=500,
                                      max_features=300,
                                      max_depth=20)

        # xg=XGBClassifier()
        clfs=[rf,lr,gb]
        meta_clf=meta_clf(**kwargs)
        self.classify(StackingCVClassifier,classifiers=clfs,meta_classifier=meta_clf,use_probas=True)
    
def get_params(param_grid, n):
    params = {}
    for key, value in param_grid.items():
        i = len(value)
        r = n % i
        n = n // i
        params[key] = value[r]

    return params



def main(dir, path, load=False, **kwargs):
    m = tf_idf(dir, path, load, **kwargs)

    # params = dict(n_jobs=3)
    # param_grid = {'solver': ['newton-cg', 'lbfgs',  'sag', 'saga'],'max_iter':[200]}
    # m.param_grid(LogisticRegression,params,param_grid)
    m.classify(LogisticRegression)

    # params = dict(n_estimators=100,
    #               max_features=None,
    #               max_depth=3,
    #               min_impurity_decrease=0e-6,
    #               random_state=1024,
    #               verbose=1
    #               )
    # param_grid = {'max_features': [200,400],'max_depth':[20],'n_estimators':[500]}
    # m.param_grid(GradientBoostingClassifier,params,param_grid)

    # params = dict(n_estimators=10,
    #               max_depth=3,
    #               objective='multi:softmax',
    #               random_state=1024,
    #               n_jobs=-1)
    # param_grid = {'max_depth':[30],'learning_rate':[0.1],'n_estimators':[100]}
    # m.param_grid(XGBClassifier,params,param_grid)

    # params = dict()
    # param_grid = {'C':[1]}
    # m.param_grid(SVC,params,param_grid)

    params = dict(n_estimators=100,
                  max_features=None,
                  max_depth=None,
                  min_impurity_decrease=0e-6,
                  oob_score=True,
                  random_state=1024,
                  verbose=1,
                  n_jobs=-1)
    param_grid = {'max_features': [None]}
    # m.param_grid(RandomForestClassifier,params,param_grid)


    # m.stacking(LogisticRegression)
    # m.stacking_param_grid(LogisticRegression)

    fi = m.classify(RandomForestClassifier,**params)
    m.feature_selection(dir+'shuffled\\',fi, [10000, 20000, 30000])


if __name__ == '__main__':
    out_dir = 'E:\corpus\cnews\\'
    # path = r'E:\corpus\cnews\tfidf_save.pickle'
    path= out_dir + r'len_one\tfidf_save.pickle'
    # path= out_dir + r'len_one\selected_tfidf_10000.pickle'
    main(out_dir, path, load=False)
