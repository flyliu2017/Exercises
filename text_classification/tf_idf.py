from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle

class tf_idf(object):
    def __init__(self,dir,path,load=False,**kwargs):
        self.dir=dir

        if not load:

            file_names = ['train_token.txt', 'val_token.txt', 'test_token.txt']
            train_label, train_set = self.divide_text(dir + file_names[0])
            val_label, val_set = self.divide_text(dir + file_names[1])
            test_label, test_set = self.divide_text(dir + file_names[2])
            corpus_set = train_set + val_set + test_set
            tfidf = self.cal_tfidf(corpus_set, **kwargs)

            encoder = preprocessing.LabelEncoder()
            self.train_label = encoder.fit_transform(train_label)
            self.val_label = encoder.fit_transform(val_label)
            self.test_label = encoder.fit_transform(test_label)

            with open(path, 'wb') as f:
                pickle.dump((tfidf, self.train_label, self.val_label, self.test_label), f)
        else:
            with open(path, 'rb') as f:
                tfidf, self.train_label, self.val_label, self.test_label = pickle.load(f)

        self.train_tfidf = tfidf[:50000]
        self.val_tfidf = tfidf[50000:55000]
        self.test_tfidf = tfidf[55000:]

    def divide_text(self,path):
        with open(path, 'r',encoding='utf8') as f:
            lines=f.readlines()
            label_list=[line.split('\t')[0] for line in lines]
            text_list=[line.split('\t')[1] for line in lines]
        return label_list, text_list


    def cal_tfidf(self,texts, **kwargs):
        vectorizer = TfidfVectorizer(**kwargs)
        tfidf = vectorizer.fit_transform(texts)
        return tfidf

    def rf_classify(self,param_grid=None,**kwargs):
        rf = RandomForestClassifier(**kwargs)
        if param_grid:
            cv=GridSearchCV(rf,param_grid,n_jobs=-1,cv=5,verbose=1)
        else:
            cv=rf
        cv.fit(self.train_tfidf, self.train_label)
        print(cv.best_params_)
        score = cv.score(self.train_tfidf, self.train_label)
        print("train score:%s" % score)
        score = cv.score( self.val_tfidf, self.val_label)
        print("validation score:%s" % score)
        y = cv.predict(self.test_tfidf)
        print(classification_report(self.test_label, y))


    def lr_classify(self,**kwargs):
        lr = LogisticRegression(**kwargs)
        lr.fit(self.train_tfidf, self.train_label)
        score = lr.score(self.val_tfidf, self.val_label)
        print("score:%s" % score)
        y = lr.predict(self.test_tfidf)
        print(classification_report(self.test_label, y))




def main(dir,path,load=False,**kwargs):

    m=tf_idf(dir,path,load,**kwargs)
    # m.lr_classify()
    m.rf_classify(param_grid={'max_features':[200]},
                  n_estimators=200,
                  max_depth=None,
                  min_impurity_decrease=0e-6,
                  oob_score=True,
                  random_state=1024,
                  n_jobs=-1)


if __name__ == '__main__':
    dir = 'E:\corpus\cnews\\'
    path=r'E:\corpus\cnews\tfidf_save.pickle'
    main(dir,path,True)

