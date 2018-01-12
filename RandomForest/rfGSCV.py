import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

fx=open('/root/PycharmProjects/dataset/x1000.txt')
fy=open('/root/PycharmProjects/dataset/y1000.txt')


xraw=np.array([[float(i) for i in line.split() ] for line in fx.readlines()])
yraw=[ float(line) for line in fy.readlines()]
xdata=np.matrix(xraw,dtype=np.float32)

num_features=xdata[0].size
print('num_features: %d' %num_features)

print('trainset size: %d' %len(xdata))



# param_test1 = {'min_samples_split':range(2,20,5),'min_samples_leaf':range(1,10,2)}
# param_test1 = {'max_depth':range(5,40,5)}
param_test1 = {'max_leaf_nodes':range(5,100,3)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(oob_score=False,n_estimators=90,max_features=2,
                                  min_samples_split=17,min_samples_leaf=7,random_state=10,max_depth=10),
                       param_grid = param_test1, scoring='roc_auc',cv=10,return_train_score=True,refit=True)
gsearch1.fit(xdata,yraw)
print(gsearch1.best_params_)
print(gsearch1.best_score_)
es=gsearch1.best_estimator_
print(es.score(xraw,yraw))

cv_result = pd.DataFrame.from_dict(gsearch1.cv_results_)
with open('cv_result.csv','w') as f:
    cv_result.to_csv(f)