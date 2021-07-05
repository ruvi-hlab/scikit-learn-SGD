#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:34:01 2021

@author: user0220
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 13:28:06 2021

@author: user0220
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

RND_SEED = 42
CPU_CORES = -1
K_FOLD = 10
MAX_ITERS = 10000


print('데이터프레임 로드\n')
X_train_up = pd.read_excel('/Samsung1T/RootUsr/HelperLab/baldal3code/X_train_up.xlsx' ,sheet_name='Sheet1')
y_train_up = pd.read_excel('/Samsung1T/RootUsr/HelperLab/baldal3code/y_train_up.xlsx' ,sheet_name='Sheet1')
X_test = pd.read_excel('/Samsung1T/RootUsr/HelperLab/baldal3code/X_test.xlsx' ,sheet_name='Sheet1')
y_test = pd.read_excel('/Samsung1T/RootUsr/HelperLab/baldal3code/y_test.xlsx' ,sheet_name='Sheet1')
y_train_up = np.array(y_train_up).ravel()
y_test = np.array(y_test).ravel()


print('이름, ID, 판정 제거\n')
X_train_up.drop(['ID','이름', '만나이', '성별', '전문의판정', 'ADHD=2\nADHD 위기=1\n정상=0'], axis=1, inplace=True)
X_test.drop(['ID','이름', '만나이', '성별', '전문의판정', 'ADHD=2\nADHD 위기=1\n정상=0'], axis=1, inplace=True)


print('데이터 스케일링\n')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train_up)
X_test_std = sc.transform(X_test)


from sklearn.linear_model import SGDClassifier
SGD_PENALTY = "l2" ## l1
sgd = SGDClassifier(penalty=SGD_PENALTY, max_iter=MAX_ITERS)


from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=250, random_state=42)
forest.fit(X_train_std, y_train_up)
importances = forest.feature_importances_ 

print('그리드 서치\n')
from sklearn.model_selection import GridSearchCV
param_grid = {'loss':['hinge','log'],
              'alpha':[0.0001, 0.001, 0.002, 0.004, 0.008, 0.01]}
gs = GridSearchCV(estimator=sgd, param_grid=param_grid, scoring='f1_micro',
                  cv=K_FOLD, n_jobs=CPU_CORES)
gs.fit(X_train_std,y_train_up)
print('gs.best score: ',gs.best_score_)
print('Grid search:',gs.best_params_,'\n')

GSsgd = SGDClassifier(penalty=SGD_PENALTY, loss=gs.best_params_['loss'], 
                      alpha=gs.best_params_['alpha'] , max_iter=MAX_ITERS)
sgd1 = GSsgd
sgd1.fit(X_train_std,y_train_up)
print('GSsgd. Train: %.4f  Test: %.4f\n'%(
    f1_score(y_train_up, sgd1.predict(X_train_std),average='micro'), 
    f1_score(y_test, sgd1.predict(X_test_std), average='micro')))
CM_sgd1 = confusion_matrix(y_test, sgd1.predict(X_test_std))

from sklearn.linear_model import LogisticRegression
LRclf = LogisticRegression(random_state=RND_SEED, max_iter=MAX_ITERS).fit(X_train_std, y_train_up)
print('logi Reg. Train: %.4f  Test: %.4f\n'%(
    f1_score(y_train_up, LRclf.predict(X_train_std),average='micro'), 
    f1_score(y_test, LRclf.predict(X_test_std), average='micro')))

from sklearn.ensemble import GradientBoostingClassifier
GBclf = GradientBoostingClassifier(n_estimators=8, learning_rate=0.6, random_state=RND_SEED).fit(X_train_std, y_train_up)
print('Gra Boo. Train: %.4f  Test: %.4f\n'%(
    f1_score(y_train_up, GBclf.predict(X_train_std),average='micro'), 
    f1_score(y_test, GBclf.predict(X_test_std), average='micro')))

from sklearn.ensemble import RandomForestClassifier
RFclf = RandomForestClassifier(n_estimators=64, random_state=RND_SEED, n_jobs=CPU_CORES).fit(X_train_std, y_train_up)
print('Ran For. Train: %.4f  Test: %.4f'%(
    f1_score(y_train_up, RFclf.predict(X_train_std),average='micro'), 
    f1_score(y_test, RFclf.predict(X_test_std), average='micro')))
