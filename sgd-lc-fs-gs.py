# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:15:23 2021

@author: user0220
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import time

RND_SEED = 42
CPU_CORES = -1
K_FOLD = 10
N_SAMPLES = 10000
N_FEATURES = 400  ##독립변수=피처
N_CLUSTERS_OF_A_CLASS = 1
N_CLASSES = 3  ##종속변수=클래스
CLUSTERS_SPLIT = [0.35, 0.35, 0.3]
RANDOM_NOISE = 0 # 0.5%의 오차 노이즈
N_INF = 15  ##독립 변수 중 종속 변수와 상관 관계가 있는 성분의 수, 디폴트 2
N_RED = 0  ##독립 변수 중 다른 독립 변수의 선형 조합으로 나타나는 성분의 수, 디폴트 2
N_REP = 0  ##독립 변수 중 단순 중복된 성분의 수, 디폴트 0

##데이터 생성. 분포도 조절이 가능함
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES, 
       n_informative=N_INF, n_redundant=N_RED, n_repeated=N_REP, n_classes=N_CLASSES, 
       n_clusters_per_class=N_CLUSTERS_OF_A_CLASS, weights=CLUSTERS_SPLIT, 
       flip_y=RANDOM_NOISE, random_state=RND_SEED)
# np.save('./Xset',X)
# np.save('./yset',y)


##세트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                       shuffle=True, stratify=y, random_state=RND_SEED)
print('shape:{}\nTrain data: {}, Test data: {}\n'.format(X.shape, X_train.shape[0], X_test.shape[0]))


##피처 스케일링
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


##SGD모델 정의
SGD_PENALTY = "l2" ## l1
SGD_LOOP = 10000
sgd = SGDClassifier(penalty=SGD_PENALTY, max_iter=SGD_LOOP)
from sklearn.svm import SVC
svm = SVC(kernel='linear')


##KFold그리드 서치로 파라미터 탐색
##alpha default=0.0001  ##learning_rate에 영향 줌
from sklearn.model_selection import GridSearchCV
param_grid = {'loss':['hinge','log'],
              'alpha':[0.0001, 0.001, 0.01]}
gs = GridSearchCV(estimator=sgd, param_grid=param_grid, scoring='f1_micro',
                  cv=K_FOLD, n_jobs=CPU_CORES)
gs.fit(X_train_std,y_train)
# print(gs.best_score_)
print('Grid search:',gs.best_params_,'\n')


##f1 스코어 평가
normal_sgd = sgd
normal_sgd.fit(X_train_std, y_train)
print('F1 micro score')
print('  sgd. Train: %.4f  Test: %.4f'%(
    f1_score(y_train, normal_sgd.predict(X_train_std),average='micro'), 
    f1_score(y_test, normal_sgd.predict(X_test_std), average='micro')))
GSsgd = SGDClassifier(penalty=SGD_PENALTY, loss=gs.best_params_['loss'], 
                      alpha=gs.best_params_['alpha'] , max_iter=SGD_LOOP)
CM_sgd = confusion_matrix(y_test, normal_sgd.predict(X_test_std))
normal_GSsgd = GSsgd
normal_GSsgd.fit(X_train_std, y_train)
print('GSsgd. Train: %.4f  Test: %.4f'%(
    f1_score(y_train, normal_GSsgd.predict(X_train_std),average='micro'), 
    f1_score(y_test, normal_GSsgd.predict(X_test_std), average='micro')))
CM_GSsgd = confusion_matrix(y_test, normal_GSsgd.predict(X_test_std))  ##위치바꾸면 왜 피처수 에러생기는거지??

from sklearn.linear_model import LogisticRegression
LRclf = LogisticRegression(random_state=RND_SEED).fit(X_train_std, y_train)
print('logi Reg. Train: %.4f  Test: %.4f\n'%(
    f1_score(y_train, LRclf.predict(X_train_std),average='micro'), 
    f1_score(y_test, LRclf.predict(X_test_std), average='micro')))
# ## L1-based feature selection
# from sklearn.svm import LinearSVC
# lsvc01 = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train_std, y_train)
# lsvc001 = LinearSVC(C=0.001, penalty="l1", dual=False).fit(X_train_std, y_train)
# model = SelectFromModel(lsvc01, prefit=True)
# X_01l1 = model.transform(X)
# print('L1 based feature C=0.01:',X_01l1.shape)
# model = SelectFromModel(lsvc001, prefit=True)
# X_001l1 = model.transform(X)
# print('L1 based feature C=0.001:',X_001l1.shape,'\n')

N_TREES = 100  ##default=100
##Tree based feature selection
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
Extree = ExtraTreesClassifier(n_estimators=N_TREES).fit(X_train, y_train)
X_tree = SelectFromModel(Extree, prefit=True).transform(X_train)
print('Tree based festure sel:',X_tree.shape)

##Tree based score 계산
tree_imp = Extree.feature_importances_
tree_ix = np.argsort(tree_imp)[::-1] 
# X_train_std[:,X_tree.shape[1]]  X_test_std[:,X_tree.shape[1]]
X_tree_train_std = X_train_std[:, tree_ix[:X_tree.shape[1]]]
X_tree_test_std = X_test_std[:, tree_ix[:X_tree.shape[1]]]

tree_fs = sgd
tree_fs.fit(X_tree_train_std,y_train)
tree_fsGS = GSsgd
tree_fsGS.fit(X_tree_train_std,y_train)
print('  tree_fssgd. Train: %.4f  Test: %.4f'%(
    f1_score(y_train, tree_fs.predict(X_tree_train_std),average='micro'), 
    f1_score(y_test, tree_fs.predict(X_tree_test_std), average='micro')))
print('tree_fsGSsgd. Train: %.4f  Test: %.4f\n'%(
    f1_score(y_train, tree_fsGS.predict(X_tree_train_std),average='micro'), 
    f1_score(y_test, tree_fsGS.predict(X_tree_test_std), average='micro')))


##랜덤 포레스트를 활용한 피처 중요도 평가
##나무기반 모델은 표준화, 정규화가 필요하지 않음
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=N_TREES,n_jobs=CPU_CORES,random_state=RND_SEED)
forest.fit(X_train,y_train)
importances = forest.feature_importances_

# ##인덱스 순서대로 피처 중요도 시각화
# plt.title('Random Forest Feature Importances')
# plt.bar(range(X_train.shape[1]),importances,color='lightblue')
# # plt.xticks(range(X_train.shape[1]),)
# plt.show()

##중요도 순서대로 인덱스 시각화 [::-1]를 사용하면 높은순 정렬됨
ix = np.argsort(importances)[::-1]  
plt.title('Random Forest Feature Importances')
plt.bar(range(X_train.shape[1]),importances[ix],color='lightblue')
plt.xticks(range(X_train.shape[1]),ix)
plt.show()

NUM_F = 20
plt.title('Random Forest Feature Importances')
plt.bar(range(NUM_F),importances[ix[:NUM_F]],color='lightblue')
plt.xticks(range(NUM_F),ix[:NUM_F])
plt.show()

X_sel = SelectFromModel(forest,threshold=0.01,prefit=True).transform(X_train)
print('0.01',X_sel.shape)
X_sel = SelectFromModel(forest,threshold=0.02,prefit=True).transform(X_train)
print('0.02',X_sel.shape,'\n')

from sklearn.metrics import classification_report

# from sklearn.metrics import plot_confusion_matrix
# ##모델들: normal_sgd  normal_GSsgd  tree_fs  tree_fsGS
# ##데이터들   X_test_std    y_test    X_tree_test_std
CM_fssgd = confusion_matrix(y_test, tree_fs.predict(X_tree_test_std))
CM_fsGSsgd = confusion_matrix(y_test, tree_fsGS.predict(X_tree_test_std))


from sklearn.model_selection import learning_curve
tr_sizes, tr_scores, te_scores = learning_curve(estimator=normal_sgd, X=X_train_std, y=y_train, 
               train_sizes=np.linspace(0.1, 1, 10),cv=K_FOLD, n_jobs=CPU_CORES)

train_mean = np.mean(tr_scores, axis=1)
train_std = np.std(tr_scores, axis=1)
test_mean = np.mean(te_scores, axis=1)
test_std = np.std(te_scores, axis=1)
plt.plot(tr_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Acc')
plt.fill_between(tr_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(tr_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation Acc')
plt.fill_between(tr_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.title('SGD')
plt.xlabel('Training samples')
plt.ylabel('Acc')
plt.legend(loc='lower right')
plt.show()

tr_sizes, tr_scores, te_scores = learning_curve(estimator=normal_GSsgd, X=X_train_std, y=y_train, 
               train_sizes=np.linspace(0.1, 1, 10),cv=K_FOLD, n_jobs=CPU_CORES)
train_mean = np.mean(tr_scores, axis=1)
train_std = np.std(tr_scores, axis=1)
test_mean = np.mean(te_scores, axis=1)
test_std = np.std(te_scores, axis=1)
plt.plot(tr_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Acc')
plt.fill_between(tr_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(tr_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation Acc')
plt.fill_between(tr_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.title('GS-SGD')
plt.xlabel('Training samples')
plt.ylabel('Acc')
plt.legend(loc='lower right')
plt.show()

tr_sizes, tr_scores, te_scores = learning_curve(estimator=tree_fs, X=X_tree_train_std, y=y_train, 
               train_sizes=np.linspace(0.1, 1, 10),cv=K_FOLD, n_jobs=CPU_CORES)
train_mean = np.mean(tr_scores, axis=1)
train_std = np.std(tr_scores, axis=1)
test_mean = np.mean(te_scores, axis=1)
test_std = np.std(te_scores, axis=1)
plt.plot(tr_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Acc')
plt.fill_between(tr_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(tr_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation Acc')
plt.fill_between(tr_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.title('FS-SGD')
plt.xlabel('Training samples')
plt.ylabel('Acc')
plt.legend(loc='lower right')
plt.show()

tr_sizes, tr_scores, te_scores = learning_curve(estimator=tree_fsGS, X=X_tree_train_std, y=y_train, 
               train_sizes=np.linspace(0.1, 1, 10),cv=K_FOLD, n_jobs=CPU_CORES)
train_mean = np.mean(tr_scores, axis=1)
train_std = np.std(tr_scores, axis=1)
test_mean = np.mean(te_scores, axis=1)
test_std = np.std(te_scores, axis=1)
plt.plot(tr_sizes, train_mean, color='blue', marker='o', label='Training Acc')
plt.fill_between(tr_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(tr_sizes, test_mean, color='green', linestyle='--', marker='s', label='Validation Acc')
plt.fill_between(tr_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.title('FS,GS-SGD')
plt.xlabel('Training samples')
plt.ylabel('Acc')
plt.legend(loc='lower right')
plt.show()

tr_sizes, tr_scores, te_scores = learning_curve(estimator=LRclf, X=X_train_std, y=y_train, 
               train_sizes=np.linspace(0.1, 1, 10),cv=K_FOLD, n_jobs=CPU_CORES)
train_mean = np.mean(tr_scores, axis=1)
train_std = np.std(tr_scores, axis=1)
test_mean = np.mean(te_scores, axis=1)
test_std = np.std(te_scores, axis=1)
plt.plot(tr_sizes, train_mean, color='blue', marker='o', label='Training Acc')
plt.fill_between(tr_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(tr_sizes, test_mean, color='green', linestyle='--', marker='s', label='Validation Acc')
plt.fill_between(tr_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.title('Logistic Regressor')
plt.xlabel('Training samples')
plt.ylabel('Acc')
plt.legend(loc='lower right')
plt.show()

# ##f1 마이크로 스코어를 출력하는 방법
# print('Train F1: %.4f, Test F1: %.4f'%(f1_score(y_train, sgd.predict(X_train_std), average='micro'), f1_score(y_test, sgd.predict(X_test_std), average='micro')))
# print('Train CM:\n {},\n\n Test CM:\n {}'.format(confusion_matrix(y_train, sgd.predict(X_train_std)), confusion_matrix(y_test, sgd.predict(X_test_std))))
# print('Train F1: %.4f, Test F1: %.4f'%(f1_score(y_train, FSsgd.predict(X_train_std[:,FS]), average='micro'), f1_score(y_test, FSsgd.predict(X_test_std[:,FS]), average='micro')))
# print('Train CM:\n {},\n\n Test CM:\n {}'.format(confusion_matrix(y_train, FSsgd.predict(X_train_std[:,FS])), confusion_matrix(y_test, FSsgd.predict(X_test_std[:,FS]))))
