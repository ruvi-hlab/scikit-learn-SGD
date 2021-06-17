# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:15:23 2021

@author: user0220
"""



import numpy as np
import pandas as pd

from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_validate

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.metrics import f1_score

import time

RND_SEED = 42
CPU_CORES = -1
K_FOLD = 8
N_SAMPLES = 10000
N_FEATURES = 200  ##독립변수=피처
N_CLUSTERS_OF_A_CLASS = 1
N_CLASSES = 3  ##종속변수=클래스
CLUSTERS_SPLIT = [0.7, 0.2, 0.1]
RANDOM_NOISE = 0.005 # 0.5%의 오차 노이즈
N_INF = 15  ##독립 변수 중 종속 변수와 상관 관계가 있는 성분의 수, 디폴트 2
N_RED = 10  ##독립 변수 중 다른 독립 변수의 선형 조합으로 나타나는 성분의 수, 디폴트 2
N_REP = 5  ##독립 변수 중 단순 중복된 성분의 수, 디폴트 0

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

##SGD모델 정의
SGD_PENALTY = "l2" ## l1
SGD_LOOP = 1000
sgd = SGDClassifier(penalty=SGD_PENALTY, max_iter=SGD_LOOP)
from sklearn.svm import SVC
svm = SVC(kernel='linear')

##피처 스케일링
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

##KFold그리드 서치로 파라미터 탐색
from sklearn.model_selection import GridSearchCV
##alpha default=0.0001  ##learning_rate에 영향 줌
param_grid = {'loss':['hinge','log'],
              'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 1]}
gs = GridSearchCV(estimator=sgd, param_grid=param_grid, scoring='f1_micro',
                  cv=K_FOLD, n_jobs=CPU_CORES)
gs.fit(X_train_std,y_train)
# print(gs.best_score_)
print('Grid search:',gs.best_params_,'\n')


##f1 스코어 평가
normal_sgd = sgd
normal_sgd.fit(X_train_std, y_train)
print('F1-score micro\n  sgd. Train, Acc: %.4f Test Acc: %.4f'%(normal_sgd.score(X_train_std,y_train),
                                                    normal_sgd.score(X_test_std,y_test)))
GSsgd = SGDClassifier(penalty=SGD_PENALTY, loss=gs.best_params_['loss'], 
                      alpha=gs.best_params_['alpha'] , max_iter=SGD_LOOP)
normal_GSsgd = GSsgd
normal_GSsgd.fit(X_train_std, y_train)
print('GSsgd. Train, Acc: %.4f Test Acc: %.4f'%(normal_GSsgd.score(X_train_std,y_train),
                                               normal_GSsgd.score(X_test_std,y_test)))
normal_svm = svm
normal_svm.fit(X_train_std, y_train)
print('  svm. Train, Acc: %.4f Test Acc: %.4f\n'%(normal_svm.score(X_train_std,y_train),
                                              normal_svm.score(X_test_std,y_test)))

N_TREES = 100  ##default=100
##Tree based feature selection
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=N_TREES).fit(X_train_std, y_train)
clf.feature_importances_ 
model = SelectFromModel(clf, prefit=True)
X_tree = model.transform(X)
print('Tree based festure sel:',X_tree.shape)

## L1-based feature selection
from sklearn.svm import LinearSVC
lsvc01 = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train_std, y_train)
lsvc001 = LinearSVC(C=0.001, penalty="l1", dual=False).fit(X_train_std, y_train)
model = SelectFromModel(lsvc01, prefit=True)
X_01l1 = model.transform(X)
print('L1 based feature C=0.01:',X_01l1.shape)
model = SelectFromModel(lsvc001, prefit=True)
X_001l1 = model.transform(X)
print('L1 based feature C=0.001:',X_001l1.shape)

##Tree based score 계산
tree_imp = clf.feature_importances_
tree_ix = np.argsort(tree_imp)[::-1] 
# X_train_std[:,X_tree.shape[1]]  X_test_std[:,X_tree.shape[1]]
X_tree_train_std = X_train_std[:, tree_ix[:X_tree.shape[1]]]
X_tree_test_std = X_test_std[:, tree_ix[:X_tree.shape[1]]]
tree_fs = GSsgd
tree_fs.fit(X_tree_train_std,y_train)
print('tree_fsGSsgd. Train, Acc: %.4f Test Acc: %.4f\n'%(tree_fs.score(X_tree_train_std,y_train),
                                              tree_fs.score(X_tree_test_std,y_test)))

# ##L1 베이스 피처선택은 어떻게 주요피처 정렬하는지 안나와있어서 모르겠음
# lsvc001.feature_importances_
# lsvc01_imp = lsvc01.feature_importances_
# lsvc01_ix = np.argsort(lsvc01_imp)[::-1] 
# # X_train_std[:,X_tree.shape[1]]  X_test_std[:,X_tree.shape[1]]
# X_lsvc01_train_std = X_train_std[:, lsvc01_ix[:X_01l1.shape[1]]]
# X_lsvc01_test_std = X_test_std[:, lsvc01_ix[:X_01l1.shape[1]]]
# lsvc01_fs = GSsgd음
# lsvc01_fs.fit(X_lsvc01_train_std,y_train)
# print('Lsvc01_fsGSsgd. Train, Acc: %.4f Test Acc: %.4f\n'%(lsvc01_fs.score(X_lsvc01_train_std,y_train),
#                                               lsvc01_fs.score(X_lsvc01_test_std,y_test)))



##랜덤 포레스트를 활용한 피처 중요도 평가
##나무기반 모델은 표준화, 정규화가 필요하지 않음
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=N_TREES,n_jobs=CPU_CORES,random_state=RND_SEED)
forest.fit(X_train,y_train)
importances = forest.feature_importances_
plt.title('Random Forest Feature Importances')
plt.bar(range(X_train.shape[1]),importances,color='lightblue')
# plt.xticks(range(X_train.shape[1]),)
plt.show()

plt.title('Random Forest Feature Importances')
ix = np.argsort(importances)[::-1]  ##인덱스를 크기순으로 정렬하는 법, 높은 중요도순으로 ix저장
plt.bar(range(X_train.shape[1]),importances[ix],color='lightblue')
plt.xticks(range(X_train.shape[1]),ix)
plt.show()

NUM_F = 30
plt.title('Random Forest Feature Importances')
plt.bar(range(NUM_F),importances[ix[:NUM_F]],color='lightblue')
plt.xticks(range(NUM_F),ix[:NUM_F])
plt.show()

X_sel = SelectFromModel(forest,threshold=0.01,prefit=True).transform(X_train)
print('0.01',X_sel.shape)
X_sel = SelectFromModel(forest,threshold=0.02,prefit=True).transform(X_train)
print('0.02',X_sel.shape,'\n')


##피처 선택 스코어
X_sel_train_std = X_train_std[:, ix[:NUM_F]]
X_sel_test_std = X_test_std[:, ix[:NUM_F]]
FSsgd = sgd
FSsgd.fit(X_sel_train_std,y_train)
print('fssgd. Train, Acc: %.4f Test Acc: %.4f\n'%(FSsgd.score(X_sel_train_std,y_train),
                                              FSsgd.score(X_sel_test_std,y_test)))

##그리드서치+피처 선택 스코어
X_sel_train_std = X_train_std[:, ix[:NUM_F]]
X_sel_test_std = X_test_std[:, ix[:NUM_F]]
GSFSsgd = GSsgd
GSFSsgd.fit(X_sel_train_std,y_train)
print('GSFSsgd. Train, Acc: %.4f Test Acc: %.4f\n'%(GSFSsgd.score(X_sel_train_std,y_train),
                                              GSFSsgd.score(X_sel_test_std,y_test)))

# ##교차 검증을 이용한 모델 평가
# print('\n교차 검증을 이용한 모델의 평균 성능 평가')
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(estimator=sgd, X=X_train_std, y=y_train, cv=K_FOLD, n_jobs=CPU_CORES)
# print('SGD Train score\nmean: %.4f  std: %.4f' % (np.mean(scores), np.std(scores)))
# print('Acc: {}'.format(scores[:]))
# # scores = cross_val_score(estimator=FSsgd, X=X_train_std[:,FS], y=y_train, cv=K_FOLD, n_jobs=CPU_CORES)
# # print('\nFS-SGD Train score\nmean: %.4f  std: %.4f' % (np.mean(scores), np.std(scores)))
# # print('Acc: %s' %scores)
# scores = cross_val_score(estimator=sgd, X=X_test_std, y=y_test, cv=K_FOLD, n_jobs=CPU_CORES)
# print('\nSGD Test score\nmean: %.4f  std: %.4f' % (np.mean(scores), np.std(scores)))
# print('Acc: %s' %scores)
# # scores = cross_val_score(estimator=FSsgd, X=X_test_std[:,FS], y=y_test, cv=K_FOLD, n_jobs=CPU_CORES)
# # print('\nFS-SGD Test score\nmean: %.4f  std: %.4f' % (np.mean(scores), np.std(scores)))
# # print('Acc: %s' %scores)




# ##f1 마이크로 스코어를 출력하는 방법
# print('Train F1: %.4f, Test F1: %.4f'%(f1_score(y_train, sgd.predict(X_train_std), average='micro'), f1_score(y_test, sgd.predict(X_test_std), average='micro')))
# print('Train CM:\n {},\n\n Test CM:\n {}'.format(confusion_matrix(y_train, sgd.predict(X_train_std)), confusion_matrix(y_test, sgd.predict(X_test_std))))
# print('Train F1: %.4f, Test F1: %.4f'%(f1_score(y_train, FSsgd.predict(X_train_std[:,FS]), average='micro'), f1_score(y_test, FSsgd.predict(X_test_std[:,FS]), average='micro')))
# print('Train CM:\n {},\n\n Test CM:\n {}'.format(confusion_matrix(y_train, FSsgd.predict(X_train_std[:,FS])), confusion_matrix(y_test, FSsgd.predict(X_test_std[:,FS]))))




'''
# 분류기

start = time.time() 

clafi = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
#clf = SGDClassifier(loss="hinge", alpha=0.001, max_iter=10)
clafi.fit(X_train, y_train)

print("SVM-SGD clf Time:", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

# # 교차검증, 5등분
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(clafi, X_train, y_train, cv=5)
# print("@@@ %0.4f acc with a s.d of %0.4f @@@" % (scores.mean(), scores.std()))

# # 그리드 서치
# from sklearn import svm
# from sklearn.model_selection import GridSearchCV
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# svc = svm.SVC()
# clf = GridSearchCV(svc, parameters)
# clf.fit(X_train, y_train)
# print(sorted(clf.cv_results_.keys()) )


# 예측: 학습 셋, 검증 셋

start = time.time() 

train_pred = clafi.predict(X_train)
test_pred = clafi.predict(X_test)

print("SVM-SGD pred Time:", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

# # 정확도를 소수점 4째 자리까지 표기
tr_acc = np.round(accuracy_score(y_train, train_pred),4)
te_acc = np.round(accuracy_score(y_test, test_pred),4)
print('SVM-SGD Train Acc: {}, Test Acc: {}'.format(tr_acc, te_acc))



# 다른 dict 생성 방법    
# pr_val_dic = { i : 1  for  i  in  range ( 2 , len(feature_-2) )}  
# pr_ran_dic = { i : 1  for  i  in  range ( 2 , len(feature_-2) )}  

# 혼동 행렬, f1 스코어
print('Train F1: {}, Test F1: {}'.format(f1_score(y_train, train_pred, average='micro'), f1_score(y_test, test_pred, average='micro')))
print('Train CM:\n {},\n\n Test CM:\n {}'.format(confusion_matrix(y_train, train_pred), confusion_matrix(y_test, test_pred)))


# # 결정영역 시각화 고차원 대응을 위한 부분
# pr_val_dic = {}
# pr_ran_dic = {}
# for i in range(X.shape[1] - 2):
#     pr_val_dic[2+i] = 1
#     pr_ran_dic[2+i] = 1

start = time.time() 

####### 시각화 삭제, 시각화 실행시간 삭제

print('\n')
############################################################

start = time.time() 

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

print("SVM clf Time:", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

# 교차검증, 5등분
from sklearn.model_selection import cross_val_score
scores = cross_val_score(svm, X_train, y_train, cv=5)
print("@@@ %0.4f acc with a s.d of %0.4f @@@" % (scores.mean(), scores.std()))


start = time.time() 

train_pred = svm.predict(X_train)
test_pred = svm.predict(X_test)

print("SVM pred Time:", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

# # 정확도를 소수점 4째 자리까지 표기
tr_acc = np.round(accuracy_score(y_train, train_pred),4)
te_acc = np.round(accuracy_score(y_test, test_pred),4)
print('SVM Train Acc: {}, Test Acc: {}'.format(tr_acc, te_acc))

tr_f1 = f1_score(y_train, train_pred, average='micro')
te_f1 = f1_score(y_test, test_pred, average='micro')
print('SVM Train F1: {}, Test F1: {}'.format(tr_f1, te_f1))
print('SVM Train CM:\n {},\n\n Test CM:\n {}'.format(confusion_matrix(y_train, train_pred), confusion_matrix(y_test, test_pred)))

####### 시각화 삭제, 시각화 실행시간 삭제

#######테스트 입력 삭제
'''
