# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:15:23 2021

@author: user0220
"""
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from itertools import combinations
import numpy as np
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
 
    def fit(self, X, y):

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features: # 피쳐 부분집합의 성능을 평가
            scores = []
            subsets = []
 
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
 
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
 
        return self
 
    def transform(self, X):
        return X[:, self.indices_]
 
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score # SBS 연속형 피쳐 선택 알고리즘을 sklearn KNN으로 구현
############

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
K_FOLD = 10
N_SAMPLES = 6000
N_FEATURES = 24
N_CLUSTERS_OF_A_CLASS = 1
N_CLASSES = 3
CLUSTERS_DISTRIBUTION = [0.7, 0.2, 0.1]
RANDOM_NOISE = 0.04

##데이터 생성. 분포도 조절이 가능함
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES, n_classes=N_CLASSES, 
                            n_clusters_per_class=N_CLUSTERS_OF_A_CLASS, weights=CLUSTERS_DISTRIBUTION, 
                            flip_y=RANDOM_NOISE, random_state=RND_SEED)

# from sklearn.datasets import make_blobs
# X, y = make_blobs(n_samples=N_SAMPLES, n_features=N_FEATURES, centers=N_CLASSES, cluster_std=15, random_state=RND_SEED)
# ##데이터가 불균형하게 되도록 바꿔주기
# y = np.where(y == 3, 2, y)

plt.scatter(X[:,0], X[:,1], color=np.array(['r','g','b'])[y] )
plt.xlabel('X[0] feature 0')
plt.ylabel('X[1] feature 1')
plt.show()
print('X shape:',X.shape)

##세트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    shuffle=True, stratify=y, random_state=RND_SEED)
print('Train data: {}, Test data: {} \n'.format(X_train.shape[0], X_test.shape[0]))

##SGD모델 정의
SGD_LOSS = "hinge" ## hinge log
SGD_PENALTY = "l2" ## l1
SGD_LOOP = 10000

sgd = SGDClassifier(loss=SGD_LOSS, penalty=SGD_PENALTY, max_iter=SGD_LOOP)
FSsgd = SGDClassifier(loss=SGD_LOSS, penalty=SGD_PENALTY, max_iter=SGD_LOOP)
KFsgd = SGDClassifier(loss=SGD_LOSS, penalty=SGD_PENALTY, max_iter=SGD_LOOP)
FSKFsgd = SGDClassifier(loss=SGD_LOSS, penalty=SGD_PENALTY, max_iter=SGD_LOOP)
GSsgd = SGDClassifier(penalty=SGD_PENALTY, max_iter=SGD_LOOP)

##피처 스케일링
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# # Tree based 피처 설렉션
# from sklearn.ensemble import ExtraTreesClassifier
# clf = ExtraTreesClassifier(n_estimators=50)
# clf = clf.fit(X, y)
# clf.feature_importances_ 
# model = SelectFromModel(clf, prefit=True)
# Xn = model.transform(X)
# print('T.B F.S Xn shape:',Xn.shape)

##피처 선택
# SBS를 구현하여 사용(Sequential Backward Selection)
K_FEATURES = 2 # 반환할 피처 개수를 정하기 위한 파라미터
# 부분집합으로 이해. 1개피처를 반환할때는 20개의 case, 20개의 피처는 1개의 case
import matplotlib.pyplot as plt
sbs = SBS(sgd, k_features=K_FEATURES)
sbs.fit(X_train_std, y_train) # 내부에서 train, test split을 한번 더 진행한다. test set이 섞이지 않도록 함
# 정확도 시각화
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.2, 1.1])
#plt.xlim([0, 20])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()
print('max Acc Features num:',X.shape[1] - sbs.scores_.index(max(sbs.scores_)))
print('selectied Features:',sbs.subsets_[sbs.scores_.index(max(sbs.scores_))])
# 피처가 16개일때 가장 우수한 성능이라면
# 부분집합을 출력하려면 (전체피처수 - 16)
# print(sbs.subsets_[20-16])
# pd 데이터프레임에선 인덱스 위치로 피처 이름을 알 수 있다.

##f1 스코어 평가
sgd.fit(X_train_std, y_train)
print('\nF1-score micro\nSVM-SGD Train Acc: %.4f Test Acc: %.4f'%(sgd.score(X_train_std,y_train),
                                                                  sgd.score(X_test_std,y_test)))
FS = list(sbs.subsets_[sbs.scores_.index(max(sbs.scores_))]) #FS:Features Selected
FSsgd.fit(X_train_std[:,FS], y_train)
print(' FS-SGD Train Acc: %.4f Test Acc: %.4f'%(FSsgd.score(X_train_std[:,FS],y_train),
                                                FSsgd.score(X_test_std[:,FS],y_test)))

##교차 검증을 이용한 모델 평가
print('\n교차 검증을 이용한 모델의 평균 성능 평가')
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=sgd, X=X_train_std, y=y_train, cv=K_FOLD, n_jobs=CPU_CORES)
print('SGD Train score\nmean: %.4f  std: %.4f' % (np.mean(scores), np.std(scores)))
print('Acc: {}'.format(scores[:]))
scores = cross_val_score(estimator=FSsgd, X=X_train_std[:,FS], y=y_train, cv=K_FOLD, n_jobs=CPU_CORES)
print('\nFS-SGD Train score\nmean: %.4f  std: %.4f' % (np.mean(scores), np.std(scores)))
print('Acc: %s' %scores)
scores = cross_val_score(estimator=sgd, X=X_test_std, y=y_test, cv=K_FOLD, n_jobs=CPU_CORES)
print('\nSGD Test score\nmean: %.4f  std: %.4f' % (np.mean(scores), np.std(scores)))
print('Acc: %s' %scores)
scores = cross_val_score(estimator=FSsgd, X=X_test_std[:,FS], y=y_test, cv=K_FOLD, n_jobs=CPU_CORES)
print('\nFS-SGD Test score\nmean: %.4f  std: %.4f' % (np.mean(scores), np.std(scores)))
print('Acc: %s' %scores)


##K폴드 교차 검증을 이용한 모델 평가
print('\nk-fold 교차 모델 학습 및 평가')
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=RND_SEED)
scores = []
for train, test in kfold.split(X_train_std, y_train):
    KFsgd.fit(X_train_std[train], y_train[train])
    score = KFsgd.score(X_train[test], y_train[test])
    scores.append(score)
    # print('폴드: %2d, 클래스 분포: %s, 정확도: %.4f' % (k+1, np.bincount(y_train[train]), score))
print('k fold SGD Train f1 mean: %.4f  std: %.4f' % (np.mean(scores), np.std(scores)))
print('k fold SGD Test: %.4f' % (KFsgd.score(X_test_std,y_test)))

kfold = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=RND_SEED)
scores = []
for train, test in kfold.split(X_train_std[:,FS], y_train):
    FSKFsgd.fit(X_train_std[train], y_train[train])
    score = FSKFsgd.score(X_train[test], y_train[test])
    scores.append(score)    
    # print('\n폴드: %2d, 클래스 분포: %s, 정확도: %.4f' % (k+1, np.bincount(y_train[train]), score))    
print('F.sel k fold SGD Train f1 mean: %.4f  std: %.4f' % (np.mean(scores), np.std(scores)))
# print('F.sel k fold SGD Test: %.4f' % (FSKFsgd.score(X_test_std[:,FS],y_test)))


from sklearn.model_selection import GridSearchCV
##alpha default=0.0001  ##learning_rate에 영향 줌
param_grid = {'loss':['log','hinge','modified_huber','squared_hinge'],
              'alpha':[0.0001, 0.001, 0.01, 0.1, 1]}
gs = GridSearchCV(estimator=GSsgd, param_grid=param_grid, scoring='accuracy',
                  cv=K_FOLD, n_jobs=CPU_CORES)
gs.fit(X_train_std,y_train)
print(gs.best_score_)
print(gs.best_params_)

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
