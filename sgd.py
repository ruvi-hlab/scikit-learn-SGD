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
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier

from mlxtend.plotting import plot_decision_regions

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate


from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.metrics import f1_score
from sklearn.svm import SVC

import time




# RND_SEED = 42
# N_SAMPLES = 500 # 데이터 개수
# N_FEATURE = 12 #12~120
# N_CENTERS = 3+1  # 클러스터 개수(=정답개수)
# CLUSTER_STD = 80
# CENTER_DISTANCE = (0,20)
# # 데이터 생성
# X, y = make_blobs(n_samples=N_SAMPLES, n_features=N_FEATURE, centers=N_CENTERS, cluster_std=CLUSTER_STD, center_box=CENTER_DISTANCE, random_state=1)
# # 데이터가 불균형하게 되도록 바꿔주기
# y = np.where(y == 3, 2, y)
# # y = np.where(y == 4, 1, y)

# # 원 데이터 시각화 (2개피쳐에 대하여)
# plt.scatter(X[:,0], X[:,1], color=np.array(['r','g','b'])[y] )
# plt.xlabel('X[0] feature 0')
# plt.ylabel('X[1] feature 1')
# plt.show()
# print('X shape:',X.shape)



# # 트레인 셋, 테스트 셋 분할, 개수출력
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, random_state=RND_SEED)
# print('Train data: {}, Test data: {} \n'.format(X_train.shape[0], X_test.shape[0]))

sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=10000)

# # Cross Validation
# scores = cross_validate(sgd, X, y, cv=5, scoring='r2', return_train_score=True)
# print(scores['train_score'])
# print(scores)


from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
# enumerate the splits and summarize the distributions
for train_ix, test_ix in kfold.split(X, y):
	# select rows
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# summarize train and test composition
	train_0, train_1, train_2 = len(train_y[train_y==0]), len(train_y[train_y==1]), len(train_y[train_y==2])
	test_0, test_1, test_2 = len(test_y[test_y==0]), len(test_y[test_y==1]), len(test_y[test_y==2])
	print('>Train: 0=%d, 1=%d, 2=%d, Test: 0=%d, 1=%d, 2=%d' % (train_0, train_1, train_2, test_0, test_1, test_2))
    
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_clusters_per_class=1, weights=[0.7, 0.2, 0.1], flip_y=0.01, random_state=1)
# split into train/test sets with same class ratio
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# summarize
train_0, train_1, train_2 = len(trainy[trainy==0]), len(trainy[trainy==1]), len(train_y[train_y==2])
test_0, test_1, test_2  = len(testy[testy==0]), len(testy[testy==1]), len(test_y[test_y==2])
print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
plt.scatter(X[:,0], X[:,1], color=np.array(['r','g','b'])[y] )
plt.xlabel('X[0] feature 0')
plt.ylabel('X[1] feature 1')
plt.show()
print('X shape:',X.shape)


from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
#pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=2)),('clf', LogisticRegression(random_state=1))])
pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=2)),('clf', sgd)])
scores = cross_val_score(estimator=pipe_lr, X=X, y=y, cv=10, n_jobs=-1)
print('\nCV accuracy scores: %s' %scores)
print('CV accuracy: %.4f +/- %.4f' % (np.mean(scores), np.std(scores)))
print()
print('SVM-SGD Train Acc: %.4f Test Acc: %.4f'%(sgd.score(X_train_std,y_train),sgd.score(X_test_std,y_test)))


# # 피처 스케일링
# sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.fit_transform(X_train)
# X_test_std = sc.transform(X_test)


# # 피쳐 선택, SBS를 구현하여 사용(Sequential Backward Selection)
# K_FEATURES = 1 # 반환할 피처 개수를 정하기 위한 파라미터
# # 부분집합으로 이해. 1개피처를 반환할때는 20개의 case, 20개의 피처는 1개의 case
# import matplotlib.pyplot as plt
# sbs = SBS(sgd, k_features=K_FEATURES)
# sbs.fit(X_train_std, y_train) # 내부에서 train, test split을 한번 더 진행한다. test set이 섞이지 않도록 함
# # 정확도 시각화
# k_feat = [len(k) for k in sbs.subsets_]
# plt.plot(k_feat, sbs.scores_, marker='o')
# plt.ylim([0.2, 1.1])
# #plt.xlim([0, 20])
# plt.ylabel('Accuracy')
# plt.xlabel('Number of features')
# plt.grid()
# plt.show()


# print('max Acc Features num:',X.shape[1] - sbs.scores_.index(max(sbs.scores_)))
# print('selectied Features:',sbs.subsets_[sbs.scores_.index(max(sbs.scores_))])
# # 피처가 16개일때 가장 우수한 성능이라면
# # 부분집합을 출력하려면 (전체피처수 - 16)
# # print(sbs.subsets_[20-16])
# # pd 데이터프레임에선 인덱스 위치로 피처 이름을 알 수 있다.

# sgd.fit(X_train_std, y_train) # 기본 분류기 성능을 알아본다
# # tr_acc = accuracy_score(y_train, sgd.predict(X_train_std))
# # te_acc = accuracy_score(y_test, sgd.predict(X_test_std))
# # print('SVM-SGD Train Acc: %.4f Test Acc: %.4f'%(tr_acc,te_acc))
# # 분류기.score(문제X,정답y) == accuracy_score(정답y, 분류기.predict(문제X))
# print('SVM-SGD Train Acc: %.4f Test Acc: %.4f'%(sgd.score(X_train_std,y_train),sgd.score(X_test_std,y_test)))

# FS = list(sbs.subsets_[sbs.scores_.index(max(sbs.scores_))]) #FS:Features Selected
# # 선택된 피처로 분류기를 학습해본다
# FSsgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=10000)
# FSsgd.fit(X_train_std[:,FS], y_train)
# print(' FS-SGD Train Acc: %.4f Test Acc: %.4f'%(FSsgd.score(X_train_std[:,FS],y_train),FSsgd.score(X_test_std[:,FS],y_test)))


# # 그리드 서치로 하이퍼 파라미터를 튜닝 # 문제있는듯 너무 오래걸림
# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC
# pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
# param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
# param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']}, {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
# gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
# gs = gs.fit(X_train, y_train)
# print(gs.best_score_)
# print(gs.best_params_)
# gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
# scores = cross_val_score(gs, X, y, scoring='accuracy', cv=5)
# print('CV accuracy: %.4f +/- %.4f' %(np.mean(scores), np.std(scores)))

# # 그리드 서치로 파라미터 튜닝하기
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# pipe_svc = make_pipeline(StandardScaler(),
#                          SVC(random_state=1))
# param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# param_grid = [{'svc__C': param_range, 
#                'svc__kernel': ['linear']},
#               {'svc__C': param_range, 
#                'svc__gamma': param_range, 
#                'svc__kernel': ['rbf']}]
# gs = GridSearchCV(estimator=pipe_svc, 
#                   param_grid=param_grid, 
#                   scoring='accuracy', 
#                   cv=10,
#                   n_jobs=-1)
# gs = gs.fit(X_train, y_train)
# print(gs.best_score_)
# print(gs.best_params_)


# # 사이킷 런의 K 폴드
# from sklearn.model_selection import StratifiedKFold
# skf = StratifiedKFold(n_splits=5)
# # for train_index, test_index in skf.split(X, y): #split(X_train, y_train) 
# #     print("TRAIN:", train_index, "TEST:", test_index)

# # k겹 교차검증을 이용한 모델 평가
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import cross_val_score
# #pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=2)),('clf', LogisticRegression(random_state=1))])
# pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=2)),('clf', sgd)])
# scores = cross_val_score(estimator=pipe_lr, X=X_train_std, y=y_train, cv=10, n_jobs=-1)
# print('\nCV accuracy scores: %s' %scores)
# print('CV accuracy: %.4f +/- %.4f' % (np.mean(scores), np.std(scores)))
# print()
# pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=2)),('clf', sgd)])
# scores = cross_val_score(estimator=pipe_lr, X=X_test_std, y=y_test, cv=10, n_jobs=-1)
# print('CV accuracy scores: %s' %scores)
# print('CV accuracy: %.4f +/- %.4f' % (np.mean(scores), np.std(scores)))
# print()
# pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=2)),('clf', FSsgd)])
# scores = cross_val_score(estimator=pipe_lr, X=X_train_std, y=y_train, cv=10, n_jobs=-1)
# print('CV accuracy scores: %s' %scores)
# print('CV accuracy: %.4f +/- %.4f' % (np.mean(scores), np.std(scores)))
# print()
# pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=2)),('clf', FSsgd)])
# scores = cross_val_score(estimator=pipe_lr, X=X_test_std, y=y_test, cv=10, n_jobs=-1)
# print('CV accuracy scores: %s' %scores)
# print('CV accuracy: %.4f +/- %.4f' % (np.mean(scores), np.std(scores)))



# # 피처 설렉션, 낮은 분산 제거
# from sklearn.feature_selection import VarianceThreshold
# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# sel.fit_transform(X)

# # L1 정규화로 모델 복잡도를 줄이기ㅡ 희소성 증가...
# from sklearn.linear_model import LogisticRegression
# LogisticRegression(penalty='l1')...

# from sklearn.feature_selection import SelectFromModel
# # L1 based 피처 설렉션, ...
# from sklearn.svm import LinearSVC
# lsvc = LinearSVC(C=1.0, penalty="l1", dual=False).fit(X, y)
# model = SelectFromModel(lsvc, prefit=True)
# Xn = model.transform(X)
# print('L1 F.S Xn shape:',Xn.shape)

# # Tree based 피처 설렉션
# from sklearn.ensemble import ExtraTreesClassifier
# clf = ExtraTreesClassifier(n_estimators=50)
# clf = clf.fit(X, y)
# clf.feature_importances_ 
# model = SelectFromModel(clf, prefit=True)
# Xn = model.transform(X)
# print('T.B F.S Xn shape:',Xn.shape)

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
