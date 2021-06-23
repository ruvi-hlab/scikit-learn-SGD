#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 13:28:06 2021

@author: user0220
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import pandas as pd
import numpy as np
print('전체 데이터셋 로드\n')
df = pd.read_excel('/home/user0220/Downloads/발달코드/시즌2 데이터/최종/분류기/data/전체_데이터_20200223.xlsx' ,sheet_name='Sheet1')

## df  : 원본 엑셀 데이터
## df2 : 정답, ID, 결측행이 제거된 데이터
RND_SEED = 42
CPU_CORES = -1
K_FOLD = 10
# ########################
# ## 기존 생성 데이터를 업샘플링 되는지 테스트
# ########################

# N_SAMPLES = 350
# N_FEATURES = 210  ##독립변수=피처
# N_CLUSTERS_OF_A_CLASS = 1
# N_CLASSES = 3  ##종속변수=클래스
# CLUSTERS_SPLIT = [0.35, 0.35, 0.3]
# RANDOM_NOISE = 0 # 0.5%의 오차 노이즈
# N_INF = 15  ##독립 변수 중 종속 변수와 상관 관계가 있는 성분의 수, 디폴트 2
# N_RED = 15  ##독립 변수 중 다른 독립 변수의 선형 조합으로 나타나는 성분의 수, 디폴트 2
# N_REP = 10  ##독립 변수 중 단순 중복된 성분의 수, 디폴트 0
# from sklearn.datasets import make_classification
# X, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES, 
#        n_informative=N_INF, n_redundant=N_RED, n_repeated=N_REP, n_classes=N_CLASSES, 
#        n_clusters_per_class=N_CLUSTERS_OF_A_CLASS, weights=CLUSTERS_SPLIT, 
#        flip_y=RANDOM_NOISE, random_state=RND_SEED)


##########
## 데이터프레임 조작 테스트
##########
print('정답 레이블(y) 분리\n전문의 판정feature를 정답으로 사용함\n')
y = df['전문의판정']

print('결측 행 제거\n')
df2 = df.dropna(axis=1)

print('정답 제거된 데이터프레임 생성\n')
df2.drop(['전문의판정','ADHD=2\nADHD 위기=1\n정상=0'], axis=1, inplace=True)

print('이름, ID 제거\n')
df2.drop(['ID','이름'], axis=1, inplace=True)

print('음성:{}\n위기:{}\n양성:{}\n'.format(list(y).count(0),list(y).count(1),list(y).count(2)))

print('세트 분할')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df2, y, test_size=0.2, 
                         shuffle=True, stratify=y, random_state=RND_SEED)
print('Train data: {}, Test data: {}\n'.format(X_train.shape[0], X_test.shape[0]))

###########
## 순서문제
###########
print('데이터 스케일링\n')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

print('불균형 데이터 샘플링(SMOTE)')
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=RND_SEED)
X_train_std_over, y_train_over = smote.fit_sample(X_train_std, y_train)
print('음성:{}\n위기:{}\n양성:{}\n'.format(list(y_train_over).count(0),list(y_train_over).count(1),list(y_train_over).count(2)))



from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
SGD_PENALTY = "l2" ## l1
SGD_LOOP = 10000
sgd = SGDClassifier(penalty=SGD_PENALTY, max_iter=SGD_LOOP)
param_grid = {'loss':['hinge','log'],
              'alpha':[0.0001, 0.001, 0.01]}
gs = GridSearchCV(estimator=sgd, param_grid=param_grid, scoring='f1_micro',
                  cv=K_FOLD, n_jobs=CPU_CORES)
gs.fit(X_train_std_over,y_train_over)
# print(gs.best_score_)
print('Grid search:',gs.best_params_,'\n')

GSsgd = SGDClassifier(penalty=SGD_PENALTY, loss=gs.best_params_['loss'], 
                      alpha=gs.best_params_['alpha'] , max_iter=SGD_LOOP)
GSsgd.fit(X_train_std_over, y_train_over)
print('GSsgd. Train: %.4f  Test: %.4f'%(
    f1_score(y_train_over, GSsgd.predict(X_train_std_over),average='micro'), 
    f1_score(y_test, GSsgd.predict(X_test_std), average='micro')))
CM_GSsgd = confusion_matrix(y_test, GSsgd.predict(X_test_std))  ##위치바꾸면 왜 피처수 에러생기는거지??

###########
## ANOUNCE
###########
print('기존 분류기 특징\n  데이터 특성에 따른 2가지 옵션\n  로봇데이터, 전문가데이터로 나뉘어 있음\n  본 과정에서는 이 2가지 데이터 타입을 통채로 분류모델에 적용해 본다')

# SMOTE 먼저
# GSsgd. Train: 0.9695  Test: 0.6923
# 스케일링 먼저
# GSsgd. Train: 0.9756  Test: 0.6308
