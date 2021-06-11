#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 15:00:06 2021

@author: kaandorp
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import configuration as cmp


files_labels = glob.glob(os.path.join(cmp.home_folder,cmp.label_folder)+'/*' )
folder_ML_models = os.path.join(cmp.home_folder,cmp.ML_folder)

classifier = 'RFC' #choose from random forest classifier (RFC) or Gaussian proc. class. (GPC)

for i1, file_ in enumerate(files_labels):
    if i1 == 0:
        df_total = pd.read_csv(file_,header=2,index_col=0)
    else:
        df_total_ = pd.read_csv(file_,header=2,index_col=0)
        df_total = pd.concat((df_total,df_total_))
        
    
labels_use = (df_total.iloc[:,15] == 'PE') | (df_total.iloc[:,15] == 'PP') | (df_total.iloc[:,15] == 'PS') | \
    (df_total.iloc[:,15] == 'organic') | (df_total.iloc[:,15] == 'other') 


df_use = df_total[labels_use]
df_use = df_use.replace('organic','other')


#%% Assess performance using k-fold cross validation
n_splits = 10
kf = KFold(n_splits=n_splits,shuffle=True)

if classifier == 'RFC':
    classifier_test = RandomForestClassifier(max_features=.33)
elif classifier == 'GPC':
    classifier_test = GaussianProcessClassifier()
else:
    raise RuntimeError('unknown classifier')

plt.figure(figsize=(5,4))

classes = ['PE','PP','PS','other']
mean_fpr = np.linspace(0, 1, 100)
mean_tpr = {}
mean_thres = {}
all_tpr = {}
array_tpr = {}
array_fpr = {}
array_thres = {}
TPR = {}
FPR = {}
for class_ in classes:
    mean_tpr[class_] = 0.
    mean_thres[class_] = 0.
    all_tpr[class_] = []

    array_tpr[class_] = np.array([])
    array_fpr[class_] = np.array([])
    array_thres[class_] = np.array([])
    TPR[class_] = 0
    FPR[class_] = 0
accuracy = 0

for i2, (i_train, i_test) in enumerate(kf.split(df_use)):

    X_train = df_use.iloc[i_train,:-1].values
    X_test = df_use.iloc[i_test,:-1].values
    Y_train = df_use.iloc[i_train,-1].values
    Y_test = df_use.iloc[i_test,-1].values

    classifier_test.fit(X_train,Y_train)

    probas_ = classifier_test.predict_proba(X_test)
    Y_pred = classifier_test.predict(X_test)
    
    acc_ = (Y_pred == Y_test).sum() / len(Y_pred)
    accuracy += (acc_) / n_splits
    print(acc_)
    
    for i3,class_ in enumerate(classes):

        TP = (Y_test == class_) & (Y_pred == class_)
        FP = (Y_test != class_) & (Y_pred == class_)
        TN = (Y_test != class_) & (Y_pred != class_)
        FN = (Y_test == class_) & (Y_pred != class_)        
        
        TPR[class_] += (TP.sum() / (TP.sum()+FN.sum())) / n_splits
        FPR[class_] += (FP.sum() / (TN.sum()+FP.sum())) / n_splits

        Y_class_id = (Y_test == class_)
        Y_pred_id = probas_[:,i3]
        
        fpr, tpr, thresholds = roc_curve(Y_class_id, Y_pred_id)
        mean_tpr[class_] += interp(mean_fpr, fpr, tpr) / n_splits
        mean_tpr[class_][0] = 0.0
        mean_thres[class_] += interp(mean_fpr, fpr, thresholds) / n_splits
        
        array_tpr[class_] = np.append(array_tpr[class_],tpr)
        array_fpr[class_] = np.append(array_fpr[class_],fpr)
        array_thres[class_] = np.append(array_thres[class_],thresholds)
        
        assert(len(tpr)==len(fpr))
        assert(len(fpr)==len(thresholds))
        roc_auc = auc(fpr, tpr)
    print(i2)
    
for class_ in classes:    
    plt.plot(mean_fpr,mean_tpr[class_],label=class_)
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='1:1')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

fig,ax = plt.subplots(2,sharex=True)
for class_ in classes:    
    ax[0].plot(array_thres[class_],array_tpr[class_],'o',label=class_)
    ax[1].plot(array_thres[class_],array_fpr[class_],'o',label=class_)

ax[1].set_xlabel('threshold')
ax[0].set_ylabel('True positive')
ax[1].set_ylabel('False positive')
ax[0].set_xlim(0,1)
ax[1].legend()

#%% Train classifier on all data

labels_use = (df_total.iloc[:,15] == 'PE') | (df_total.iloc[:,15] == 'PP') | (df_total.iloc[:,15] == 'PS') | \
        (df_total.iloc[:,15] == 'organic') | (df_total.iloc[:,15] == 'other')  
df_total_ = df_total[labels_use]
df_total_ = df_total_.replace('organic','other')

X_total = df_total_.iloc[:,:-1].values  
Y_total = df_total_.iloc[:,-1].values

if classifier == 'RFC':
    classifier_total = RandomForestClassifier(max_features=.33)
elif classifier == 'GPC':
    classifier_total = GaussianProcessClassifier()
else:
    raise RuntimeError('unknown classifier')
    
classifier_total.fit(X_total,Y_total)

SAVE = True
LOAD = False
if SAVE:
    from joblib import dump
    from datetime import datetime
    
    # folder_ML_models = '00_ML_models'
    now_ = datetime.now()
    if not os.path.exists(folder_ML_models):
        os.mkdir(folder_ML_models)
        
    filename_model = os.path.join(folder_ML_models, '%s_%4.4i%2.2i%2.2i%2.2i.joblib' % (classifier,now_.year,now_.month,now_.day,now_.hour) )
    dump(classifier_total, filename_model) 
