#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:10:02 2021
versions:
    v2: label spectra
    v5: raman cross-check
    v6: ROC curve using CV
@author: kaandorp
"""

import os
# from spectral import imshow, save_rgb
# import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt

# from pathlib import Path
from scipy import signal

# from scipy.ndimage.filters import gaussian_filter
# from skimage.transform import rescale
import pandas as pd
# import re
# from argparse import ArgumentParser
import glob
import matplotlib.image as mpimg

from classify import calculate_and_label
import configuration as cmp


spectrum_folder = os.path.join(cmp.home_folder,cmp.spectra_folder)
# spectrum_folder = '/Volumes/externe_SSD/kaandorp/Data/SO279/00_spectra/'
spectrum_files = glob.glob(os.path.join(spectrum_folder,'post*.spectrum')) 
out_folder = os.path.join(cmp.home_folder,cmp.label_folder)

# out_folder = '/Volumes/externe_SSD/kaandorp/Data/SO279/00_labels/'

# cmap = plt.cm.tab10


    
#%%    
    
# peak_presences = calculate_and_label(spectrum_files[0], out_folder, mode='calculate',PLOT=False) #mode: label/calculate/predict   
# peak_presences = calculate_and_label(spectrum_files[15], out_folder, mode='label',PLOT=True) #mode: label/calculate/predict   
  

#%% create training dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from scipy import interp

files_labels = glob.glob('/Volumes/externe_SSD/kaandorp/Data/SO279/00_labels/*')


for i1,file_ in enumerate(files_labels[:5]):
    
    if i1 == 0:
        df_label = pd.read_csv(file_,header=2,index_col=0)
    else:
        df_label_new = pd.read_csv(file_,header=2,index_col=0)
        df_label = pd.concat((df_label,df_label_new))
        
#reference raman data PE/PP
for i1,file_ in enumerate(files_labels[5:]):
    
    if i1 == 0:
        df_raman = pd.read_csv(file_,header=2,index_col=0)
    else:
        df_raman_new = pd.read_csv(file_,header=2,index_col=0)
        df_raman = pd.concat((df_raman,df_raman_new))


# i_df_raman_PE = np.where(df_raman.iloc[:,15] == 'PE')[0]
# i_df_raman_PP = np.where(df_raman.iloc[:,15] == 'PP')[0]
# i_df_raman_other = np.where(df_raman.iloc[:,15] == 'other')[0]
# i_raman_use = np.append(i_df_raman_PE,i_df_raman_PP)

# percentage_false_arr = []
df_total = pd.concat((df_label,df_raman))
labels_use = (df_total.iloc[:,15] == 'PE') | (df_total.iloc[:,15] == 'PP') | (df_total.iloc[:,15] == 'PS') | \
    (df_total.iloc[:,15] == 'organic') | (df_total.iloc[:,15] == 'other') 

df_use = df_total[labels_use]
df_use = df_use.replace('organic','other')

n_splits = 10
kf = KFold(n_splits=n_splits,shuffle=True)

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
    # print(i_test)
    # print (i_train.shape,i_test.shape)

    # i_train = i_raman_use[ii_train]
    # i_test = i_raman_use[ii_test]

    # df_label_ = pd.concat((df_label,df_raman.iloc[i_train,:]))
    # labels_use = (df_label_.iloc[:,15] == 'PE') | (df_label_.iloc[:,15] == 'PP') | (df_label_.iloc[:,15] == 'PS') | \
    #     (df_label_.iloc[:,15] == 'organic') | (df_label_.iloc[:,15] == 'other') 
    # df_train = df_label_[labels_use]

    # df_test = df_raman.iloc[i_test,:]
    X_train = df_use.iloc[i_train,:-1].values
    X_test = df_use.iloc[i_test,:-1].values
    Y_train = df_use.iloc[i_train,-1].values
    Y_test = df_use.iloc[i_test,-1].values

    # classifier = GaussianProcessClassifier()
    classifier = RandomForestClassifier(max_features=.33)
    classifier.fit(X_train,Y_train)

    probas_ = classifier.predict_proba(X_test)
    Y_pred = classifier.predict(X_test)
    
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

#%%
# # plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i2, roc_auc))

# for i2 in range(20):
    
#     #reference materials
    
#     i_raman_train_PE,i_raman_test_PE = train_test_split(i_df_raman_PE,test_size=.25)
#     i_raman_train_PP,i_raman_test_PP = train_test_split(i_df_raman_PP,test_size=.25)
    
#     df_label_ = pd.concat((df_label,df_raman.iloc[i_raman_train_PE,:]))
#     df_label_ = pd.concat((df_label_,df_raman.iloc[i_raman_train_PP,:]))
    
#     labels_use = (df_label_.iloc[:,15] == 'PE') | (df_label_.iloc[:,15] == 'PP') | (df_label_.iloc[:,15] == 'PS') | \
#         (df_label_.iloc[:,15] == 'organic') | (df_label_.iloc[:,15] == 'other') 
#     df_train = df_label_[labels_use]
    
#     df_train = df_train.replace('organic','other')
    
    
#     df_test = df_raman.iloc[np.append(i_raman_test_PE,i_raman_test_PP)]
    
#     X_train = df_train.iloc[:,:-1].values
#     X_test = df_test.iloc[:,:-1].values
    
#     Y_train = df_train.iloc[:,-1].values
#     Y_test = df_test.iloc[:,-1].values
    
#     # classifier = DecisionTreeClassifier()
#     classifier = RandomForestClassifier(max_features=.33)
#     # classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3))
#     # classifier = GaussianProcessClassifier()
    
#     classifier.fit(X_train,Y_train)
    
#     predictions = classifier.predict(X_test)
    
#     true_prediction = (predictions==Y_test)
#     print((~true_prediction).sum(),len(true_prediction),(~true_prediction).sum()/len(true_prediction))
#     # print(true_prediction)

#     percentage_false_arr.append((~true_prediction).sum()/len(true_prediction))

# print(np.mean(np.array(percentage_false_arr)) )


#%% Train the classifier on all data

df_total = pd.concat((df_label,df_raman))
labels_use = (df_total.iloc[:,15] == 'PE') | (df_total.iloc[:,15] == 'PP') | (df_total.iloc[:,15] == 'PS') | \
        (df_total.iloc[:,15] == 'organic') | (df_total.iloc[:,15] == 'other')  
df_total_ = df_total[labels_use]
df_total_ = df_total_.replace('organic','other')

X_total = df_total_.iloc[:,:-1].values  
Y_total = df_total_.iloc[:,-1].values

classifier = 'RFC'
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
    from joblib import dump, load
    from datetime import datetime
    
    now_ = datetime.now()
    filename_model = '01_ML_models/%s_%4.4i%2.2i%2.2i%2.2i.joblib' % (classifier,now_.year,now_.month,now_.day,now_.hour)
    dump(classifier_total, filename_model) 
if LOAD:
    classifier_total = load('01_ML_models/RFC_2021052616.joblib') 
    
#%%
PLOT=False
plot_folder = '/Volumes/externe_SSD/kaandorp/Data/SO279/00_figures/'

for i1, spectrum_file_ in enumerate(spectrum_files):
    # print(spectrum_file_)
    if i1 == 0:
        peaks_predict, df_summary = calculate_and_label(spectrum_file_, mode='predict', classifier=classifier_total, PLOT=PLOT, plot_folder=plot_folder)
    else:
        peaks_predict, df_summary_ = calculate_and_label(spectrum_file_, mode='predict', classifier=classifier_total, PLOT=PLOT, plot_folder=plot_folder)        
        df_summary = pd.concat((df_summary,df_summary_))

        
        
cols = list(df_summary.columns.values)

df_save = df_summary[[ 'label',
                      'particle ID',
                    'L',
                    'l',
                    'area',
                    'filled_area',
                    'major_axis',
                    'minor_axis',
                    'equivalent_diameter',
                    'perimeter',
                    'material_peak',
                    'ML_guess',
                    'ML_proba_PE',
                    'ML_proba_PP',
                    'ML_proba_PS',
                    'ML_proba_other',
                    'raman_label']]
df_save = df_save.rename(columns={ 'material_peak':'guess_peakfinding_algorithm'})
# df_save.to_excel('HSI_summary.xlsx')
        
#%%
PE_thres = 0.785
PP_thres = 0.320
other_thres = 0.243

i_PE = df_summary.loc[:,'ML_proba_PE'] > PE_thres
i_PP = df_summary.loc[:,'ML_proba_PP'] > PP_thres
i_PS = df_summary.loc[:,'ML_guess'] == 'PS'
i_other = df_summary.loc[:,'ML_proba_other'] > other_thres

per_PE = (i_PE).sum() / len(df_summary)
per_PP = (i_PP).sum() / len(df_summary)
per_PS = (i_PS).sum() / len(df_summary)
per_other = (i_other).sum() / len(df_summary)

percentages = [per_PE,per_PP,per_other,per_PS]
labels = ['PE','PP','other','PS']
explode = (0,0.1,0.1,0.1)

fig,ax = plt.subplots(1,2,figsize=(10,4.5))
ax[0].pie(percentages, explode=explode, labels=labels, autopct='%1.1f%%',
        pctdistance=0.85, labeldistance=1.2, shadow=False, startangle=90)
ax[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
       
bins_l = np.logspace(np.log10(0.3),np.log10(40),10)
bins_space = bins_l[1:] - bins_l[:-1]
bins_midpoints = 10**(.5*(np.log10(bins_l[1:]) + np.log10(bins_l[:-1])))

hist_PE = np.histogram(df_summary.loc[i_PE,'L'],bins=bins_l)[0] / bins_space
hist_PP = np.histogram(df_summary.loc[i_PP,'L'],bins=bins_l)[0] / bins_space
hist_plastic = np.histogram(df_summary.loc[(i_PE | i_PP),'L'] ,bins=bins_l)[0] / bins_space    
hist_other = np.histogram(df_summary.loc[i_other,'L'],bins=bins_l)[0] / bins_space  

# fig,ax = plt.subplots(1)
ax[1].loglog(bins_midpoints,hist_PE/hist_PE.sum(),'o-',label='PE')
ax[1].loglog(bins_midpoints[hist_PP>0],(hist_PP/hist_PP.sum())[hist_PP>0],'o-',label='PP')
ax[1].loglog(bins_midpoints[hist_other>0],(hist_other/hist_other.sum())[hist_other>0],'o-',label='other')

# ax.loglog(bins_midpoints,hist_plastic/hist_plastic.sum(),'o-')
ax[1].legend()
ax[1].set_xlabel('Particle length [mm]',fontsize=14)
ax[1].set_ylabel(r'Normalized abundance [n mm$^{-1}$]',fontsize=14)
fig.tight_layout()

# possible PS particles
df_summary.loc[df_summary.loc[:,'ML_proba_PS'] > 0.3,['label','particle ID'] ]

#%% calculate some slopes
from scipy.optimize import minimize_scalar
from scipy.integrate import simps
from scipy.optimize import curve_fit

def data_cdf(h,i_min):
    '''
    cumulative distribution function of empirical data
    '''
    return np.cumsum(h[i_min:]) / h[i_min:].sum()

def log_lik(alpha,bins,h,i_min=0):
    '''
    log-likelihood function defined in Virkar & Clauset (2014)
    '''
    n = np.sum(h[i_min:])
    L = n*(alpha-1)*np.log(bins[i_min]) + np.sum(h[i_min:]*np.log(bins[i_min:-1]**(1-alpha) - bins[i_min+1:]**(1-alpha) ) )
    return L

def J_alpha(alpha,bins,h,i_min=0):
    '''
    cost function to be minimized to find the powerlaw slope based its likelihood
    '''
    return -log_lik(alpha,bins,h,i_min)    
    
def powerlaw_d_pdf(alpha,b,i_min):
    b_min = b[i_min]
    C = ((alpha-1)*b_min**(alpha-1))
    return (C / (alpha-1)) * (b[:-1]**(1-alpha) - b[1:]**(1-alpha))

def powerlaw_d_cdf(alpha,b,i_min):
    '''
    cumulative distribution function of the powerlaw pdf
    '''
    return powerlaw_d_pdf(alpha,b,i_min)[i_min:].cumsum()

def normal_pdf(x,mu,sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-.5*((x-mu)/sigma)**2)

def calculate_powerlaw_parameters(b,h,lowerbound,plot=False):
    '''
    Calculate powerlaw slope based on the paper by Virkar and Clauset (2014)
    b: particle size
    h: amount of particles in bin
    lowerbound: if true, the Kolmogorov Smirnov statistic is minimized to find the lowerbound. 
                If int, the lowerbound is directly specified
    '''
    if type(lowerbound)==bool and lowerbound==True:
        KS = np.array([])
        alphas = np.array([])
        i_mins = np.arange(len(b)-3)
        for i_min_ in i_mins:
            res = minimize_scalar(J_alpha,  bounds=[1.1,5], args=(b,h,i_min_),  method='bounded')
            alpha_ = res.x
            alphas = np.append(alphas,alpha_)
            
            cdf_model = powerlaw_d_cdf(alpha_,b,i_min_)
            cdf_emp = data_cdf(h,i_min_)
            KS_ = np.max(np.abs(cdf_emp - cdf_model))
            
            KS = np.append(KS,KS_)
        
        if plot:
            plt.figure()
            plt.plot(i_mins,KS,'o-')    
        
        i_min_opt = i_mins[np.argmin(KS)]
        alpha_opt = alphas[np.argmin(KS)]
        KS_opt = np.min(KS)
    elif type(lowerbound)==bool and lowerbound==False:
        res = minimize_scalar(J_alpha,  bounds=[1.1,5], args=(b,h,0),  method='bounded')
        i_min_opt = 0
        alpha_opt = res.x
        
        cdf_model = powerlaw_d_cdf(alpha_opt,b,i_min_opt)
        cdf_emp = data_cdf(h,i_min_opt)
        KS_opt = np.max(np.abs(cdf_emp - cdf_model))    
    elif type(lowerbound)==int:
        res = minimize_scalar(J_alpha,  bounds=[1.1,5], args=(b,h,lowerbound),  method='bounded')
        i_min_opt = lowerbound
        alpha_opt = res.x
        
        cdf_model = powerlaw_d_cdf(alpha_opt,b,i_min_opt)
        cdf_emp = data_cdf(h,i_min_opt)
        KS_opt = np.max(np.abs(cdf_emp - cdf_model))          
    else:
        raise RuntimeError('not defined')

    return alpha_opt,i_min_opt,KS_opt


def calculate_alpha_sigma(alpha_opt,b,h,i_min,plot=False):
    '''
    Fit a normal distribution through the normalized likelihood curve to estimate the powerlaw slope and its uncertainty
    The fitted alpha should be almost the same as the one obtained in the calculate_powerlaw_parameters function
    Result should be plotted to ensure that the integration domain is wide enough such that the area~1
    '''
    alpha_integrate = np.linspace(max(alpha_opt-1,1.01),alpha_opt+1,10000)
    arr_loglikelihood = np.array([log_lik(alpha_,b,h,i_min=i_min) for alpha_ in alpha_integrate])
    arr_likelihood = np.exp(arr_loglikelihood - arr_loglikelihood.max())
    
    if not (arr_likelihood[0] < 1e-10 and arr_likelihood[-1] < 1e10):
        print('----------------warning--------------')
        print(arr_likelihood[0],arr_likelihood[0])
    
    I = simps(arr_likelihood,alpha_integrate)
    posterior = arr_likelihood / I
        
    fit = curve_fit(normal_pdf,alpha_integrate,posterior,p0=[alpha_opt,.05])
    
    if plot:
        plt.figure()
        plt.plot(alpha_integrate,posterior)
        plt.plot(alpha_integrate,normal_pdf(alpha_integrate,fit[0][0],fit[0][1]),'--')
    
    print('alpha: %3.2f  sigma: %3.2f' %(fit[0][0],fit[0][1]))
    print('size range: from %3.3f mm, from index %i' % (b[i_min],i_min))
    
    alpha_fit = fit[0][0]
    sigma_fit = fit[0][1]

    return alpha_fit,sigma_fit

hist_nonnorm_PE = np.histogram(df_summary.loc[i_PE,'L'],bins=bins_l)[0]
alpha_opt,i_min_opt,KS_opt = calculate_powerlaw_parameters(bins_l,hist_nonnorm_PE,4,plot=True)
alpha_fit,sigma_fit = calculate_alpha_sigma(alpha_opt,bins_l,hist_nonnorm_PE,i_min_opt)
print('slope PE: %f, +- %f' %(alpha_fit,sigma_fit))

hist_nonnorm_PP = np.histogram(df_summary.loc[i_PP,'L'],bins=bins_l)[0]
alpha_opt,i_min_opt,KS_opt = calculate_powerlaw_parameters(bins_l,hist_nonnorm_PP,4,plot=True)
alpha_fit,sigma_fit = calculate_alpha_sigma(alpha_opt,bins_l,hist_nonnorm_PP,i_min_opt)
print('slope PP: %f, +- %f' %(alpha_fit,sigma_fit))

