#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:43:01 2021

@author: kaandorp
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import matplotlib.image as mpimg
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
import configuration as cmp


def value_in_range(val,lower,upper):
    i_valid = np.where((val > lower) & (val < upper))[0]
    
    return i_valid.size > 0
    

def calculate_and_label(file, mode='predict', classifier=None, PLOT=True, plot_folder=None):
    
      
    print('Labelling %s' % file)
    
    # the label, extraced from the filename
    label = os.path.basename(file).split('.')[0].split('_',1)[1]
    
    # folder in which the segment figures are present. These help with the labelling process
    figures_folder = os.path.join(cmp.home_folder, label, 'output/')
    
    # spectrum file to be labelled
    folder_spectra_info = os.path.join(cmp.home_folder, cmp.spectra_folder)
    out_folder = os.path.join(cmp.home_folder, cmp.label_folder)
    label_file = os.path.join(out_folder, 'label_'+label)
    file_particle_info = os.path.join(folder_spectra_info, 'post_' + label + '.particle_info')
    
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
        
    df_info_classification = None
    
    data = pd.read_csv(file)
    wavelengths = np.array(list(data.columns)[1:],dtype=float)
    spectra = data.values[:,1:]

    smooth_window = cmp.defaults['smooth_window']
    prominence = cmp.defaults['peak_prominence']

    #location ranges of the throughs to look for:
    feature_ranges_min = np.array([[1538,1542],
                                   [1193,1199],
                                   [1194,1218],
                                   [1213,1218],
                                   [1420,1425],
                                   [1141,1146],
                                   [1205,1212],
                                   [1412,1416],
                                   [1675,1682]])
    
    #location ranges of the peaks to look for:
    feature_ranges_max = np.array([[970,995],
                                   [1275,1310],
                                   [1310,1330],
                                   [1525,1560],
                                   [1515,1525],
                                   [1570,1590]])
    
    
    peak_presences = np.zeros(feature_ranges_min.shape[0]+feature_ranges_max.shape[0])
    
    if mode == 'label':
        header_0 = [feature_ranges_min[i,0] for i in range(len(feature_ranges_min))] + [feature_ranges_max[i,0] for i in range(len(feature_ranges_max))] + ['material_label']
        header_1 = [feature_ranges_min[i,1] for i in range(len(feature_ranges_min))] + [feature_ranges_max[i,1] for i in range(len(feature_ranges_max))] + ['']
        header_2 = ['trough']*len(feature_ranges_min) + ['peak']*len(feature_ranges_max) + ['']
        
        header = [header_0,header_1,header_2]
        
        df = pd.DataFrame(columns = header )
    
    elif mode == 'predict':
        df_info_classification = pd.read_csv(file_particle_info,index_col=0)
        df_info_classification['label'] = label
        df_info_classification['ML_guess'] = 0
        df_info_classification['ML_proba_PE'] = 0
        df_info_classification['ML_proba_PP'] = 0
        df_info_classification['ML_proba_PS'] = 0
        df_info_classification['ML_proba_other'] = 0
        df_info_classification['raman_label'] = ''
        
    c = 0
    for i1 in range(spectra.shape[0]):

        
        if mode == 'label':
            figure_post = figures_folder + 'post_Segment_%i.png'%i1
            if os.path.exists(figure_post):
                pass
            else:
                figure_post = figures_folder + 'Segment_%i.png'%i1
            img = mpimg.imread(figure_post,format='png')
            plt.figure()
            plt.imshow(img)
            plt.show(block=False)
            
        spectrum = spectra[i1,:]    
        
        smoothed = signal.savgol_filter(spectrum, window_length=smooth_window, polyorder=2)
        minimas,_ = signal.find_peaks(-smoothed, prominence=prominence)
        maximas,_ = signal.find_peaks(smoothed, prominence=prominence)

        wl_minimas = wavelengths[minimas]
        wl_maximas = wavelengths[maximas]

        peak_presence = np.zeros(feature_ranges_min.shape[0]+feature_ranges_max.shape[0])
        
        for i2,range_min in enumerate(feature_ranges_min):
            for peak_val in wl_minimas:    
                if value_in_range(peak_val,range_min[0],range_min[1]):
                    peak_presence[i2] = 1
 
        for i3,range_max in enumerate(feature_ranges_max):
            for peak_val in wl_maximas:    
                if value_in_range(peak_val,range_max[0],range_max[1]):
                    peak_presence[i2+i3+1] = 1
        
        if c == 0:
            peak_presences = peak_presence.copy()
        else:
            peak_presences = np.vstack((peak_presences,peak_presence))
        
        
        if mode == 'label':
            # The following lines aid in labelling the particles. Can be removed to avoid biases
            material = []
            if value_in_range(wavelengths[minimas],1538,1542):
                material.append('PE')
            elif value_in_range(wavelengths[minimas],1193,1199) or value_in_range(wavelengths[minimas],1194,1218):
                material.append('PP')
            if len(material) == 0 and value_in_range(wavelengths[minimas],1213,1218) and value_in_range(wavelengths[minimas],1420,1425):
                material.append('PE')
            
        if mode == 'predict':
            if type(classifier) == GaussianProcessClassifier or type(classifier) == RandomForestClassifier:
                material_predict = classifier.predict(peak_presence.reshape(1,-1))
                materials_predict = classifier.predict_proba(peak_presence.reshape(1,-1))
            
                df_info_classification.loc[c,'ML_guess'] = material_predict
                df_info_classification.loc[c,'ML_proba_PE'] = materials_predict[0][0]
                df_info_classification.loc[c,'ML_proba_PP'] = materials_predict[0][1]
                df_info_classification.loc[c,'ML_proba_PS'] = materials_predict[0][2]
                df_info_classification.loc[c,'ML_proba_other'] = materials_predict[0][3]
            else:
                material_predict = classifier.predict(peak_presence.reshape(1,-1))
                df_info_classification.loc[c,'ML_guess'] = material_predict
                
            if os.path.exists(label_file):
                raman_labels = pd.read_csv(label_file,index_col=0,header=2)
                df_info_classification.loc[c,'raman_label'] = raman_labels.loc[c,'Unnamed: 16']
        
        if PLOT:
            plt.figure()
            plt.plot(wavelengths,smoothed)            
            plt.plot(wavelengths[minimas],smoothed[minimas],'x')
            plt.plot(wavelengths[maximas],smoothed[maximas],'o')

            if mode == 'predict':
                if type(classifier) == GaussianProcessClassifier:
                    plt.title('Classifier prediction, most likely: %s\n Probabilities\n%s\n%s' %(str(material_predict[0]),str(classifier.classes_),str(materials_predict)))                
                else:        
                    plt.title('Classifier prediction particle %i: %s' % (c,material_predict))
            elif mode == 'label':
                plt.title(str(material)+'?')
                    
            if plot_folder:
                plt.tight_layout()
                if not os.path.exists(os.path.join(plot_folder,label)):
                    os.mkdir(os.path.join(plot_folder,label))
                plot_filename = os.path.join(plot_folder,label,str(c)+'.png')
                plt.savefig(plot_filename)
            plt.show(block=False)
            
        if mode == 'label':
            material_input = input('material? (Segment %i) \n' % c)
            table_input = np.append(peak_presence,material_input)
            
            df.loc[c] = table_input
        
        c+=1

    if mode == 'label':
        df.to_csv(label_file)    
        return 0
    else:
    
        return peak_presences, df_info_classification
    
    