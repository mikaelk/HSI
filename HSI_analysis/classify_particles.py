#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 15:28:53 2021

@author: kaandorp
"""

import os
import configuration as cmp
from argparse import ArgumentParser
from joblib import load
import glob
from classify import calculate_and_label
import pandas as pd

if __name__=="__main__":
    
    p = ArgumentParser(description="""Particle classification""")
    p.add_argument('-string_classify', '--string_classify', default='NEMICAT',help='search string')
    p.add_argument('-ML_model', '--ML_model', default='RFC_2021052616.joblib',help='saved ML model')

    args = p.parse_args()
    data_name = args.string_classify
    filename_ML = args.ML_model
    
    file_ML = os.path.join(cmp.home_folder,cmp.ML_folder,filename_ML)

    classifier_total = load(file_ML) 

    spectrum_folder = os.path.join(cmp.home_folder,cmp.spectra_folder)
    spectrum_files = glob.glob(spectrum_folder + '/post_*'+data_name+'*.spectrum')
    
    out_folder = os.path.join(cmp.home_folder,cmp.output_folder)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    print('Writing output to %s' % out_folder)


    for i1, spectrum_file_ in enumerate(spectrum_files):
        if i1 == 0:
            peaks_predict, df_summary = calculate_and_label(spectrum_file_, mode='predict', classifier=classifier_total, PLOT=True, plot_folder=out_folder)
        else:
            peaks_predict, df_summary_ = calculate_and_label(spectrum_file_, mode='predict', classifier=classifier_total, PLOT=True, plot_folder=out_folder)        
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
    df_save.to_excel(os.path.join(out_folder,'Summary.xlsx'))