# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 09:55:43 2021

@author: kaandorp
"""

home_folder = '/Users/kaandorp/Git_repositories/HSI/examples_test' 
# home_folder = '/Users/kaandorp/Git_repositories/HSI/HSI_paper_2021' 
spectra_folder = '00_spectra'
label_folder = '00_labels'
ML_folder = '00_ML_models'
output_folder = '00_final_output'

# decision tree parameters which can be used as a first guess
characterization = {
    'PE':{'hard':[1045, 1215, 1396],
          'soft':[]},
    'PP':{'hard':[1196,1701],
          'soft':[]},
    'PA66':{'hard':[1203,1538,1573,1602],
            'soft':[]},
    'PMMA':{'hard':[1175,1680],
            'soft':[]},
    'PET':{'hard':[1130,1663],
           'soft':[1705.91]},
    'PS':{'hard':[1143,1641,1678],
          'soft':[]},
    'PC':{'hard':[1137, 1666],
          'soft':[1189.37]},
    'PVC':{'hard':[1192.87, 1424],
           'soft':[1400]}
}


'''
######################################################
default settings for the peak finding algorithm
######################################################
'''
defaults = {
    'peak_prominence': None,
    'smooth_window': 7 # odd integer
}


'''
######################################################
# settings for the sobol algorithm.
The image can be cut on top/bottom (y_start,y_end)
An estimate of the pixels per mm can be given to remove the smallest (likely invalid) particles, where the minimum
area is given by min_particle_area
A mask can be used for round scans (e.g. petri dishes), where the center pixel and radius can be specified
A intensity thresold can be set for the background (=black, so lower values are masked)
calibration_ID_pix is used to either:
    -specify a string, which is the particle ID used to calibrate the pixel size (e.g. the white bar), where calibration_mm is set to the bar width in mm
    -or, easier: specify a float giving an amount of pixels (when the system is already calibrated, and you know the amount of pixels per mm)
######################################################
'''
settings_sobol = {'y_start':0, #cut the image if scan domain is too large
    'y_end':-1,
    'pixel_to_mm_estim':15/600, #rough estimate
    'use_petri_mask':False,
    'petri_center':[840,320],
    'petri_radius':800,
    'threshold_background':.14,
    'plot':True,
    'min_particle_area':.5,  #mm2
    'calibration_ID_pix':1, #string: particle ID or float: amount of pixels
    'calibration_mm':0.019 #how many mm is the calibration bar or the specified pixels
    }