#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:58:20 2021

@author: kaandorp
"""

import os
from classify import calculate_and_label
import configuration as cmp
from argparse import ArgumentParser


if __name__=="__main__":
    
    p = ArgumentParser(description="""Particle labelling""")
    p.add_argument('-name', '--name', default='post_01_NEMICAT_27_1.spectrum',help='folder name')

    args = p.parse_args()
    data_name = args.name
    
    spectrum_folder = os.path.join(cmp.home_folder,cmp.spectra_folder)
    spectrum_file = os.path.join(spectrum_folder,data_name)
    
    out_folder = os.path.join(cmp.home_folder,cmp.label_folder)

    peak_presences = calculate_and_label(spectrum_file, mode='label', PLOT=True) #mode: label/calculate/predict   


    