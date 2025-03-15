#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:17:14 2023

@author: sshi
"""
import os
import numpy as np
import pandas as pd

def read_areas(path):
    filenames = os.listdir(path)
    # read all soundings and see...
    df = pd.read_csv(path + '/' + filenames[0])
    df['ID'] = filenames[0].split('_')[0]
    for i in range(1, len(filenames)):
        df1 = pd.read_csv(path + '/' + filenames[i])
        df1['ID'] = filenames[i].split('_')[0]
        df = pd.concat([df, df1], ignore_index=True)
        
    df.loc[df.posi_area2_t==9999, 'posi_area2_t'] = np.nan
    df.loc[df.nega_area2_t==9999, 'nega_area2_t'] = np.nan
    df.loc[df.posi_area2_tw==9999, 'posi_area2_tw'] = np.nan
    df.loc[df.nega_area2_tw==9999, 'nega_area2_tw'] = np.nan    
    
    df['PA_t'] = df.loc[:, 'posi_area1_t':'posi_area3_t'].sum(axis=1)
    df['NA_t'] = -df.loc[:, 'nega_area1_t':'nega_area3_t'].sum(axis=1)
    df['PA_tw'] = df.loc[:, 'posi_area1_tw':'posi_area3_tw'].sum(axis=1)
    df['NA_tw'] = -df.loc[:, 'nega_area1_tw':'nega_area3_tw'].sum(axis=1)
    
    # df = df[pd.DatetimeIndex(df['datetime']).month.isin([1,2,3, 4, 10,11,12])]
    return df