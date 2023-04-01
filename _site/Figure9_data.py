# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:28:42 2023

@author: ssynj
"""
import pandas as pd
import numpy as np
from function.classification import *

datapath = './data/'
figpath = './figure/'
df = pd.read_csv(datapath + 'final_all_cleaned_data.txt', index_col=0)
df = df.set_index('ID')

# test data: type1 and type2
test = df[  ~df.t.isna() & 
            ~df.td.isna() & 
            ~df.p.isna() & 
            ~df.lapse_rate_tw.isna() &
            ~df.posi_area1_tw.isna() &
            ~df.lapse_rate_t.isna() &
            ~df.posi_area1_t.isna() &
            df.posi_area2_t.isna() & df.nega_area3_t.isna() &
            df.posi_area2_tw.isna() & df.nega_area3_tw.isna()]

IDs = test.index.unique()
test = rename_df_columns(test, 'tw')
scores = pd.DataFrame(index=IDs, columns=['ndata', 'nrain', 'nsnow', 'accuracy', 'recall', 'precision', 'f1score'])
bias_energy = pd.DataFrame(index=IDs, columns=['absolute'])
for ID in IDs:
    test_station = test.loc[[ID]]
    test_one = test_station[test_station.nega_area1.isna()]
    test_two = test_station[~test_station.nega_area1.isna()]
    
    pre_rain_one, pre_snow_one = classify_one_layer(test_one)
    pre_rain_two, pre_snow_two = classify_two_layers(test_two)
    
    pre_rain = pd.concat([pre_rain_one, pre_rain_two])
    pre_snow = pd.concat([pre_snow_one, pre_snow_two])   
    
    scores.loc[ID, 'ndata'] = len(test)
    scores.loc[ID, 'nrain'] = sum(test.wwflag==1)
    scores.loc[ID, 'nsnow'] = sum(test.wwflag==2)
    scores.loc[ID, 'accuracy'], scores.loc[ID, 'recall'], scores.loc[ID, 'precision'], scores.loc[ID, 'f1score'] = metrics(pre_rain, pre_snow)
    
    n_pre_rain = len(pd.concat([pre_rain]))
    n_pre_snow = len(pd.concat([pre_snow]))
    snowp_pre = n_pre_snow/len(test_station)
    snowp_obs = len(test_station[test_station.wwflag==2])/len(test_station)
    bias_energy.loc[ID, 'absolute'] = snowp_pre-snowp_obs 