#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:18:08 2023

@author: sshi
"""
import pandas as pd
def categorize(df):
    # note that all the soundings stop at 700hPa
    cat0 = df[(df.posi_area1>0) & df.freezing_level1.isna()] # all warm, all rain
    
    cat1 = df[((df.posi_area1==0) & (df.nega_area1==0)) | 
              ((df.posi_area1==0) & (df.nega_area1<0))] # all cold or all warm, or sounding touches T=0 but does not cross
    
    cat2 = df[(df.posi_area1>0) & (df.nega_area1==0) & 
              ~df.freezing_level1.isna()] # W has a warm layer at the surface
    cat3 = df[(df.posi_area1>0) & (df.nega_area1<0) & 
              (df.posi_area2.isna()) & (df.nega_area2.isna())] # CW or WC  one cold layer at surface, one warm in the middle layer
    cat4 = df[(df.posi_area1>0) & (df.nega_area1<0) & 
              (df.posi_area2>0) & (df.nega_area2.isna()) & 
              (df.posi_area3.isna())] # WCW
    cat5 = df[(df.posi_area1>0) & (df.nega_area1<0) & 
              (df.posi_area2>0) & (df.nega_area2<0) & 
              (df.posi_area3.isna()) & (df.nega_area3.isna())] # CWCW
    cat6 = df[(df.posi_area1>0) & (df.nega_area1<0) & 
              (df.posi_area2>0) & (df.nega_area2<0) & 
              (df.posi_area3>0) & (df.nega_area3.isna())] # WCWCW
    cat7 = df[(df.posi_area1>0) & (df.nega_area1<0) & 
              (df.posi_area2>0) & (df.nega_area2<0) & 
              (df.posi_area3>0) & (df.nega_area3<0)]# CWCWCW
    
    cat32 = cat3[~cat3.freezing_level2.isna()]
    cat2 = pd.concat([cat2,cat32])
    cat3.drop(cat32.index, inplace=True)
    return cat0, cat1, cat2, cat3, cat4, cat5, cat6, cat7

def rename_df_columns(tstr):
    '''rename columns of df
    if tstr is 't', then rename those with subscripts _t
    if tstr is 'tw', then remove the subscripts _tw'''
    if tstr=='t':
        columns_mapper={'posi_area1_t':'posi_area1', 
                        'posi_area2_t':'posi_area2', 
                        'posi_area3_t':'posi_area3', 
                        'nega_area1_t':'nega_area1', 
                        'nega_area2_t':'nega_area2', 
                        'nega_area3_t':'nega_area3',
                        'freezing_level1_t':'freezing_level1', 
                        'freezing_level2_t':'freezing_level2', 
                        'freezing_level3_t':'freezing_level3'}
    elif tstr=='tw':
        columns_mapper={'posi_area1_tw':'posi_area1', 
                    'posi_area2_tw':'posi_area2', 
                    'posi_area3_tw':'posi_area3', 
                    'nega_area1_tw':'nega_area1', 
                    'nega_area2_tw':'nega_area2', 
                    'nega_area3_tw':'nega_area3',
                    'freezing_level1_tw':'freezing_level1', 
                    'freezing_level2_tw':'freezing_level2', 
                    'freezing_level3_tw':'freezing_level3'}
    return columns_mapper