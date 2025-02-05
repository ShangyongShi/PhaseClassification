#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:36:40 2023

@author: sshi
"""

def categorize(df):
    '''
    determine sounding type based on number of positive and negative
    areas and freezing level
    
    return: type0, type1, type2, others
    '''
    nPA = (~df.loc[:, 'posi_area1':'posi_area3'].isna()).sum(axis=1)
    nNA = (~df.loc[:, 'nega_area1':'nega_area3'].isna()).sum(axis=1)
    nFL = (~df.loc[:, 'freezing_level1':'freezing_level3'].isna()).sum(axis=1)
    
    type0 = df.loc[
                   ((nPA==0) & (nNA==0) & (nFL==0)) |
                   ((nPA==0) & (nNA==1) & (nFL==0)) |
                   ((nPA==1) & (nNA==0) & (nFL==0)) 
                  ]
    type1 = df.loc[
                   ((nPA==1) & (nNA==0) & (nFL==1)) 
                  ]  
    type2 = df.loc[
                   ((nPA==1) & (nNA==1) & (nFL==1)) 
                  ] 
    others= df.loc[
                   ((nPA==1) & (nNA==2) & (nFL==1)) |
                   ((nPA==2) & (nNA==0) & (nFL==1)) |
                   ((nPA==2) & (nNA==1) & (nFL==1)) |
                   ((nPA==2) & (nNA==1) & (nFL==2)) |
                   ((nPA==2) & (nNA==2) & (nFL==2)) 
                  ]
    # bad data: 1 1 0
    return type0, type1, type2, others


def divide_into_training_and_evaluation_set(df):
    df_de = df[(abs(df.lon-df.IGRA_lon)<=0.01) &
               (abs(df.lat-df.IGRA_lat)<=0.01)] 
    df_ev = df.drop(df_de.index) 
    return df_de, df_ev


def rename_columns(df, tstr):
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
    df1 = df.copy()
    df1.rename(columns=columns_mapper, inplace=True)
    return df1