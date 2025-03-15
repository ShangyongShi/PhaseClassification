#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:05:41 2023

@author: sshi
"""
import pandas as pd
import numpy as np
def read_NCEP_STNID_CALL():
    ncep_stnid_call = pd.read_csv('/r1/sshi/sounding_phase/data/NCEP_STNID_CALL.txt',
                                  header=None, names=['stnid', 'call'], dtype=str)
    ncep_stnid_call.set_index('stnid', inplace=True)
    ncep_stnid_call = ncep_stnid_call[~ncep_stnid_call.call.isna()]
    return ncep_stnid_call

def omit_callid(stations_ncep_igra):
    # if there is a call letter, then omit the call letter station.
    ncep_stnid_call = read_NCEP_STNID_CALL()
    
    id_with_call = stations_ncep_igra.index[stations_ncep_igra.index.isin(ncep_stnid_call.index)]
    call_letters = ncep_stnid_call.loc[id_with_call, 'call']
    stations_ncep_igra.drop(index=call_letters[call_letters.isin(stations_ncep_igra.index)], inplace=True)
    return stations_ncep_igra

def extract_datetime_str(df_pre, df_pre_index):
    df_pre['year'] = df_pre_index.year
    df_pre['month'] = df_pre_index.month
    df_pre['day'] = df_pre_index.day
    df_pre['hour'] = df_pre_index.hour
    
    yyyy =  df_pre.year.astype(str) 
    
    mm = df_pre.month.astype(str) 
    mm[df_pre.month<=9] = '0'+ mm[df_pre.month<=9] 
    
    dd = df_pre.day.astype(str)
    dd[df_pre.day<=9] = '0' + dd[df_pre.day<=9]
    
    hh = df_pre.hour.astype(str)
    hh[df_pre.hour<=9] = '0' + hh[df_pre.hour<=9]
    
    pre_datetime = yyyy + ' ' + mm + ' ' + dd + ' ' + hh
    return pre_datetime

def flag_pre(df):
    # create new 'wwflag' column. 1 for rain, 2 for snow
    df['wwflag']=np.nan
    mask_rain=(((df['ww']>=60)&(df['ww']<=69)) |
          ((df['ww']>=80)&(df['ww']<=84)) |
          ((df['ww']>=87)&(df['ww']<=99)))
    df.loc[mask_rain,'wwflag']=1

    mask_snow=((df['ww']==85) |
          (df['ww']==86) |
          ((df['ww']>70)&(df['ww']<=79)))
    df.loc[mask_snow, 'wwflag']=2
    return df