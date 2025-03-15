#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:35:06 2023

@author: sshi
"""
import pandas as pd
import numpy as np
from ..watervapor import rh2tw, rh2ti, td2rh

def read_sounding(datapath, IGRA_ID, NCEP_ID):
    '''
    read the sounding files I produced from the original IGRA data
    Inputs:
        datapath: root directory of the data, 
                  include /sounding, /pre_datetime, etc
        IGRA_ID: str
        NCEP_ID: str, usually 5-digit number, can be chars
    Outputs:
        p, t, rh, tw, z: dataframes, Datetime as index
    '''
    file = datapath+'/sounding/temp/'+IGRA_ID + '_'+NCEP_ID+'_temp_sounding.txt'
    df = pd.read_csv(file)
    df.set_index(pd.to_datetime(df.loc[:, 'year':'hour']), inplace=True)

    file2 = datapath+'/sounding/rh/'+IGRA_ID + '_'+NCEP_ID+'_rh_sounding.txt'
    df2 = pd.read_csv(file2)
    df2.set_index(pd.to_datetime(df2.loc[:, 'year':'hour']), inplace=True)
    
    file3 = datapath+'/sounding/gph/'+IGRA_ID + '_'+NCEP_ID+'_gph_sounding.txt'
    df3 = pd.read_csv(file3)
    df3.set_index(pd.to_datetime(df3.loc[:, 'year':'hour']), inplace=True)

    # modified 2023.2.17
    # use dpdp if rh is unavailable
    file4 = datapath+'/sounding/dpdp/'+IGRA_ID + '_'+NCEP_ID+'_dpdp_sounding.txt'
    df4 = pd.read_csv(file4)
    df4.set_index(pd.to_datetime(df4.loc[:, 'year':'hour']), inplace=True)

    # sounding data. Datetime as index
    p = pd.DataFrame(index=df.index, columns=df.loc[:, '0':].columns)
    t = pd.DataFrame(index=df.index, columns=df.loc[:, '0':].columns) 
    rh = pd.DataFrame(index=df.index, columns=df.loc[:, '0':].columns) 
    z  = pd.DataFrame(index=df.index, columns=df.loc[:, '0':].columns) 
    dpdp = pd.DataFrame(index=df.index, columns=df.loc[:, '0':].columns) 
    for col in df.loc[:, '0':].columns:
        p.loc[:, col] = df.loc[:, col].map(lambda x: x.split(',')[0]).astype(float)
        t.loc[:, col] = df.loc[:, col].map(lambda x: x.split(',')[1]).astype(float)
        rh.loc[:, col] = df2.loc[:, col].map(lambda x: x.split(',')[1]).astype(float)
        z.loc[:, col] = df3.loc[:, col].map(lambda x: x.split(',')[1]).astype(float)
        dpdp.loc[:, col] = df4.loc[:, col].map(lambda x: x.split(',')[1]).astype(float)
        
    # tw = pd.DataFrame(data=np.nan, index=p.index, columns=p.columns)
    # for row in p.index:
    #     for col in p.columns:
    #         pmb = p.loc[row, col]
    #         tc = t.loc[row, col]
    #         r = rh.loc[row, col]
    #         dp = dpdp.loc[row, col]
    #         if ~np.isnan(pmb) & ~np.isnan(tc) :
    #             if ~np.isnan(r): 
    #                 tw.loc[row, col] = rh2tw(pmb, tc, r)
    #             else: # if rh not available, use dpdp if not nan
    #                 if ~np.isnan(dp):
    #                     tdc = tc - dp
    #                     tw.loc[row, col] = td2tw(tc, pmb, tdc)
    #                 else:
    #                     tw.loc[row, col] = np.nan
    #         else:
    #             tw.loc[row, col] = np.nan
                    
    return p, t, rh, z, dpdp

def sounding_tw_or_ti(p, t, rh, dpdp, wori):
    tw = pd.DataFrame(data=np.nan, index=p.index, columns=p.columns)
   
    for row in p.index:
        for col in p.columns:
            pmb = p.loc[row, col]
            tc = t.loc[row, col]
            r = rh.loc[row, col]
            dp = dpdp.loc[row, col]
            if ~np.isnan(pmb) & ~np.isnan(tc) :
                if ~np.isnan(r): 
                    if wori==0:
                        tw.loc[row, col] = rh2tw(pmb, tc, r)
                    else:
                        tw.loc[row, col] = rh2ti(pmb, tc, r)
                            
                else: # if rh not available, use dpdp if not nan
                    if ~np.isnan(dp):
                        tdc = tc - dp
                        r = td2rh(tc, tdc)
                        
                        if wori==0:
                            tw.loc[row, col] = rh2tw(pmb, tc, r)
                        else:
                            tw.loc[row, col] = rh2ti(pmb, tc, r)
                    else:
                        tw.loc[row, col] = np.nan
            else:
                tw.loc[row, col] = np.nan
    return tw