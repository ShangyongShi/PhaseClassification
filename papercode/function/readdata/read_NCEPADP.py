#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:55:58 2023

@author: sshi
"""
import pandas as pd
import numpy as np
import os

from ..watervapor import td2tw
from ..slp2sp import slp2sp
from .maskdf import flag_pre



# def flag_pre(df):
#     # create new 'wwflag' column. 1 for rain, 2 for snow
#     df['wwflag']=np.nan
#     mask_rain=(((df['ww']>=60)&(df['ww']<=69)) |
#           ((df['ww']>=80)&(df['ww']<=84)) |
#           ((df['ww']>=87)&(df['ww']<=99)))
#     df.loc[mask_rain,'wwflag']=1

#     mask_snow=((df['ww']==85) |
#           (df['ww']==86) |
#           ((df['ww']>70)&(df['ww']<=79)))
#     df.loc[mask_snow, 'wwflag']=2
#     return df

##############################################################################
# Read NCEP ADP data. Combine ds464.0 and ds461.0
def read_NCEP_STNID_CALL():
    ncep_stnid_call = pd.read_csv('/r1/sshi/sounding_phase/data/NCEP_STNID_CALL.txt',
                                  header=None, names=['stnid', 'call'], dtype=str)
    ncep_stnid_call.set_index('stnid', inplace=True)
    ncep_stnid_call = ncep_stnid_call[~ncep_stnid_call.call.isna()]
    return ncep_stnid_call


def read_464(ID):
    # ds464.0
    file = '/data/sshi/NCEPADP/stations/'+ID+'.txt'
    colNames=['flag', 'year','month','day','hour','time','lat', 'lon', 'elev',
              'ww', 'pw', 'slp', 'p', 't', 'td']
    if not os.path.exists(file):
        df464 = pd.DataFrame(columns=colNames)
    else:
        df464 = pd.read_csv(file, names=colNames, sep='\t')
        df464.index = pd.to_datetime({'year':df464.year, 'month':df464.month,
                                      'day':df464.day, 'hour':df464.hour})
        df464.loc[df464.ww>100, 'ww'] = np.nan
        df464.loc[df464.t>900, 't'] = np.nan
        df464.loc[df464.td>90, 'td'] = np.nan
        df464.loc[df464.slp>9000, 'slp'] = np.nan
        df464.loc[df464.p>9000, 'p'] = np.nan
        
        df464 = df464[~df464.ww.isna()]
    return df464


def read_461(ID):
    # ds461.0
    colNames = ['datetime', 'sfctype', 'type', 'lat', 'lon', 'elev',
                't', 'td', 'p', 'slp', 'tp3', 'tp24', 'ww']
    file = '/data/sshi/ds461.0/stations/all/'+ID+'.txt'
    if not os.path.exists(file):
        df461 = pd.DataFrame(columns=colNames)
    else:
        df461 = pd.read_csv(file, names=colNames, sep='\t', skiprows=1)
    
        df461.datetime = pd.to_datetime(df461.datetime)
        df461.set_index('datetime', inplace=True)
        df461.set_index(df461.index.round('60min'), inplace=True)
        df461.t -= 273.15
        df461.td -= 273.15
        df461.p /= 100
        df461.slp /= 100
    
        df461.loc[df461.ww<0, 'ww'] = np.nan
        df461.loc[df461.p<0, 'p'] = np.nan # added 20230222
        df461.loc[df461.slp<0, 'slp'] = np.nan
        df461.loc[df461.t<-90, 't'] = np.nan
        df461.loc[df461.td<-90, 'td'] = np.nan
        
        df461.loc[df461.lon<0, 'lon'] += 360
        
        df461 = df461[~df461.ww.isna()]
    return df461


def read_stnid_callid(ID, dataset):
    '''
    to read the NCEPADP data based on 5-digit WMO station id.
    data for stations with call letters available would be included.
    
    ID: 5-digit str
    dataset: str, '1' for ds461.0, '4' for ds464.0
    '''
    # some stations have call letters
    ncep_stnid_call = read_NCEP_STNID_CALL()
    if ID in ncep_stnid_call.index:
        
        callid = ncep_stnid_call.loc[ID].values[0]
        
        if dataset=='4':
            df1 = read_464(ID)
            df2 = read_464(callid)
        elif dataset=='1':
            df1 = read_461(ID)
            df2 = read_461(callid)
            
        df = pd.concat([df1, df2])
        df.sort_index(inplace=True)  
    else:
        if dataset=='4':
            df = read_464(ID)
        elif dataset=='1':
            df = read_461(ID)
            
    return df




def read_NCEPADP(ID):
    '''
    read NCEPADP data, combining ds464.0 and ds461.0
    
    Input: NCEP_ID
    output: df
    - use the first location information if multiple are available
    - tw is calculated if t and td and p (or slp+elev) is available
    - wwflag is flagged, 1 for rain and 2 for snow. 9 for others
    - datetime is the index
    '''
    df464 = read_stnid_callid(ID, '4')
    df461 = read_stnid_callid(ID, '1')
    ##### combine NCEPADP
    df = pd.concat([df464.loc[:, ['lat', 'lon', 'elev', 't', 'td', 'p', 'slp',
                                  'ww']],
                    df461.loc[:, ['lat', 'lon', 'elev', 't', 'td', 'p', 'slp',
                                  'ww']]])
    # remove duplicated lines
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)
    
    # correct the location information
    if len(df)>1:
        location = df.loc[:, 'lat':'elev'].mode().values[0]
        df.lat, df.lon, df.elev = location[0], location[1], location[2]
    # calculate tw
    df['tw'] = np.nan
    idxs = df[  
              (~df['t'].isna()) &
              (~df['td'].isna()) &
              ( (~df['slp'].isna()) | (~df['p'].isna()) )
              ].index
    for idx in idxs:
        tc = df.loc[idx, 't']
        tdc = df.loc[idx, 'td']
        pmb = df.loc[idx, 'p']
        if (~np.isnan(pmb)):
            df.loc[idx, 'tw'] = td2tw(tc, pmb, tdc)
        else:
            pmb = slp2sp(df.loc[idx, 'slp'], df.loc[idx, 'elev'], tc)
            df.loc[idx, 'tw'] = td2tw(tc, pmb, tdc)
            
    # wwflag
    df = flag_pre(df)
    
    df.index.rename('datetime', inplace=True)
    return df
