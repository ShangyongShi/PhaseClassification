#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:20:37 2023

@author: sshi
"""

import numpy as np



def cal_lapse_rate(t, p, z):
    '''
    Calculate the low level 500m lapse rate based on IGRA sounding data.
    The lower level is the pressure level just above the surface pressure.
    The uper level is the level higher than the lower level 
    with a geopotential difference closest to 500m
    ---
    Input:
        p: sounding pressure, unit: hPa, pd.Series
        t: sounding temperature, unit:C, pd.Series
        z: sounding geopotential height, unit: m,pd.Series
    ---
    Output:
        g0_t: low level 500-m lapse rate, unit: C/km

    '''
    z = fill_gph_nan(t, p, z)
    
    t = t[~t.isna() & ~z.isna()]
    z = z[~t.isna() & ~z.isna()]
    
    if len(t):
        lev = np.argmin(abs(z-z[0]-500))
        dz = z.iloc[lev] - z[0]
        dt = t.iloc[lev] - t[0]
        
        if dz<250 or dz>750:
            g0_t = np.nan
        else:
            g0_t = -dt/dz*1000
    else:
        g0_t = np.nan
    return g0_t


def fill_gph_nan(tt, pp, zz):
    gph = zz.copy()
    for i in range(1, len(tt)):
        if np.isnan(zz[i]):
            # find the nearest available pairs for interpolation
            # first find if there are anything below
            # if not look up
            if np.isnan(tt[i]) | np.isnan(pp[i]):
                continue
            idx_lower = (~np.isnan(tt[0:i]) & 
                         ~np.isnan(pp[0:i]) & 
                         ~np.isnan(zz[0:i]))
            
            idx_upper = (~np.isnan(tt[i+1:]) & 
                             ~np.isnan(pp[i+1:]) & 
                             ~np.isnan(zz[i+1:]))
            
            if sum(idx_lower)>0:
                tt_low = tt[0:i][idx_lower][-1]
                pp_low = pp[0:i][idx_lower][-1]
                zz_low = zz[0:i][idx_lower][-1]
                tt_top = tt[i]
                pp_top = pp[i]  
                
                gph[i] = zz_low + 287/9.8*((tt_top+tt_low)/2+273.15)* \
                        (np.log(pp_low)-np.log(pp_top))
                        
            elif sum(idx_upper)>0:
                tt_low = tt[i]
                pp_low = pp[i]
                tt_top = tt[i+1:][idx_upper][0]
                pp_top = pp[i+1:][idx_upper][0]
                zz_top = zz[i+1:][idx_upper][0]
                
                gph[i] = zz_top - 287/9.8*((tt_top+tt_low)/2+273.15)* \
                        (np.log(pp_low)-np.log(pp_top))
            else:
                continue
            
    return gph