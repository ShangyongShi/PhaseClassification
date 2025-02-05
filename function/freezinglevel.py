#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:34:09 2023

@author: sshi
"""
import numpy as np
from  function.mymath import linear

def freezing_level_t_height(tt, gph):
    '''
    Return all the level heights where temperature change from + to -
    Froidurot: the highest level where temp is above 0C
    Inputs:
        tt: list of temperature
        gph: corresponding geopotential heights
    Output:
        z: list
    '''
    nonan = ~np.isnan(tt) & ~np.isnan(gph)
    tt = tt[nonan]
    gph = gph[nonan]
    
    below_ge_zero = (tt[:-1].values>=0)
    above_lt_zero = (tt[1:].values<0)
    
    uppers = np.where( below_ge_zero & above_lt_zero)[0]  
    
    z = []
    if len(uppers)==0 or (max(uppers) ==len(tt) - 1):
        z = [np.nan]
    else:
        for idx in uppers:
            value_zero = ( tt[idx]==0)
            
            if idx==0:
                if value_zero: 
                    continue
                t1, z1, t2, z2 = tt[idx], gph[idx], tt[idx+1], gph[idx+1]
                slope, intercept = linear(t1, z1, t2, z2)
                z.append(intercept)
            else:   
                # if sounding does not cross zero not a fl
                two_sides_same_sign = (tt[idx-1] * tt[idx+1] > 0)
                sounding_not_cross_zero = value_zero & two_sides_same_sign
                if sounding_not_cross_zero: 
                    continue
                
                # if below is all negative, and multiple consecutive zeros 
                # in the sounding, does not count as freezing level
                below_is_zero = tt[idx-1]==0
                if value_zero & below_is_zero:
                    if idx==1: # 0 0 nege
                        continue
                    else:
                        belows_are_zero = tt[:idx]==0
                        if sum(~belows_are_zero)==0: # belows are all zero
                            continue
                        else: # the first value below zeros is nega
                            for i in range(-1, -len(belows_are_zero)-1, -1):
                                if belows_are_zero[i]==False:
                                    break
                            first_nonzero_below = tt[:idx][i]
                            if first_nonzero_below <0:
                                continue            
            
                t1, z1, t2, z2 = tt[idx], gph[idx], tt[idx+1], gph[idx+1]
                slope, intercept = linear(t1, z1, t2, z2)
                z.append(intercept)
    if len(z) == 0 : z = [np.nan]
    return z