#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


2023.5.8
Important update to the package:
    freezing rain changed from rain to snow.
    calculate from the surface to surface pressure minus 300 hPa
    If multiple positive or negative are calculated, sum them up
    
This package includes following functions:
Given temperature and pressure profiles,
    - find freezing level heights (from above freezing to below freezing)
       
    - calculate total melting / refreezing energy
    
    - identify sounding type 
    
THe energy would be calculated up to the pree=ssure level that is closest
to surface pressure minus 350 hPa
If the melting cross this pressure level, keep


Created on Thu Feb 16 11:19:06 2023
@author: sshi
"""


import numpy as np


from sklearn.linear_model import LinearRegression

def linear(t1, p1, t2, p2):
    '''
    
    Input: two data pairs
    Output: slope and intercept

    '''
    slope = (p2 - p1)/(t2 - t1)
    intercept = p1 - slope*t1
    return slope, intercept


# ----------------------------------------------------------------
# codes for calculating freezing level height
# ----------------------------------------------------------------
def freezing_level_height(tt, gph):
    '''
    Return all the level heights where temperature change from + to -
    
    Inputs:
        tt: list of temperature
        gph: corresponding geopotential heights
    Output:
        z: list
    '''
    z = []
    
    uppers = freezing_level_idx(tt, gph)
    
    for idx in uppers:
        t1, z1, t2, z2 = tt[idx], gph[idx], tt[idx+1], gph[idx+1]
        slope, intercept = linear(t1, z1, t2, z2)
        z.append(intercept)
        
    if len(z) == 0 : z = [np.nan]
    
    return z


def freezing_level_idx(tt, z):
    '''
    find the index for locating freezing level
    temperature at the index >=0, at the index+1 < 0
    
    excluded situation: if the t[idx]=0 and...
        - idx is at surface level
        - sounding does not cross zero: above and below<0, 
        - below it, the temperatures are all zero
        - below it, the only nonzero value is negative
        
    we can use this index to calculate freezing level height or energy area 
    
    Input:
        tt: temperatures
        z: vertical axis, geopotential heights or pressure
        
    '''
    nonan = ~np.isnan(tt) & ~np.isnan(z)
    tt = tt[nonan]
    z = z[nonan]
    
    below_ge_zero = (tt[:-1].values>=0)
    above_lt_zero = (tt[1:].values<0)
    
    uppers = list(np.where( below_ge_zero & above_lt_zero)[0]  )
    
       
    for idx in uppers:
        value_zero = ( tt[idx]==0)
        
        if idx==0:
            if value_zero: 
                uppers.remove(idx)  # surface level is zero
        else:   
            # if sounding does not cross zero, not a fl
            two_sides_same_sign = (tt[idx-1] * tt[idx+1] > 0)
            sounding_not_cross_zero = value_zero & two_sides_same_sign
            if sounding_not_cross_zero: 
                uppers.remove(idx)
            
            # if below is all negative, and multiple consecutive zeros 
            # in the sounding, does not count as freezing level
            below_is_zero = tt[idx-1]==0
            if value_zero & below_is_zero:
                if idx==1: # 0 0 nege
                    uppers.remove(idx)
                else:
                    belows_are_zero = tt[:idx]==0
                    if sum(~belows_are_zero)==0: # belows are all zero
                        uppers.remove(idx)
                    else: # the first value below zeros is nega
                        for i in range(-1, -len(belows_are_zero)-1, -1):
                            if belows_are_zero[i]==False:
                                break
                        first_nonzero_below = tt[:idx][i]
                        if first_nonzero_below <0:
                            uppers.remove(idx)                       
    return uppers

# ----------------------------------------------------------------
# codes for calculating energy area
# ----------------------------------------------------------------
def cal_energy_area(tt, pp):
    '''
    main function
    given the sounding profile tt and pp, return the positive areas
    and the negative areas (list)
    
    Input: tt, pp
    Output: areas, positive_areas, negative_areas
    
    '''
    
    tt = tt[~np.isnan(tt) & ~np.isnan(pp)]
    pp = pp[~np.isnan(tt) & ~np.isnan(pp)]
    
    if len(tt) <2 : # not enough data
        return [np.nan], [np.nan]
    
    # find layers enclosed by 0C line and the tt profile
    idx = find_layer_idx(tt)
    
    areas = []
    positive_areas = []
    negative_areas = []
    if len(idx)==1:  # sounding does not cross T=0
        return [0], [0]                
    else:     
        for i in range(0, len(idx)-1):
            is_surface_layer = (i==0)
            # loop through each layer
            idx_bottom = idx[i]
            idx_top = idx[i+1]
            area = layer_area(tt, pp, idx_bottom, idx_top, is_surface_layer)
            areas.append(area)
            
            
            if area > 0:
                positive_areas.append(area)
            elif area<0: #fixed 2023.3.10, avoid lowest level=0 and area=0
                negative_areas.append(area)
                
    if len(negative_areas)==0: negative_areas=[0]
    if len(positive_areas)==0: positive_areas=[0]
    
    return areas, positive_areas, negative_areas   

def find_layer_idx(tt, pp):
    '''
    For the purpose of calculating energy areas
    find the index of the data pairs in the sounding
    idx and idx+1 are lines crossing T=0
    
    Added 2023.5.9
    the pressure should be higher than surface pressure-300 (+-50)
    calculation started from the highest freezing level 
    below the pressure level around surface pressure-350 Hpa
    
    Also, we will add all the posi or nega areas together
    so it won't matter if the sounding crosses or not
    (it would matter for the determination of freezing level height)
    
    # corrected 2023.3.9
    # if tt[idx+1] and tt[idx-1] same sign, 
    # means the sounding does not cross 0. 
    # need to consider such situation
    '''
    
    
    # >=0 to <0
    uppers = list(np.where((tt[:-1].values>=0) & (tt[1:].values<0))[0] )
    
    for idx in uppers:
       if pp[idx]<pp[0]-350:
           uppers.remove[idx]
    
    # negatives to positives
    lowers = list(np.where((tt[0:-1].values<=0) & (tt[1:].values>0))[0] )
    
    for idx in lowers:
        if idx>max(uppers):
            lowers.remove[idx]
    
    idx = [0] + lowers + uppers
    
    # # commented 2023.5.9
    # # if sounding just touches zero but not crosses, remove index
    # for i in idx:
    #     if i==0:
    #         continue
    #     else:
    #         value_zero = ( tt[i]==0)
    #         two_sides_same_sign = (tt[i-1] * tt[i+1] > 0)
            
    #         sounding_not_cross_zero = value_zero & two_sides_same_sign
    #         if sounding_not_cross_zero: 
    #             idx.remove(i)
                
    idx.sort()
    return idx


def layer_area(tt, pp, idx_bottom, idx_top, is_surface_layer):
    '''
    calculate the area of one layer, given the full t proifle, 
    and the index at the bottom and the top of the layer
    
    e.g.
        tt = [1, 0.5, -1, -2, -1, 1] #C
        pp = [1000, 990, 985, 950, 925, 900] # mb
        idx_bottom = [1]
        idx_top = [4]
        
    Added 2023.3.9
        is_surface_layer: True or False
    If true, the bottom value would be tt[0] and pp[0]
    instead of the interpolated value between indexes idx_bottom and idx_top
    '''
    Rd = 287
    pp = np.log(pp)

    # bottom ln(p) of the layer
    
    # if idx_bottom==0: # corrected 2023.3.9
    if is_surface_layer: 
        # this is to calculate area from surface to first 0C level
        # the first 0C level is between p[0] and p[1], 
        # so idx_bottom and idx_top are both zero
        p_bottom = pp[0]
        t_bottom = tt[0]
    else:
        t1, p1 = tt[idx_bottom], pp[idx_bottom]
        t2, p2 = tt[idx_bottom+1], pp[idx_bottom+1]
        _, p_bottom = linear(t1, p1, t2, p2)
        t_bottom = 0
        
    # top ln(p) of the layer   
    t1, p1, t2, p2 = tt[idx_top], pp[idx_top], tt[idx_top+1], pp[idx_top+1]
    slope, p_top = linear(t1, p1, t2, p2)    
    t_top = 0
    
    # all vertices of the layer 
    p_middle = list(pp[idx_bottom+1 : idx_top+1].values)
    t_middle = list(tt[idx_bottom+1 : idx_top+1].values)
    ps = [p_bottom] + p_middle + [p_top]
    ts = [t_bottom] + t_middle + [t_top]
    area = 0
    for it in range(len(ts)-1):
        # add areas
        area += (ts[it] + ts[it+1])/2 * (ps[it] - ps[it+1])
    area *= Rd       
    
    return area

# ----------------------------------------------------------------
# codes for determining sounding type
# ----------------------------------------------------------------
fl = freezing_level_height(tt, gph)

