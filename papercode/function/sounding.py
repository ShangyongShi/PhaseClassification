#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2023.7.26
makes it the same as exp_1.5
update sounding_type
more than 3 layers only use the first 3 layers

2023.6.8
Update:
    cutoff threshold change: calculate from the surface to 2km
    need to use the measured geopotential height. If missing,
    calculate from temperature and pressure soundings
    

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

cutoff = 2000
import numpy as np

Rd = 287
g = 9.8


        
        

def linear(t1, p1, t2, p2):
    '''
    
    Input: two data pairs
    Output: slope and intercept

    '''
    if t1==t2:
        return np.nan, np.nan
    slope = (p2 - p1)/(t2 - t1)
    intercept = p1 - slope*t1
    return slope, intercept





# ----------------------------------------------------------------
# codes for calculating freezing level height
# ----------------------------------------------------------------
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

def freezing_level_height(tt, gph):
    '''
    Return all the level heights where temperature change from + to -
    
    
    Inputs:
        tt: list of temperature
        gph: corresponding geopotential heights
            2024.1.2: this is height differences from the surface
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
    tt = np.array(tt)
    z = np.array(z)
    
    
    nonan = ~np.isnan(tt) & ~np.isnan(z)
    tt = tt[nonan]
    z = z[nonan]
    
    below_ge_zero = (tt[:-1]>=0)
    above_lt_zero = (tt[1:]<0)
    
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

def level_height(t, p):
    '''use t and p profiles, calculate correspongding heights
    based on hydrostatic equation
    Input 
        t in C
        p in mb
    Output
        z in m
    '''
    
    dz = 0
    z = [0]
    for i in range(len(t)-1):
        t1 = t[i]
        t2 = t[i+1]
        p1 = p[i]
        p2 = p[i+1]
        
        t_mean = (t1+t2)/2+273.15
        dz += Rd/g*t_mean*np.log(p1/p2) 
        z += [dz]
    return np.array(z)

def t_p_at_cutoff_height(tt, pp, zz, idx_cutoff):
    '''
    use t and p profiles, get the t and p at 2km
    
    first use hydrostatic to calculate height at each data point
    find the data pairs around the cutoff height
    calculate the linear relation between t and lnp
    then get the t at the pressure at the cutoff height
    
    Input:
        t: C
        p: mb
    Output:
        t_cutoff: C
        p_cutoff:mb

    '''
    cutoff = 2000
   
    
    z1 = zz[idx_cutoff]
    p1 = pp[idx_cutoff]
    t1 = tt[idx_cutoff]+273.15
    t2 = tt[idx_cutoff+1]+273.15
    p2 = pp[idx_cutoff+1]
    
    t_mean = (t1+t2)/2
    p_cutoff = p1/(np.exp(g/Rd/t_mean*(cutoff-z1)))
    
    slope, intercept = linear(t1, np.log(p1), t2, np.log(p2))
    if np.isnan(slope):
        t_cutoff = t1-273.15
    else:
        t_cutoff = (np.log(p_cutoff)-intercept)/slope-273.15
    
    return t_cutoff, p_cutoff



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
    
    nonan = ~np.isnan(t) & ~np.isnan(z)
    t = t[nonan]
    z = z[nonan]
    
    if len(t):
        lev = np.argmin(abs(z-z[0]-500))
        dz = z[lev] - z[0]
        dt = t[lev] - t[0]
        
        if dz<250 or dz>750:
            g0_t = np.nan
        else:
            g0_t = -dt/dz*1000
    else:
        g0_t = np.nan
    return g0_t

# ----------------------------------------------------------------
# codes for calculating energy area
# ----------------------------------------------------------------
def cal_energy_area(tt, pp, zz):
    '''
    main function
    given the sounding profile tt and pp, return the positive areas
    and the negative areas (list)
    
    Input: 
        tt, pp
        zz: optional
    Output: areas, positive_areas, negative_areas
    
    '''
    tt = np.array(tt)
    pp = np.array(pp)
    
    
    nonan = ~np.isnan(tt) & ~np.isnan(pp)
    tt = tt[nonan]
    pp = pp[nonan]
    
    if len(tt) <2 : # not enough data
        return [], [np.nan], [np.nan]   
    
    
    if zz is None: # if no input of z, calculate from t and p
        zz = level_height(tt, pp)
        zz = np.array(zz)
    else:
        zz = fill_gph_nan(tt, pp, zz)
        zz = [0] + list(zz[1:]-zz[0]) # minus elevation = layer thickness
        zz = np.array(zz)
        
    
    # idx and idx+1 cross cutoff height
    cutoff = 2000
    idx_cutoff = np.where((zz[0:-1]<cutoff) & (zz[1:]>=cutoff))[0]
        
    def cutoff_at_height(tt, pp, zz, idx_cutoff):
        '''
        return the profile cutoff at threshold height (2km from surface)
        the temperature and pressure would be calculated at the threshold 
        height based on the t, p pairs countering 2km
        
        '''        
        if len(idx_cutoff)==0:
            return tt, pp, zz
        else:
            t_cutoff, p_cutoff = t_p_at_cutoff_height(tt, pp, zz, idx_cutoff[0])
            tt = np.append(tt[:idx_cutoff[0]+1], t_cutoff)
            pp = np.append(pp[:idx_cutoff[0]+1], p_cutoff)
            zz = np.append(zz[:idx_cutoff[0]+1], 2000)
        return tt, pp, zz
    
    
    # cutoff the sounding at 2km above the surface
    tt, pp, zz = cutoff_at_height(tt, pp, zz, idx_cutoff)
          
        
        
    # find layers enclosed by 0C line and the tt profile
    idx = find_layer_idx(tt, pp)
    if len(idx_cutoff)>0:
        idx.append(idx_cutoff[0]+1)
    
    
    areas = []
    positive_areas = []
    negative_areas = []
    if len(idx)==1:  # sounding does not cross T=0
        return [], [0], [0]                
    else:     
        for i in range(0, len(idx)-1):
            is_surface_layer = (i==0)
            
            if len(idx_cutoff)>0:#
                is_top_layer = (idx[i+1]==idx_cutoff[0]+1)
            else:
                is_top_layer=False
            
            # loop through each layer
            idx_bottom = idx[i]
            idx_top = idx[i+1]
            area, flag = layer_area(tt, pp, zz, 
                              idx_bottom, idx_top, 
                              is_surface_layer, is_top_layer)
            if area!=0:
                areas.append(area)
            
            if area > 0:
                positive_areas.append(area)
            elif area<0: #fixed 2023.3.10, avoid lowest level=0 and area=0
                negative_areas.append(area)
                
    if len(areas)==0: areas=[]       
    if len(negative_areas)==0: negative_areas=[np.nan]
    if len(positive_areas)==0: positive_areas=[np.nan]
    
    return areas, positive_areas, negative_areas   

def find_layer_idx(t, p):
    '''
     modified 2023.6.8
    cutoff change to surface + 2km
    so we need z sounding or calculate from t and p
    do not sum the areas, but use the dominant one or two layers.
    see categotize for details
    ------
    2023.5.21
    commented the pressure - 300. want to play with 
    WCW sounding to see whether to classify it as type1 or type 2
    
    ------
    For the purpose of calculating energy areas
    find the index of the data pairs in the sounding
    idx and idx+1 are lines crossing T=0
    
    Added 2023.5.9
    the pressure should be higher than surface pressure-300 (+-50)
    calculation started from the highest freezing level 
    below the pressure level around surface pressure-300 Hpa
    
    Also, we will add all the posi or nega areas together
    so it won't matter if the sounding crosses or not
    (it would matter for the determination of freezing level height)
    
    # corrected 2023.3.9
    # if tt[idx+1] and tt[idx-1] same sign, 
    # means the sounding does not cross 0. 
    # need to consider such situation
    '''
    tt = np.array(t)
    pp = np.array(p)
    # tt = t.values
    # pp = p.values
    # >=0 to <0
    uppers = list(np.where((tt[:-1]>=0) & (tt[1:]<0))[0] )
    lowers = list(np.where((tt[0:-1]<=0) & (tt[1:]>0))[0] )
    
    # # cutoff pressure level
    # for i in uppers:
    #    if pp[i]<pp[0]-300:
    #        uppers.remove(i)
    
    
    # if surface temp=0, remove
    if (len(lowers)>0) :
        if (lowers[0]==0) & (tt[0]==0):
            lowers.remove(0)
            
    if len(uppers)>0:
        if (uppers[0]==0) & (tt[0]==0):
            uppers.remove(0)
            
    # # remove indexes that have smaller pressure than cutoff pressure
    # if len(uppers)>0:
    #     for i in lowers:
    #         if i>max(uppers):
    #             lowers.remove(i)
    # else:
    #     lowers = []
    
    
    idx = [0] + lowers + uppers
    
    
    # if sounding just touches zero but not crosses, remove index
    for i in idx[1:]:
        value_zero = ( tt[i]==0)
        two_sides_same_sign = (tt[i-1] * tt[i+1] > 0)
        
        sounding_not_cross_zero = value_zero & two_sides_same_sign
        if sounding_not_cross_zero: 
            idx.remove(i)
                
    idx.sort()
    return idx


def layer_area(tt, pp, zz, 
               idx_bottom, idx_top,  
               is_surface_layer, is_top_layer):
    '''
    calculate the area of one layer, given the full t, p, z proifle, 
    and the index at the bottom and the top of the layer.
    whether the layer is surface or cutoff by the threshold height
    is also considered.
    
    e.g.
        tt = [1, 0.5, -1, -2, -1, 1] #C
        pp = [1000, 990, 985, 950, 925, 900] # mb
        idx_bottom = [1]
        idx_top = [4]
        is_surface_layer = True
        is_top_layer = False
       
    Input:
        tt, pp, zz | list or arrays of sounding t (C), p (mb), z (m)
        
        idx_bottom, idx_top | the bottom/top of the layer
            where t[idx] and t[idx+1] cross 0C
            
        idx_cutoff | the cutoff height of 2km lies between idx and idx+1
        
        is_surface_layer | True or False # Added 2023.3.9
            If true, the bottom value would be tt[0] and pp[0]
            instead of the interpolated value between indexes 
            idx_bottom and idx_top
    
        is_top_layer | True or False #Added 2023.6.8
            if the layer is cut through by the cutoff height
            If true, will calculate the energy area below the interpolated
            pressure at 2km from the surface
    
    Output:
        area: the energy area of this layer
        flag: True or False, if this layer is cutoff by the threshold height
            
        
    '''
        
    Rd = 287
    pp = np.log(pp)

    flag = is_top_layer
    # bottom ln(p) of the layer
    
    # if idx_bottom==0: # corrected 2023.3.9
    if is_surface_layer: 
        '''
        this is to calculate area from surface to first 0C level
        the first 0C level is between p[0] and p[1], 
        so idx_bottom and idx_top are both zero'''
        p_bottom = pp[0]
        t_bottom = tt[0]
        if is_top_layer:
            '''the layer is cutoff by the threshold height
            calculate the available area enclosed by 0C and the sounding below
            '''
            #t_top, p_top = t_p_at_cutoff_height(tt, np.exp(pp), zz, idx_cutoff)
            t_top = tt[-1]
            p_top = pp[-1]
        else:
            # top ln(p) of the layer   
            t1, p1 = tt[idx_top], pp[idx_top]
            t2, p2 = tt[idx_top+1], pp[idx_top+1]
            _, p_top = linear(t1, p1, t2, p2)    
            t_top = 0  
    else:
        t1, p1 = tt[idx_bottom], pp[idx_bottom]
        t2, p2 = tt[idx_bottom+1], pp[idx_bottom+1]
        _, p_bottom = linear(t1, p1, t2, p2)
        t_bottom = 0
        
        if is_top_layer:
            #t_top, p_top = t_p_at_cutoff_height(tt, pp, zz, idx_cutoff)
            t_top = tt[-1]
            p_top = pp[-1]
        else:
            # top ln(p) of the layer   
            t1, p1 = tt[idx_top], pp[idx_top]
            t2, p2 = tt[idx_top+1], pp[idx_top+1]
            slope, p_top = linear(t1, p1, t2, p2)    
            t_top = 0
     
    # all vertices of the layer 
    p_middle = list(pp[idx_bottom+1 : idx_top+1])
    t_middle = list(tt[idx_bottom+1 : idx_top+1])
    ps = [p_bottom] + p_middle + [p_top]
    ts = [t_bottom] + t_middle + [t_top]
    area = 0
    for it in range(len(ts)-1):
        # add areas
        area += (ts[it] + ts[it+1])/2 * (ps[it] - ps[it+1])
    area *= Rd       
    
    return area, flag

# ----------------------------------------------------------------
# codes for determining sounding type
# ----------------------------------------------------------------
def sounding_type(areas, heights):
    '''
    Identify the sounding type based on areas and freezing level heights
    
    If one layer, would be type 1 if freeing level height<2km 
    If two layers,
        Type 0 if all cold (C-C)
        Type 1 if surface above freezing (W)
        Type 2 if surface below freezing (C) and aloft layer above freezing (W)
    If three layers,
        Type 0 if all cold 
        Type 2 if surface below freezing
        Type 1 if surface above freezing (W), except when the surface is 
            too small and would be ignored to make it Type 2
    More than three layers, only use the first 3 layers
    
        
    Input:
        areas, heights
    Output:
        sounding_type
        
    '''
    areas = np.array(areas)
    heights = np.array(heights)
    

        
    if len(areas)>=4:
        areas=areas[0:3]
        
    if (len(areas)==0):
        TYPE = 0
    else:
        lowest_area = areas[0]
        if len(areas)==1:
            TYPE = 0
            if (lowest_area>0) :
                if (len(heights)>0): 
                    if (heights[0]<=cutoff): # warm (W)
                        TYPE = 1
        elif len(areas)==2:
            if lowest_area>0:  # W
                TYPE = 1
            else: #C
                if areas[1]>0:  #C-W
                    TYPE = 2
                else: # C-C
                    TYPE = 0
        elif len(areas)==3:
            if lowest_area<0:
                if (areas[1]<0) & (areas[2]<0): # all cold
                    TYPE = 0
                else:
                    TYPE = 2
            else: # lowest area>0
                TYPE = 1
                if (areas[1]<0) & (areas[2]>0): # W-C-W
                # if surface layer small and far smaller than freezing layer
                # regard as type 2
                    if (areas[0]<1) & (areas[1]/areas[0]<-50):
                        TYPE = 2
    return TYPE

def first_posi_nega_area(areas, TYPE):
    if len(areas)>3:
        areas = areas[0:3]
        
    # surface is very small, ignore it
    if len(areas)==3:
        if ((areas[0]>0) & (areas[0]<1) & (areas[1]<0) & 
            (areas[1]/areas[0]<-50) & (areas[2]>0)):
            areas[0] = areas[1]
            areas[1] = areas[2]
            areas[2] = np.nan
            TYPE = 2
    
    if TYPE == 2:
        # CWW, merge two warm layer
        if (len(areas)==3):
            if(areas[1]>0) & (areas[2]>0):
                areas[1] = areas[1] + areas[2]
                areas[2] = np.nan
        
        NA = areas[0]
        PA = areas[1]
    elif TYPE == 1:
        PA = areas[0]
        NA = np.nan
    else:
        PA, NA = np.nan, np.nan
    return PA, NA, TYPE, areas

def cal_lr_fl_area_type(tt, pp, zz):
    ''' calculate lapse rate'''
    # if np.isnan(zz[0]):
    #     zz[0] = df.loc[idx, 'elev']
    if zz is None: # if no input of z, calculate from t and p
        zz = level_height(tt, pp)
        zz = np.array(zz)
    else:
        zz = fill_gph_nan(tt, pp, zz)
        zz = [0] + list(zz[1:]-zz[0]) # minus elevation = layer thickness
        zz = np.array(zz)
        
    # gph = fill_gph_nan(tt, pp, zz)
    g0 = cal_lapse_rate(tt, pp, zz)
    
    idx = np.where(~np.isnan(tt) & ~np.isnan(pp))[0]
    tt = tt[idx]
    pp = pp[idx]
    zz = zz[idx]
    
    
    '''find freezing level'''
    heights = freezing_level_height(tt, zz)
    nmax = [len(heights) if len(heights)<=3 else 3][0]
    freezing_level = heights[0:nmax]

    '''positive and negative areas'''
    areas, posis, negas = cal_energy_area(tt, pp, zz)

    ''' sounding type'''
    TYPE =  sounding_type(areas, heights)
    
    PA, NA, TYPE, areas = first_posi_nega_area(areas, TYPE)
    return g0, freezing_level, areas, PA, NA, TYPE
    
    
# def sounding_type(areas, heights):
#     '''
#     From top to bottom, below the highest freezing level, 
#     '''
#     areas = np.array(areas)
    
#     if (len(areas)==0) | (sum(areas>0)==0):
#         TYPE = 0
#         return TYPE
    
#     # the highest melting energy
#     lowest_area = areas[0]
#     idx = max(np.where(areas>0)[0])
    
#     if idx==0: # typical type 1
#         if heights[0] <= 2000:
#             TYPE = 1
#         else:
#             TYPE = 0    
#     elif idx==1: # typical type 2 sounding
#         if lowest_area>=0:
#             TYPE = 1
#         elif lowest_area<0:
#             TYPE=2
#     else: # layers >= 3, usually regard as type 2 except when all positive
    
#         lower_levels_all_positive = sum(areas[:idx]>0)==idx
#         if lower_levels_all_positive:
#             TYPE = 1
#         else:
#             if lowest_area>=0:
#                 TYPE = 1
#                 if (lowest_area<1) & (areas[1]<0) & (abs(areas[1]/lowest_area)>50):
#                     TYPE= 2
#             else:
#                 TYPE = 2
#     return TYPE     

