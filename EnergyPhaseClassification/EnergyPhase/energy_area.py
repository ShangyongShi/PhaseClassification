#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute energy area from t, p, z profiles

Major function: cal_energy_area(tt, pp, zz=None, cutoff=2000)

@author: ssynj
"""
import numpy as np
from .profile_utils import fill_gph_nan,find_cutoff_index,t_p_at_cutoff_height

#! note:::: 
    # the first value is the lowest
    
cutoff = 2000
max_layers = 3


def energy_area(tt, pp, zz=None):
    """
    Calculate lapse rate, freezing levels, energy areas, and sounding type.

    Parameters:
    tt : np.ndarray
        Temperature profile in degrees Celsius.
    pp : np.ndarray
        Pressure profile in mb.
    zz : np.ndarray, optional
        Geopotential height profile in meters. If None, it will be calculated.
   
    Returns:
    tuple
        - areas: Total energy areas for each layer.
        - PA: First positive energy area (melting).
        - NA: First negative energy area (refreezing).
        - TYPE: Sounding type classification.
        - freezing_level: List of freezing level heights (meters).
    """
    areas, ME, RE = cal_energy_area(tt, pp, zz)
    freezing_level_heights = freezing_level_height(tt, zz)
    TYPE = sounding_type(areas, freezing_level_heights)
    return areas, ME, RE, TYPE, freezing_level_heights
    


# ===============================================   
    
    
def cal_energy_area(tt, pp, zz=None):
    """
    Calculate the positive and negative energy areas based on the sounding profile.
    Not yet corrected
    
    Parameters:
    tt : np.ndarray
        Temperature profile in degrees Celsius.
    pp : np.ndarray
        Pressure profile in mb.
    zz : np.ndarray, optional
        Geopotential height profile in meters. If None, it will be calculated.
  

    Returns:
    tuple
        - areas: Total energy areas for each layer.
        - ME
        - RE
    """
    # Ensure inputs are numpy arrays and remove NaNs
    tt, pp = np.array(tt), np.array(pp)
    valid = ~np.isnan(tt) & ~np.isnan(pp)
    tt, pp = tt[valid], pp[valid]

    # Not enough data to calculate
    if len(tt) < 2:
        return np.empty(0)

    # Compute heights if not provided
    if zz is None:
        zz = fill_gph_nan(tt, pp)
    elif np.any(np.isnan(zz)):
        zz = fill_gph_nan(tt, pp, zz[valid])
    else:
        zz = zz[valid]
    zz = zz - zz[0]  # Normalize to surface elevation

    # Find the index where cutoff height lies between zz[i] and zz[i+1]
    idx_cutoff = find_cutoff_index(zz, cutoff)

    # If no cutoff height is found, return the original profiles
    if idx_cutoff is not None:
        t_cutoff, p_cutoff = t_p_at_cutoff_height(tt, pp, zz, cutoff)
        tt = np.append(tt[:idx_cutoff + 1], t_cutoff)
        pp = np.append(pp[:idx_cutoff + 1], p_cutoff)
        zz = np.append(zz[:idx_cutoff + 1], cutoff)

    # Identify layers crossing the 0°C line
    layer_indices = find_layer_idx(tt, pp)
    if idx_cutoff is not None:
        layer_indices.append(idx_cutoff + 1)

    # Calculate energy areas
    areas = []
    if len(layer_indices) > 1:
        for i in range(len(layer_indices) - 1):
            is_surface_layer = (i == 0)
            is_top_layer = (layer_indices[i+1]==idx_cutoff+1) if idx_cutoff is not None else False
            idx_bottom, idx_top = layer_indices[i], layer_indices[i + 1]

            # Compute area for each layer
            area, _ = layer_area(tt, pp, zz, idx_bottom, idx_top, is_surface_layer, is_top_layer)
            if area != 0:
                areas.append(area)
                #(positive_areas if area > 0 else negative_areas).append(area)

    # Default to NaN if no areas found
    if not areas: 
        return np.empty(0), np.array(np.nan), np.array(np.nan)

    areas = areas[:max_layers] # Limit to max_layers 


    # correct areas
    # three layers    
    if len(areas) == 3:
        # merge neighboring areas with same sign
        if areas[1]*areas[2]>0:
            areas[1] = areas[1] + areas[2]
            areas[2] = 0
            
        if areas[0]*areas[1]>0:
            areas[0] = areas[0] + areas[1]
            areas[1] = areas[2] 
            
        # special Case 1: from surfae to top: warm-cold-warm
        # Ignore very small surface melting layer and shift above two layers down
        small_surface_threshold=1
        ratio_threshold=-50
        if ((areas[0] > 0) and
            (areas[1] < 0) and 
            (areas[2] > 0)):
            # Shift areas to ignore small surface layer
            if ((areas[0] < small_surface_threshold) and 
               (areas[1]/areas[0] < ratio_threshold) ):
                areas[0], areas[1] = areas[1], areas[2]
        areas.pop(2)
    
    # C-C-C or W-W-W would become C0 or W0 after above step
    if np.any(np.array(areas)==0):           
        areas.remove(0) 
    
    if (len(areas)==2) & (areas[0]>0):
        if (areas[1]>0): # W-W, merge 
            areas[0] = areas[0] + areas[1]
        areas.pop(1) # W-C, no need to consider the cold layer
    
    if (len(areas)==1) & (areas[0]<0):
        areas.pop(0)
        
    areas = np.array(areas, dtype=float)
    
    ME = next((a for a in areas if (not np.isnan(a)) and (a > 0)), np.nan)
    RE = next((a for a in areas if (not np.isnan(a)) and (a < 0)), np.nan)

    return areas, ME, RE




# ==============================

def find_layer_idx(t, p):
    """    
    Identify indices where the temperature profile crosses 0°C,
    either from + to - or from - to +.
    ensuring the surface index (0) is included to compute the surface area.

    Parameters:
    t : np.ndarray
        Temperature profile in degrees Celsius.
    p : np.ndarray
        Pressure profile in mb.

    Returns:
    list
        Indices where temperature transitions across 0°C,
        including 0 as the first index if valid.
    """
    t = np.array(t)
    p = np.array(p)

    # Find zero-crossings
    crossing_down = np.where((t[:-1] >= 0) & (t[1:] < 0))[0]
    crossing_up = np.where((t[:-1] <= 0) & (t[1:] > 0))[0]

    # Combine indices
    idx = np.sort(np.concatenate((crossing_down, crossing_up)))

    # Remove index 0 if t[0] == 0 (only if it appears in idx)
    if len(idx) > 0 and idx[0] == 0 and t[0] == 0:
        idx = idx[1:]
        
    # Include 0 as the first index if valid (not redundant)
    if len(idx)==0 or idx[0]!=0:
        idx = np.insert(idx, 0, 0)

    # Remove indices that just touch 0°C but do not cross
    valid_indices = [0]
    for i in idx[1:]:
        if t[i] == 0 and i > 0 and i < len(t)-1:
            if t[i-1] * t[i+1] > 0:  # Same sign on both sides
                continue
        valid_indices.append(i)

    return valid_indices


def layer_area(tt, pp, zz, idx_bottom, idx_top, is_surface_layer, is_top_layer):
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
    Rd = 287  # Specific gas constant for dry air (J/kg·K)
    pp_log = np.log(pp)  # Use natural log of pressure for calculations

    # Determine bottom boundary
    if is_surface_layer:
        p_bottom = pp_log[0]
        t_bottom = tt[0]
    else:
        t1, p1 = tt[idx_bottom], pp_log[idx_bottom]
        t2, p2 = tt[idx_bottom + 1], pp_log[idx_bottom + 1]
        p_bottom = p1 + (p2 - p1) * (0 - t1) / (t2 - t1)
        t_bottom = 0

    # Determine top boundary
    if is_top_layer:
        p_top = pp_log[-1]
        t_top = tt[-1]
    else:
        t1, p1 = tt[idx_top], pp_log[idx_top]
        t2, p2 = tt[idx_top + 1], pp_log[idx_top + 1]
        p_top = p1 + (p2 - p1) * (0 - t1) / (t2 - t1)
        t_top = 0

    # Extract middle points
    p_middle = pp_log[idx_bottom+1 : idx_top+1]
    t_middle = tt[idx_bottom+1 : idx_top+1]

    # Combine all points
    ps = np.concatenate(([p_bottom], p_middle, [p_top]))
    ts = np.concatenate(([t_bottom], t_middle, [t_top]))

    # Calculate area using the trapezoidal rule
    area = -Rd * np.trapz(ts, x=ps) #area += (ts[it] + ts[it+1])/2 * (ps[it] - ps[it+1])

    return area, is_top_layer


# =================
# freezing level

def freezing_level_height(tt, gph):
    """
    Calculate the geopotential heights at freezing levels where 
    temperature crosses 0°C from + to -.

    Parameters:
    tt : np.ndarray
        Temperature profile in degrees Celsius.
    gph : np.ndarray
        Corresponding geopotential heights (height differences from the surface).
    
    Returns:
    freezing_heights: list
        Heights where temperature transitions from above to below freezing.
        Returns [np.nan] if no freezing levels are found.
    """
    # Identify freezing level indices
    uppers = freezing_level_idx(tt, gph)
    if not uppers:  # If no crossing indices are found
        return [np.nan]

    freezing_heights = []
    for idx in uppers:
        t1, z1 = tt[idx], gph[idx]
        t2, z2 = tt[idx + 1], gph[idx + 1]
        
        # Linear interpolation to find height where temperature crosses 0°C
        if t1 != t2:  # Avoid division by zero
            z_cross = z1 + (z2 - z1) * (0 - t1) / (t2 - t1)
            freezing_heights.append(z_cross)
    freezing_heights = np.array(freezing_heights[:max_layers])
    return freezing_heights


def freezing_level_idx(tt, zz):
    """
    Identify indices where the temperature profile crosses 0°C from above.

    Parameters:
    tt : np.ndarray
        Temperature profile in degrees Celsius.
    zz : np.ndarray
        Corresponding vertical axis, geopotential heights .

    Returns:
    list
        Indices where temperature crosses 0°C from above to below.
    """
    # Ensure input arrays are numpy arrays and remove NaNs
    tt = np.array(tt)
    zz = np.array(zz)
    valid = ~np.isnan(tt) & ~np.isnan(zz)
    tt, zz = tt[valid], zz[valid]

    # Identify where temperature transitions from >= 0 to < 0
    cross_down = (tt[:-1] >= 0) & (tt[1:] < 0)
    indices = np.where(cross_down)[0]

    # Filter out invalid crossings
    valid_indices = []
    for idx in indices:
        if idx == 0 and tt[idx] == 0:
            # Exclude surface level at exactly 0°C
            continue

        if tt[idx] == 0:
            # Check if the sounding does not cross zero
            if idx > 0 and tt[idx - 1] * tt[idx + 1] > 0:
                continue  # Same sign on both sides
            # Check if all values below are 0 or negative
            below = tt[:idx]
            if np.all(below == 0) or (np.all(below[below != 0] < 0)):
                continue

        valid_indices.append(idx)

    return valid_indices





    
def sounding_type(areas, heights):
    """
    Classify the sounding based on energy areas and freezing level heights.

    Parameters:
    areas : list or np.ndarray
        Energy areas of the layers (positive for melting, negative for refreezing).
    heights : list or np.ndarray
        Heights of freezing levels (meters).
    cutoff : float, optional
        Threshold height in meters for surface classification (default: 2000 m).
    max_layers : int, optional
        Maximum number of layers to consider for classification (default: 3).

    Returns:
    int
        Sounding type:
        - 0: All cold (C-C or C-C-C).
        - 1: Surface above freezing (W or W-C-W).
        - 2: Surface below freezing and aloft layer above freezing (C-W or C-W-C).
    """
    
    if len(areas) == 0:
        return 0  # All cold/warm by default

    lowest_area = areas[0]

    if len(areas) == 1:
        # One layer: Type 1 if surface above freezing, height <= cutoff
        return 1 if lowest_area > 0 and len(heights) > 0 and heights[0] <= cutoff else 0

    if len(areas) == 2:
        # Two layers: Type 0 (C-C), 1 (W), or 2 (C-W)
        if lowest_area > 0:
            return 1  # W
        return 2 if areas[1] > 0 else 0  # C-W or C-C

    # if len(areas) == 3:
    #     # Three layers: Type 0 (C-C-C), 1 (W-C-W), or 2 (C-W-C)
    #     if lowest_area < 0:
    #         return 0 if areas[1] < 0 and areas[2] < 0 else 2  # All cold or C-W-C
    #     # W-C-W with small surface layer treated as C-W
    #     if (areas[1] < 0) and areas[2] > 0:
    #         return 2 if areas[0] < 1 and areas[1] / areas[0] < -50 else 1
    #     return 1  # Default to W

    return 0  # Default to all cold/warm for extra cases



