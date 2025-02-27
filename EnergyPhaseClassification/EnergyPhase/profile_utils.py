#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:57:25 2025

@author: sshi28
"""

'''
Package to process temperature, pressure, geopotential height profiles
Including nan checkes, interpolations based on hydrostatic balance,
compute or truncate t and p at cutoff height


'''


import numpy as np


# Example global constants (adjust these as needed)
Rd = 287.0       # Gas constant for dry air in J/(kg*K)
g = 9.81         # Gravitational acceleration in m/s^2
C2K = 273.15     # Celsius to Kelvin conversion constant


def fill_gph_nan(tt: np.ndarray, pp: np.ndarray, zz: np.ndarray = None) -> np.ndarray:
    """
    Fill missing geopotential heights using hydrostatic balance interpolation.
    
    The idea is that if a value is missing at level i (i.e. gph[i] is NaN)
    and both temperature and pressure are available at that level,
    then we interpolate using the nearest valid value below (or, if not 
    available, above) based on the hydrostatic balance:
    
        dz = (Rd/g) * (((T_current + T_reference)/2 + C2K) * (ln(p_reference) - ln(p_current)))
    
    Parameters:
        tt : np.ndarray
            Temperature profile in degrees Celsius.
        pp : np.ndarray
            Pressure profile in mb.
        zz : Optional[np.ndarray]
            Geopotential heights (in meters) with NaNs for missing data.
            If not provided or if all values are NaN, the surface (index 0) is set to 0.
    
    Returns:
        np.ndarray:
            Geopotential height profile with missing values filled.
    """
    # If no input heights provided, or if all values are NaN, initialize with NaNs and set surface = 0.
    if zz is None:
        zz = np.full_like(tt, np.nan, dtype=np.float64)
        zz[0] = 0.0
    else:
        zz = np.array(zz, dtype=np.float64)
        if np.all(np.isnan(zz)):
            zz[0] = 0.0

    # Work on a copy so that the original array is not modified.
    gph = np.copy(zz)
    
    # Loop over each vertical level.
    for i in range(len(tt)):
        # Process only if current geopotential height is missing but temperature and pressure are valid.
        if np.isnan(gph[i]) and not (np.isnan(tt[i]) or np.isnan(pp[i])):
            # Find the most recent valid index below i.
            valid_lower = np.where(~np.isnan(tt[:i]) & ~np.isnan(pp[:i]) & ~np.isnan(gph[:i]))[0]
            # Find the first valid index above i.
            valid_upper = np.where(~np.isnan(tt[i+1:]) & ~np.isnan(pp[i+1:]) & ~np.isnan(gph[i+1:]))[0]
            if valid_upper.size > 0:
                valid_upper += (i + 1)
            
            if valid_lower.size > 0:
                low_idx = valid_lower[-1]
                T_low = tt[low_idx]
                P_low = pp[low_idx]
                Z_low = gph[low_idx]
                # Compute the vertical difference based on hydrostatic balance.
                dz = (Rd / g) * (((tt[i] + T_low) / 2 + C2K) * (np.log(P_low) - np.log(pp[i])))
                gph[i] = Z_low + dz
            elif valid_upper.size > 0:
                up_idx = valid_upper[0]
                T_up = tt[up_idx]
                P_up = pp[up_idx]
                Z_up = gph[up_idx]
                dz = (Rd / g) * (((tt[i] + T_up) / 2 + C2K) * (np.log(pp[i]) - np.log(P_up)))
                gph[i] = Z_up - dz
            # If neither a lower nor an upper valid index exists, gph[i] remains NaN.
    
    return gph




def find_cutoff_index(zz, cutoff):
    """
    Find the index where the profile crosses the cutoff height.

    Parameters:
    zz : np.ndarray
        Geopotential height profile in meters.
    cutoff : float
        Cutoff height in meters.

    Returns:
    int or None
        Index of the profile crossing the cutoff height, or None if no crossing is found.
    """
    idx_cutoff = np.where((zz[:-1] < cutoff) & (zz[1:] >= cutoff))[0]
    return np.atleast_1d(idx_cutoff)[0] if len(idx_cutoff) > 0 else None




def t_p_at_cutoff_height(tt, pp, zz, idx_cutoff, cutoff=2000):
    """
    Calculate temperature and pressure at a specified cutoff height
    using hydrostatic balance and linear interpolation.

    Parameters:
    tt : np.ndarray
        Temperature profile in degrees Celsius.
    pp : np.ndarray
        Pressure profile in mb.
    zz : np.ndarray
        Geopotential height profile in meters.
    cutoff : float, optional
        Cutoff height in meters (default: 2000 m).
    Rd : float, optional
        Specific gas constant for dry air (default: 287 J/(kg·K)).
    g : float, optional
        Gravitational acceleration (default: 9.8 m/s²).

    Returns:
    tuple
        Temperature (°C) and pressure (mb) at the cutoff height.
        Returns (np.nan, np.nan) if the cutoff height is out of bounds.
    """
    # Find index where cutoff height lies between zz[i] and zz[i+1]
    idx_cutoff = np.where((zz[0:-1]<cutoff) & (zz[1:]>=cutoff))[0]
    
    # Handle case where no valid idx_cutoff is found
    if len(idx_cutoff) == 0:
        return np.nan, np.nan

    # Use the first valid index if there are multiple
    idx_cutoff = idx_cutoff[0]

    # Handle case where idx_cutoff is the last index
    if idx_cutoff == len(zz) - 1:
        return np.nan, np.nan
    
    z1 = zz[idx_cutoff]
    p1, p2 = pp[idx_cutoff], pp[idx_cutoff + 1]
    t1, t2 = tt[idx_cutoff] + C2K, tt[idx_cutoff + 1] + C2K  # Convert to Kelvin

    # Mean temperature for hydrostatic equation
    t_mean = (t1 + t2) / 2
    p_cutoff = p1 / np.exp(g / (Rd * t_mean) * (cutoff - z1))

    # Log-linear interpolation for temperature
    t_cutoff = np.log(p_cutoff / p1) / np.log(p2 / p1) * (t2 - t1) + t1
    t_cutoff -= C2K  # Convert back to Celsius

    return t_cutoff, p_cutoff


def truncate_profile_at_height(tt, pp, zz, cutoff=2000):
    """
    Adjust the temperature, pressure, and height profiles by interpolating
    at the specified cutoff height.

    Parameters:
    tt : np.ndarray
        Temperature profile in degrees Celsius.
    pp : np.ndarray
        Pressure profile in mb.
    zz : np.ndarray
        Geopotential height profile in meters.
    cutoff : float, optional
        Cutoff height in meters (default: 2000 m).

    Returns:
    tuple
        Updated tt, pp, zz arrays, including interpolated values at the cutoff height.
    """
    # Find the index where cutoff height lies between zz[i] and zz[i+1]
    idx_cutoff = find_cutoff_index(zz, cutoff)

    # If no cutoff height is found, return the original profiles
    if idx_cutoff is None:
        return tt, pp, zz

    # Use the first valid index
    idx_cutoff = np.atleast_1d(idx_cutoff)[0]

    # Interpolate temperature and pressure at the cutoff height
    t_cutoff, p_cutoff = t_p_at_cutoff_height(tt, pp, zz, cutoff=cutoff)

    # Update the profiles with interpolated values
    tt = np.append(tt[:idx_cutoff + 1], t_cutoff)
    pp = np.append(pp[:idx_cutoff + 1], p_cutoff)
    zz = np.append(zz[:idx_cutoff + 1], cutoff)

    return tt, pp, zz


# def preprocess_inputs(tt, pp, zz=None):
#     """
#     Validate and preprocess input temperature, pressure, and height profiles.

#     Parameters:
#     tt : array-like
#         Temperature profile in degrees Celsius.
#     pp : array-like
#         Pressure profile in mb.
#     zz : array-like, optional
#         Geopotential height profile in meters. If None, heights are calculated.

#     Returns:
#     tuple
#         - tt: Cleaned temperature profile as a numpy array.
#         - pp: Cleaned pressure profile as a numpy array.
#         - zz: Cleaned or computed height profile as a numpy array.
#     """
#     # Convert to numpy arrays
#     tt, pp = np.array(tt), np.array(pp)

#     # Validate input lengths
#     if len(tt) != len(pp):
#         raise ValueError("Temperature and pressure arrays must have the same length.")

#     # Remove NaNs from temperature and pressure
#     valid = ~np.isnan(tt) & ~np.isnan(pp)
#     tt, pp = tt[valid], pp[valid]

#     # Check for sufficient data
#     if len(tt) < 2:
#         raise ValueError("Insufficient data: temperature and pressure profiles \
#                          must have at least 2 valid points.")

#     # Process geopotential height
#     if zz is None:
#         zz = cal_height_from_surface(tt, pp)
#     else:
#         zz = np.array(zz)[valid]  # Align with valid tt and pp
#         zz = fill_gph_nan(tt, pp, zz)
#         zz = zz - zz[0]  # Normalize to surface elevation

#     return tt, pp, zz



