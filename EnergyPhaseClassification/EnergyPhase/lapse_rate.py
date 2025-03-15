#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute low level lapse rate

@author: ssynj
"""
import numpy as np

from .profile_utils import fill_gph_nan


def lapse_rate(t, p, z, target_height=500, min_dz=250, max_dz=750):
    """
    Calculate the low-level lapse rate over a target height (default: 500m)
    using temperature and geopotential height profiles.

    The lower level is the pressure level just above the surface pressure.
    The upper level is the level higher than the lower level 
    with a geopotential difference closest to 500m.
    
    Parameters:
    t : np.ndarray
        Temperature profile in degrees Celsius.
    p : np.ndarray
        Pressure profile in hPa.
    z : np.ndarray
        Geopotential height profile in meters.
    target_height : float, optional
        Target height difference in meters (default: 500 m).
    min_dz : float, optional
        Minimum acceptable height difference for calculation (default: 250 m).
    max_dz : float, optional
        Maximum acceptable height difference for calculation (default: 750 m).

    Returns:
    float
        Lapse rate in °C/km. Returns NaN if calculation is not possible.
    """


    # Remove NaN values
    valid = ~np.isnan(t) & ~np.isnan(z)
    t, z = t[valid], z[valid]

    # Ensure there is enough data
    if len(t) < 2:
        return np.nan

    # Find the index closest to the target height difference
    target_idx = np.argmin(abs(z - z[0] - target_height))

    dz = z[target_idx] - z[0]
    dt = t[target_idx] - t[0]

    # Check if the height difference is within acceptable limits
    if not (min_dz <= dz <= max_dz):
        return np.nan

    # Calculate and return the lapse rate
    return -dt / dz * 1000  # Convert to °C/km



def lapse_rate_vec(t: np.ndarray, p: np.ndarray, z: np.ndarray,
                              target_height: float = 500, 
                              min_dz: float = 250, 
                              max_dz: float = 750) -> np.ndarray:
    """
    Calculate the low-level lapse rate (in °C/km) in a fully vectorized manner.
    
    Parameters
    ----------
    t : np.ndarray
        Temperature profile in °C with shape (nlev, ...) where ... represents one
        or more horizontal dimensions (e.g. (nlev, nlat, nlon)).
    p : np.ndarray
        Pressure profile in hPa with the same shape as t (unused in the current
        computation, but kept for compatibility).
    z : np.ndarray
        Geopotential height profile in meters with the same shape as t.
    target_height : float, optional
        Target height difference from the surface in meters (default: 500 m).
    min_dz : float, optional
        Minimum acceptable vertical difference in meters (default: 250 m).
    max_dz : float, optional
        Maximum acceptable vertical difference in meters (default: 750 m).
    
    Returns
    -------
    lapse_rate : np.ndarray
        Array of lapse rates (in °C/km) with shape matching the horizontal dimensions.
        If the vertical difference is outside [min_dz, max_dz] for a profile, the
        corresponding lapse rate is set to NaN.
    
    Notes
    -----
    This function assumes that there are no NaNs in the input profiles. If some levels
    are missing, you should preprocess the data (for example, by interpolating or masking)
    before calling this function.
    """
    # Ensure inputs are numpy arrays.
    t = np.asarray(t, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    # p is not used here, but if needed, convert it as well.
    
    # Surface values (assumed at index 0) for each profile.
    t0 = t[0]  # shape: horizontal dims
    z0 = z[0]  # shape: horizontal dims
    
    # Compute the absolute difference from the target height for each level.
    # This works because z and z0 broadcast along axis 0.
    diff = np.abs(z - z0[np.newaxis, ...] - target_height)
    
    # Find, for each profile, the index along the vertical axis where the difference is minimized.
    # target_idx has shape equal to the horizontal dimensions.
    target_idx = np.argmin(diff, axis=0)
    
    # Use np.take_along_axis to extract the temperature and height at the target index.
    # We need to add a new axis to target_idx to use it along axis 0.
    t_target = np.take_along_axis(t, target_idx[np.newaxis, ...], axis=0)[0]
    z_target = np.take_along_axis(z, target_idx[np.newaxis, ...], axis=0)[0]
    
    # Compute the differences from the surface.
    dt = t_target - t0
    dz = z_target - z0
    
    # Calculate lapse rate in °C/km.
    lapse_rate = -dt / dz * 1000.0
    
    # Apply the vertical difference limits: if dz is not within [min_dz, max_dz], set lapse_rate to NaN.
    valid = (dz >= min_dz) & (dz <= max_dz)
    lapse_rate = np.where(valid, lapse_rate, np.nan)
    
    return lapse_rate