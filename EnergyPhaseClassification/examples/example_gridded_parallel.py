#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:33:50 2025

@author: ssynj
"""

'''
Below is an example for parallel computing
on Apple M3 Pro chip, ~400,000 loops takes around 10s
'''


import numpy as np
import xarray as xr

from EnergyPhase.energy_area import energy_area
from EnergyPhase.watervapor import TiFromSH

from joblib import Parallel, delayed

import urllib.request

# CONUS404, 1km hourly data.
url = 'https://data.rda.ucar.edu/d559000/wy1980/197910/wrf3d_d01_1979-10-01_00:00:00.nc'

filename = 'gridded_parallel_conus404.nc'
urllib.request.urlretrieve(url, filename)
print(f"Downloaded file saved as: {filename}")

file = './'+filename
ds = xr.open_dataset(file)

# (bottom_top: 20, south_north: 465, west_east: 650)
P  = ds['P'] [0, :20, 200:, 100:650]/100
TK = ds['TK'][0, :20, 200:, 100:650].values
Z  = ds['Z'] [0, :20, 200:, 100:650].values - ds['Z'][0, 0, 200:, 100:650].values # geopotential height => height from surface
QV = ds['QVAPOR'][0, :20, 200:, 100:650].values

Ti = TiFromSH(P, TK-273.15, QV)

nlev, nlat, nlon = np.shape(P)
ME, RE, TYPE = [np.full((nlat, nlon), np.nan) for _ in range(3)]


# Define a function that performs the computation for a single grid cell.
def compute_energy_at_cell(i, j):
    """
    Compute the energy quantities for the grid cell at indices (i, j).

    Returns:
        A tuple (i, j, freezing_levels, areas, me, re, type_val)
        where me, re, and type_val are the scalar outputs for that grid cell.
        freezing_levels and areas can be stored separately if needed.
    """
    tt = Ti[:, i, j]
    pp = P[:, i, j]
    zz = Z[:, i, j]
    areas, me, re, type_val,_ = energy_area(tt, pp, zz)
    return (i, j,  me, re, type_val)


# Assuming nlat and nlon are defined, and ME, RE, TYPE are preallocated arrays.
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(compute_energy_at_cell)(i, j)
    for i in range(nlat)
    for j in range(nlon)
)

# Now, unpack the results and fill the output arrays.
for (i, j, me, re, type_val) in results:
    ME[i, j] = me
    RE[i, j] = re
    TYPE[i, j] = type_val


