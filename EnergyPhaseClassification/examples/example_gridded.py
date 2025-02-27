#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 23:19:52 2025

@author: ssynj
"""


import pandas as pd
import numpy as np
import xarray as xr

from EnergyPhase.energy_area import energy_area
from EnergyPhase.watervapor import TiFromRH
import datetime

import urllib.request

# CONUS404, 1km hourly data.
url = 'https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NPASM.5.12.4/1980/01/MERRA2_100.inst3_3d_asm_Np.19800101.nc4'

filename = 'gridded_parallel_merra2.nc'
urllib.request.urlretrieve(url, filename)
print(f"Downloaded file saved as: {filename}")

file = './'+filename
ds = xr.open_dataset(file)
# only need around 15 levels from surface
ds_conus = ds.sel(lon=slice(-125.0, -50), lat=slice(25, 49), lev=slice(57,72))

'''
tzyx

H: midlayer heights, m
PL: midlevel pressure, Pa
QV: specific humidity, kg/kg
RH: relative humidity after moist, 1
T: air temperature, K
'''
H = ds_conus['H'].values
PL = ds_conus['PL'].values/100
QV = ds_conus['QV'].values
RH = ds_conus['RH'].values *100 # 0-100
T = ds_conus['T'].values -273.15


Ti = TiFromRH(PL, T, RH)

ntime, nlev, nlat, nlon = np.shape(Ti)
ME, RE, TYPE = [np.full((ntime, nlat, nlon), np.nan) for _ in range(3)]


for i in range(ntime):
    for j in range(nlat):
        for k in range(nlon): #650 100-0.01s
            tt = Ti[i, :, j, k]
            pp = PL[i, :, j, k]
            zz = H[i, :, j, k]
            areas, ME[i, j, k], RE[i, j, k], TYPE[i, j, k], freezing_level_heights = energy_area(tt, pp, zz)
            
            
            
