#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:25:20 2023

@author: sshi
"""

def slp2sp(slp, elev, tc):
    '''
    Based on hydrostatic balance, calculate station pressure at certain 
    elevation using the sea level pressure and near-surface temperature
        dp = -rho*g*dz, rho = p/RT 
        => p = p0*exp[(z0-z)*g/R/T]
    Input:
        slp: sea level pressure, unit: hPa
        elev: station elevation, unit: m
        tc: near-surface temperature, unit: Kelvin
    Output:
        sp: station pressure at elevation elev, unit: hPa
    '''
    import numpy as np
    R = 287;
    g = 9.8;
    tk = tc + 273.15
    sp = slp * np.exp(-elev * g / R / tk)
    return sp