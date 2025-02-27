#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 12:43:06 2024

test

@author: ssynj
"""

import numpy as np
import pandas as pd

# IGRA data
# created from IGRA sounding. Ti is computed using p, t and rh measurements.
file = './singlepoint_70273_USM00070273_20181111120000.txt'
df = pd.read_csv(file)

tt = df.t.values
pp = df.p.values
zz = df.z.values

from EnergyPhase.energy_area import energy_area
help(energy_area)

areas, ME, RE, TYPE, freezing_level_heights = energy_area(tt, pp, zz)
print(areas, ME, RE, TYPE)


# test interpolation
tt[2] = np.nan
pp[4] = np.nan
zz[3] = np.nan
areas, ME, RE, TYPE, freezing_level_heights = energy_area(tt, pp, zz)
print(areas, ME, RE, TYPE)




