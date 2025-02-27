# Installation


Unzip the zipfile under <workdir>. 

Enter the path <workdir>/EnergyPhaseClassification_v2 and run:

```pip install .```

The necessary packages in setup.py would be installed. The key packages are numpy and pandas. xarray are used in examples, while joblib is used for parallel computation for large data.

# Package overview---

/EnergyPhase:
 |
 -- energy_area.py: 

module to compute the energy area given inputs of t(C), P(mb) and Z(m).    

    key function: energy_area(t, p, z), returns the energies from low to top (areas), the lowest melting energy and refreezing energy, the profile type (0 for all warm/all cold, 1 for melting layer near surface, 2 for intersions with near surface freezing layer and a melting layer aloft), and freezing level heights (heights where the temperature equals 0C)
    functions deal with array inputs.
    
    areas, ME, RE, TYPE, freezing_level_heights = energy_area(t, p, z)
 |
 |
 -- lapse_rate.py:

 module to compute lowest 500m lapse rate. 

    key funtion: lapse_rate(t, p, z, target_height=500, min_dz=250, max_dz=750)

 |
 -- snowprob.py 

module to output the snow conditional probability/snow fraction given inputs of near surface Tw, Ti, and the ME, RE, TYPE computed from energy_area.py. The coefficients and look up tables (LUTs) are for ice-bulb temperature based results from the paper, which is the default method to use.
    key functions:

    snowprob_func
    Use the fitted functions to estimate snowprob.
    
    snowprob_LUT
    Pre generated LUT. Modified from fortran code.
    Problem: the values did not decrease to 0. Will improve later
 |
 -- snowprob_tw.py: 

similar to snowprob.py. the only difference is that the coefficients and LUTs are from wet-bulb temperature based results. 

 |
 -- profile_utils.py: module to assist energy computation
 |
 -- watervapor.py: module for water related computations.
 |
 -- basemap.py: to plot basemap for spatial figures.



# Use EnergyPhase in python

#### for computation
see /examples


#### for classification

```python
from EnergyPhase.snowprob import *
from EnergyPhase.watervapor import *

# read in P (hPa), TC (degC), QV(kg/kg)

Tw = TwFromSH(P, TC, QV)
Ti = TiFromSH(P, TC, QV)
snowp = snowprob_func(Tw, Ti, ME, RE, TYPE)
snowp = snowprob_LUT(Tw, Ti, ME, RE, TYPE)
```

