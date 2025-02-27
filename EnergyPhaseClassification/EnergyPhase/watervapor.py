# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:30:38 2022

functions about watervapor

@author: ssynj
"""
import math
import numpy as np
from numba import njit
maxi = 100000
wv_epsln = 0.622   # Rd/Rv
wv_c2k = 273.15    # const for c to k conversion
wv_lv = 2.5E+6     # latent heat for evapoartion (J/Kg)
wv_cpd = 1005.     # specific heat w/ p=const. for dry air
wv_Rv=461.5        # water vapor gas constant     
wv_Rd=287.         # Dry gas constant     
  
wv_A = 2.53E9      # coef. in Clausius-Clapeyron eq. (mb)
wv_B = 5.42E3      # coef. in C.-C. eq. (K)

wv_SMALL = 1e-2    # error allowed
    
wv_es0 = 6.112      # vapor pressure at t=0C, hPa
wv_eswa = 17.62     # const_a in es for water
wv_eswb = 243.12    # const_b in es for water
wv_esia = 22.46     # const_a in es for ice
wv_esib = 272.62    # const_b in es for ice

wv_pscwA = 6.60E-4    # const_A in psychrometer formula for wetbulb
wv_psciA = 5.82E-4    # const_A in psychrometer formula for icebulb
wv_pscB = 0.00115     # const_B in psychrometer formula 

def SatVapPreW(tc):
    '''
    Input: tc | temperature in C
    
    Output: es | saturation vapor pressure in mb
    '''
    es = wv_es0*np.exp(wv_eswa*tc/(wv_eswb+tc))
    return es

def SatVapPre(tc):
    '''
    Input: tc | temperature in C
    
    Output: es | saturation vapor pressure in mb
    '''
    es = SatVapPreW(tc)
    return es

def SatVapPreI(tc):
    '''
    Input: tc | temperature in C
    
    Output: es | saturation vapor pressure in mb
    '''
    es = wv_es0*np.exp(wv_esia*tc/(wv_esib+tc))
    return es

def SatMixRatW(pmb, tc):
    '''
    Input: 
        pmb - pressure in mb
        tc - temperature in C

    output: 
        ws - sat. vapor mixing ratio
    '''
    es = SatVapPre(tc)
    ws = wv_epsln*es/(pmb-es)
    return ws

def SatMixRat(pmb, tc):
    '''
    Input: 
        pmb - pressure in mb
        tc - temperature in C

    output: 
        ws - sat. vapor mixing ratio
    '''
    ws = SatMixRatW(pmb,tc)
    return ws

def SatMixRatI(pmb,tc):
    '''
    Input: 
        pmb - pressure in mb
        tc - temperature in C

    output: 
        ws - sat. vapor mixing ratio
    '''
    es=SatVapPreI(tc)
    ws=wv_epsln*es/(pmb-es)
    return ws

# calculate RH
def td2e(td):
    '''
    input: 
        td - dew point temperature in C
        
    output: 
        e - vapor pressure in mb'''
    e = SatVapPre(td)
    return e

# calculate RH
def tf2e(tf):
    '''
    input: 
        tf - frost point temperature in C
        
    output: 
        e - vapor pressure in mb'''
    e = SatVapPreI(tf)
    return e

def td2rh(tc, tdc):
    '''
    Input
        tc: temperature in C
        tdc: dew point temperature in C
        
    Output:
        rh - relative humidity in %
    '''
    rh = 100 * SatVapPre(tdc)/SatVapPre(tc)
    return rh

def td2w(pmb,td):
    '''
    input: 
        pmb - pressure in mb
        td - dew point temperature in C
        
    output: 
        w - mixing ratio (unitless)'''
    w = SatMixRat(pmb,td)
    return w

def rh2e(tc,rh):
    '''
    input:  
        tc - temperature in C
        rh - rel. humidity in % to water
        
    output: 
        e - vapor pressure in mb'''
    e = 0.01*rh*SatVapPre(tc)
    return e

def rh2rou(tc, rh):
    '''
    input:  
        tc - temperature in C
        rh - rel. humidity in % to water
        
    output: 
        rou - vapor density in kg/m^3'''
    e=rh2e(tc,rh)
    rou=e*100./wv_Rv/(tc+wv_c2k) 
    return rou
        
def rh2w(pmb,tc,rh):
    '''
    input: 
        pmb - pressure in mb
        tc - temperature in C
        rh - rel. humidity in %
        
    output:  
        w - mixing ratio (unitless)'''
    e = rh2e(tc,rh)
    w = wv_epsln*e/(pmb-e)
    return w



def w2rh(pmb, w, tc):
    """
    Compute relative humidity (rh, in %) from pressure, mixing ratio, and temperature.
    
    Parameters
    ----------
    pmb : float or np.ndarray
        Pressure in mb.
    w : float or np.ndarray
        Mixing ratio (unitless).
    tc : float or np.ndarray
        Temperature in °C.
        
    Returns
    -------
    rh : float or np.ndarray
        Relative humidity in %, clipped between 0 and 100.
    
    Notes
    -----
    This function accepts both scalar and array inputs. If the input is scalar,
    a scalar is returned. If the inputs are arrays, the operation is performed elementwise.
    
    The relative humidity is computed as:
        rh = 100 * pmb * w / (wv_epsln * SatVapPre(tc))
    where `wv_epsln` is a constant and `SatVapPre(tc)` is a function that computes 
    the saturation vapor pressure for a given temperature tc.
    """
    # Ensure inputs are numpy arrays (this converts scalars to 0-d arrays)
    pmb = np.asarray(pmb, dtype=np.float64)
    w   = np.asarray(w,   dtype=np.float64)
    tc  = np.asarray(tc,  dtype=np.float64)
    
    # Compute relative humidity
    # Note: Ensure that wv_epsln is defined and SatVapPre is available in your module.
    rh = 100 * pmb * w / (wv_epsln * SatVapPre(tc))
    
    # Clip relative humidity values to be between 0 and 100%
    rh = np.clip(rh, 0, 100)
    
    # Return a scalar if the inputs were scalars
    if rh.ndim == 0:
        return rh.item()
    return rh



def ah2rh(ah, tc, pmb):
    """
    Convert absolute humidity to relative humidity.
    
    Parameters
    ----------
    ah : float or np.ndarray
        Absolute humidity in kg/m³.
    tc : float or np.ndarray
        Temperature in °C.
    pmb : float or np.ndarray
        Pressure in hPa (or mb). (Note: pmb is not used in the current formula,
        but is included for compatibility.)
    
    Returns
    -------
    rh : float or np.ndarray
        Relative humidity in %, clipped between 0 and 100.
    
    The relative humidity is computed as:
    
        rh = 100 * ah * wv_Rv * (tc + wv_c2k) * 0.01 / SatVapPre(tc)
    
    where:
      - wv_Rv is a constant 461.5,water vapor gas constant 
      - wv_c2k is the conversion from Celsius to Kelvin, 273.15
      - SatVapPre(tc) returns the saturation vapor pressure at temperature tc.
    
    The function accepts both scalar and array inputs. If the result is a 0-dimensional array,
    a Python scalar is returned.
    """
    # Ensure inputs are numpy arrays for elementwise operations.
    ah = np.asarray(ah, dtype=np.float64)
    tc = np.asarray(tc, dtype=np.float64)
    pmb = np.asarray(pmb, dtype=np.float64)
    
    # Compute relative humidity using the provided formula.
    rh = 100 * ah * wv_Rv * (tc + wv_c2k) * 0.01 / SatVapPre(tc)
    
    # Clip the results so that 0 <= rh <= 100.
    rh = np.clip(rh, 0, 100)
    
    # If the result is 0-dimensional, return a scalar.
    if rh.ndim == 0:
        return rh.item()
    return rh

def td2tw(tc, pmb, tdc):
    ''' 
    Convert dew point temperature to wet bulb temperature
    
    Inputs:
        tc: near-surface temperature, unit: C
        pmb: surface pressure, unit: mb or hPa
        tdc: near-surface dew point temperature, unit: C
        
    Output:
        twbc: near-surface wet-bulb temperature, unit: C
    '''

    A = 2.53e9;
    B = 5420;
    KC = 273.15;
    AP = 0.00062;
    SMALL = 0.01;
    
    if(tdc >= tc):
        twbc = tc;
        return twbc
    
    tk = tc + KC;
    tdk = tdc + KC;
    e = A* math.exp(-B/tdk);
    
    
    twk1 = tk;
    wk1 = 1000.;
    i = 1;
    while(i):
        esw = A*math.exp(-B/twk1);
        wk2 = abs(e-esw+AP*pmb*(tk-twk1));
        if(wk2 < wk1):
            wk1  = wk2;
            twk2 = twk1;
        twk1 = twk1 - SMALL;
        if(twk1 < tdk):
            i = 0;
        
    twbc = twk2 - KC;
    return twbc

def rh2tw(pmb, tc, rh):
    '''
    Input:
        pmb - pressure in hPa or mb
        tc - temperature in C
        rh - rel. humidity in %
        
    Output:
        tw - wet bulb temperature in C
        '''
    e=rh2e(tc,rh)
    A=0
    tw=TwiByNewtonIteration(pmb,tc,e,A)
    return tw

def rh2ti(pmb, tc, rh):
    '''
    Input:
        pmb - pressure in hPa or mb
        tc - temperature in C
        rh - rel. humidity in %
        
    Output:
        ti - ice bulb temperature in C
        '''
    e=rh2e(tc,rh)
    A=1
    ti =TwiByNewtonIteration(pmb,tc,e,A)
    return ti


def TiFromRH(pmb, tc, rh):
    '''
    Compute the Ti (ice-bulb temperature) over arrays pmb, tc, and relative humidity.
    These can be 2D, 3D, or higher as long as they broadcast together.
    
    
    Input: 
        pmb - 
        tc
        rh:  0-100
        
    Output: 
        ti: ice-bulb temperature in C
    '''
    e = 0.01*rh*SatVapPre(tc)
    ti = TiByNewtonIteration(pmb, tc, e)
    return ti

def TiFromSH(pmb, tc, q):
    '''
    Compute the Ti (ice-bulb temperature) over arrays pmb, tc, and specific humidity.
    These can be 2D, 3D, or higher as long as they broadcast together.
    
    specific humidity q ~ mixing ratio w
    
    Input: 
        pmb - 
        tc
        rh:  0-100
        
    Output: 
        ti: ice-bulb temperature in C
    '''
    rh = w2rh(pmb, q, tc)
    ti = TiFromRH(pmb, tc, rh)
    return ti
    

    
@njit
def TiByNewtonIteration(pmb, tc, e, maxi=100000, tol=wv_SMALL):
    a = wv_esia
    b = wv_esib
    c = wv_psciA
    # Flatten all inputs to 1D.
    flat_tc = tc.ravel()
    flat_pmb = pmb.ravel()
    flat_e = e.ravel()
    # Start with the initial guess.
    flat_ti = flat_tc.copy()
    n = flat_ti.shape[0]
    
    for iter in range(maxi):
        # Compute exp_term, f, and fprime elementwise on the flat array.
        exp_term = np.empty(n, dtype=np.float64)
        f = np.empty(n, dtype=np.float64)
        fprime = np.empty(n, dtype=np.float64)
        for idx in range(n):
            exp_term[idx] = np.exp(a * flat_ti[idx] / (b + flat_ti[idx]))
            f[idx] = flat_e[idx] - wv_es0 * exp_term[idx] + c * (1 + wv_pscB * flat_ti[idx]) * flat_pmb[idx] * (flat_tc[idx] - flat_ti[idx])
            fprime[idx] = -a * b * wv_es0 / (b + flat_ti[idx])**2 * exp_term[idx] - c * flat_pmb[idx] * (1 + 2 * wv_pscB * flat_ti[idx] - wv_pscB * flat_tc[idx])
        
        flat_ti_new = flat_ti - f / fprime
        
        # Check convergence on the flattened array.
        diff = 0.0
        for idx in range(n):
            diff = max(diff, np.abs(flat_ti_new[idx] - flat_ti[idx]))
        if diff < tol:
            return flat_ti_new.reshape(tc.shape)
        
        # Update the entire flat array.
        for idx in range(n):
            if np.abs(flat_ti_new[idx] - flat_ti[idx]) >= tol:
                flat_ti[idx] = flat_ti_new[idx]
    
    raise RuntimeError("Newton Iteration failed to converge for some grid cells")




def TwFromRH(pmb, tc, rh):
    '''
    Compute the Ti (ice-bulb temperature) over arrays pmb, tc, and relative humidity.
    These can be 2D, 3D, or higher as long as they broadcast together.
    
    
    Input: 
        pmb - 
        tc
        rh:  0-100
        
    Output: 
        ti: ice-bulb temperature in C
    '''
    e = 0.01*rh*SatVapPre(tc)
    tw = TwByNewtonIteration(pmb, tc, e)
    return tw

def TwFromSH(pmb, tc, q):
    '''
    Compute the Ti (ice-bulb temperature) over arrays pmb, tc, and specific humidity.
    These can be 2D, 3D, or higher as long as they broadcast together.
    
    specific humidity q ~ mixing ratio w
    
    Input: 
        pmb - 
        tc
        rh:  0-100
        
    Output: 
        ti: ice-bulb temperature in C
    '''
    # Convert inputs to numpy arrays
    tc = np.asarray(tc)
    pmb = np.asarray(pmb)
    q = np.asarray(q)
    
    rh = w2rh(pmb, q, tc)
    tw = TwFromRH(pmb, tc, rh)
    return tw
    

    
@njit
def TwByNewtonIteration(pmb, tc, e, maxi=100000, tol=wv_SMALL):
    
    tc = np.asarray(tc)
    pmb = np.asarray(pmb)
    e = np.asarray(e)
    
    a = wv_eswa
    b = wv_eswb
    c = wv_pscwA
    # Flatten all inputs to 1D.
    flat_tc = tc.ravel()
    flat_pmb = pmb.ravel()
    flat_e = e.ravel()
    # Start with the initial guess.
    flat_tw = flat_tc.copy()
    n = flat_tw.shape[0]
    
    for iter in range(maxi):
        # Compute exp_term, f, and fprime elementwise on the flat array.
        exp_term = np.empty(n, dtype=np.float64)
        f = np.empty(n, dtype=np.float64)
        fprime = np.empty(n, dtype=np.float64)
        for idx in range(n):
            exp_term[idx] = np.exp(a * flat_tw[idx] / (b + flat_tw[idx]))
            f[idx] = flat_e[idx] - wv_es0 * exp_term[idx] + c * (1 + wv_pscB * flat_tw[idx]) * flat_pmb[idx] * (flat_tc[idx] - flat_tw[idx])
            fprime[idx] = -a * b * wv_es0 / (b + flat_tw[idx])**2 * exp_term[idx] - c * flat_pmb[idx] * (1 + 2 * wv_pscB * flat_tw[idx] - wv_pscB * flat_tc[idx])
        
        flat_tw_new = flat_tw - f / fprime
        
        # Check convergence on the flattened array.
        diff = 0.0
        for idx in range(n):
            diff = max(diff, np.abs(flat_tw_new[idx] - flat_tw[idx]))
        if diff < tol:
            return flat_tw_new.reshape(tc.shape)
        
        # Update the entire flat array.
        for idx in range(n):
            if np.abs(flat_tw_new[idx] - flat_tw[idx]) >= tol:
                flat_tw[idx] = flat_tw_new[idx]
    
    raise RuntimeError("Newton Iteration failed to converge for some grid cells")

# import sys

# def TwiByNewtonIteration(pmb, tc, e, wori):
#     '''
#     compute wet-bulb or ice-bulb temperature for scalar inputs
#     Input: 
#         pmb - 
#         tc
#         e
#         wori: 0 for tw (wet-bulb), 1 for ti (ice-bulb)
        
#     Output: 
#         tw (wet-bulb) or ti (ice-bulb) in C
#     ''' 
#     if(wori==0):   # for wet-bulb
#         a=wv_eswa
#         b=wv_eswb
#         c=wv_pscwA
#     else:				#for ice-bulb
#         a=wv_esia
#         b=wv_esib
#         c=wv_psciA
    
    
#     #f(x) = e-es(x)+pscwA*(1+pscwB*x)*p*(tc-x)
#     #f'(x) = -a*b*es0/(b+x)^2*exp(ax/(b+x))-c*p*(1+2Bx-Btc)
    
#     twOld = tc            #first guess is twOld = t
#     i = 0
#     # Newton's Iteration Method:
#     while i<maxi:
#         f = e-wv_es0*np.exp(a*twOld/(b+twOld))+c*(1+wv_pscB*twOld)*pmb*(tc-twOld)
#         fprime = -a*b*wv_es0/(b+twOld)**2*np.exp(a*twOld/(b+twOld)) \
#                -c*pmb*(1+2*wv_pscB*twOld-wv_pscB*tc)
#         twNew = twOld - f/fprime
#         if(abs(twNew-twOld) > wv_SMALL):
#             twOld = twNew
#             i = i+1
#         else:
#             tw = twNew # Celsius
#             break
        
#     if i==maxi:
#         print ("Newton Iteration failed after ",maxi," loops.")
#         sys.exit()
#     return tw
