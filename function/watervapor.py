# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:30:38 2022

functions about watervapor

@author: ssynj
"""
import math
import numpy as np
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
    '''
    input: 
        pmb - pressure in mb
        w - mixing ratio (unitless)
        tc - temperature in C
        
    output:  
        rh - rel. humidity in %'''
    rh = 100*pmb*w/wv_epsln/SatVapPre(tc)
    return rh

def ah2rh(ah,tc, pmb):
    '''
    Input:
        ah - absolute humidity in kg/m3
        tc - temperature in C
        pmb - pressure in hPa or mb
        
    Output:
        rh - rel. humidity in %'''
    rh = 100*ah*wv_Rv*(tc+wv_c2k)*0.01/SatVapPre(tc)
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

import sys
def TwiByNewtonIteration(pmb, tc, e, wori):
    '''
    Input: 
        pmb - 
        tc
        e
        wori: 0 for tw (wet-bulb), 1 for ti (ice-bulb)
        
    Output: 
        tw (wet-bulb) or ti (ice-bulb) in C
    '''
    maxi=100000
   
    
    if(wori==0):   # for wet-bulb
        a=wv_eswa
        b=wv_eswb
        c=wv_pscwA
    else:				#for ice-bulb
        a=wv_esia
        b=wv_esib
        c=wv_psciA
    
    
    #f(x) = e-es(x)+pscwA*(1+pscwB*x)*p*(tc-x)
    #f'(x) = -a*b*es0/(b+x)^2*exp(ax/(b+x))-c*p*(1+2Bx-Btc)
    
    twOld = tc            #first guess is twOld = t
    i = 0
    # Newton's Iteration Method:
    while i<maxi:
        f = e-wv_es0*np.exp(a*twOld/(b+twOld))+c*(1+wv_pscB*twOld)*pmb*(tc-twOld)
        fprime = -a*b*wv_es0/(b+twOld)**2*np.exp(a*twOld/(b+twOld)) \
               -c*pmb*(1+2*wv_pscB*twOld-wv_pscB*tc)
        twNew = twOld - f/fprime
        if(abs(twNew-twOld) > wv_SMALL):
            twOld = twNew
            i = i+1
        else:
            tw = twNew # Celsius
            break
        
    if i==maxi:
        print ("Newton Iteration failed after ",maxi," loops.")
        sys.exit()
    return tw



    
    # wv_epsln = 0.622
    # wv_c2k = 273.15
    # wv_lv = 2.5E+6
    # wv_cpd = 1005.
    # wv_A = 2.53E9
    # wv_B = 5.42E3
    # wv_SMALL = 1e-2
    
    # twOld = tc + wv_c2k
    # i = 0
    # # Newton's Iteration Method
    # ww = 1. / (pmb*np.exp(wv_B/(tc+wv_c2k))-wv_A) *0.01*rh
    # f = tc + wv_c2k-twOld-wv_lv/wv_cpd*wv_epsln*wv_A*(1./(pmb*np.exp(wv_B/twOld)-wv_A)-ww)
    # fprime=-1-wv_lv/wv_cpd*wv_epsln*wv_A*pmb*wv_B*np.exp(-wv_B/twOld)/twOld**2/  \
    #               (pmb-wv_A*np.exp(-wv_B/twOld))**2
    # twNew = twOld-f/fprime
    # while (abs(twNew-twOld) > wv_SMALL):
    #     twOld = twNew
    #     f = tc + wv_c2k-twOld-wv_lv/wv_cpd*wv_epsln*wv_A* \
    #                 (1./(pmb*np.exp(wv_B/twOld)-wv_A)-ww)
    #     fprime=-1 - wv_lv/wv_cpd*wv_epsln*wv_A*pmb*wv_B*np.exp(-wv_B/twOld) \
    #                 / twOld**2 / (pmb-wv_A*np.exp(-wv_B/twOld))**2
    #     twNew = twOld-f/fprime
    #     i = i+1
    #     if (i> maxi):
    #         print ("Newton Iteration failed after ",maxi," loops.")
    #         exit()
    # twbc = twNew - wv_c2k
    # return twbc
