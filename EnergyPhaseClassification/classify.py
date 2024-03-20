#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:35:34 2024

@author: Shangyong Shi
"""
import pandas as pd
import numpy as np

def classify(test, threshold, method):
    '''
    Input:
        test: DataFrame, with columns of wet-bulb temperature, ice-bulb temperature
                melting and refreezing energy, and sounding type.
            test.columns=['tw', 'ti', 'me', 're', 'type']
        threshold: threshold for conditional probability of snow, 
            available values: 0.3, 0.4, ..., 0.8
        method: 1 for Ti and 2 for Tw

    Output:
        rain: DataFrame with rows predicted as rain
        snow: DataFrame with rows predicted as snow
    '''
    test0 = test[ test['type']==0]
    test1 = test[ test['type']==1]
    test2 = test[ test['type']==2]
    
    pre_rain0, pre_snow0 = classify_type0(test0)
    pre_rain1, pre_snow1 = classify_type1(test1, threshold, method)
    pre_rain2, pre_snow2 = classify_type2(test2, threshold, method)
    
    pre_rain = pd.concat([pre_rain0, pre_rain1, pre_rain2])
    pre_snow = pd.concat([pre_snow0, pre_snow1, pre_snow2])   
    return pre_rain, pre_snow

def classify_type0(test):
    snow = test[test['tw']<=1.6]
    rain = test[test['tw']>1.6]
    return rain, snow

def type1_exp(x, m, t):
    return m * np.exp(t*x ) 

def classify_type1(test, threshold, method):
    '''
    Seperation for soundings with only one melting at the bottom
    '''
    if method==1:
        # Ti
        coefs = {
                0.3: np.array([1.68332365, -0.1811878 ]),
                0.4: np.array([1.42235443, -0.22139454]),
                0.5: np.array([1.19237535, -0.29651954]),
                0.6: np.array([0.93126144, -0.42526325]),
                0.7: np.array([0.92178506, -1.31780889]),
                0.8: np.array([0.4477541 , -2.14061253])}
    elif method==2:
        coefs = {0.3: np.array([1.8352, -0.1688]),
                0.4: np.array([1.5738, -0.2053]),
                0.5: np.array([1.2957, -0.2748]),
                0.6: np.array([1.0060, -0.3635]),
                0.7: np.array([0.7358, -0.3635]),
                0.8: np.array([0.4, -0.4])
                }
    else:
        print('Input method=1 for Ti scheme, method=2 for Tw scheme')

    popt = coefs[threshold]
    pre_snow = test[test['ti'] <= type1_exp(test['me'], *popt)]
    pre_rain = test[test['ti'] >  type1_exp(test['me'], *popt)]
    return pre_rain, pre_snow




def classify_type2(test, threshold, method):
    '''
    Separation for soundings with a melting layer and a refreezing layer
    '''
    if method==1:
        coefs = {0.3: np.array([ 0.11992169, -0.48402358,  7.40607351]),
                0.4: np.array([ 0.08766819,  0.0641349 , -3.59167715]),
                0.5: np.array([  0.14446052,   0.63535494, -14.0823996 ]),
                0.6: np.array([  0.21317597,   0.69167411, -16.86781474]),
                0.7: np.array([  0.29617753,   0.47365385, -18.36315865]),
                0.8: np.array([  0.39996308,  -0.11066325, -19.82783235])}
        def type2_tanh(x,  b, c, d):
            return -18*(np.tanh(b*x -c))+d
    elif method==2:
        coefs = {0.3: np.array([0.3216, -0.1059, 0.4968]),
                0.4: np.array([0.2770, -0.2366, -0.8472]),
                0.5: np.array([0.2819, -0.0287, -3.4724]),
                0.6: np.array([0.3112, -0.1421, -4.6392]),
                0.7: np.array([0.4217, -0.4273, -6.0914]),
                0.8: np.array([0.5647, -1.3234, -7.3206]) 
                }
        def type2_tanh(x,  b, c, d):
            return -5.38*(np.tanh(b*x -c))+d
    else:
        print('Input method=1 for Ti scheme, method=2 for Tw scheme')

    ME = test['me'].values
    RE = abs(test['re'].values)
    
    pre_rain = test[test['ti']>  type2_tanh(np.log(3*ME/(2*RE)), *coefs[threshold])]
    pre_snow = test[test['ti']<= type2_tanh(np.log(3*ME/(2*RE)), *coefs[threshold])]
    return pre_rain, pre_snow