#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:35:34 2024

@author: ssynj
"""
import pandas as pd
import numpy as np
def classify(test, threshold):
    test0 = test[ test['type_ti']==0]
    test1 = test[ test['type_ti']==1]
    test2 = test[ test['type_ti']==2]
    
    pre_rain0, pre_snow0 = classify_type0(test0)
    pre_rain1, pre_snow1 = classify_type1(test1, threshold)
    pre_rain2, pre_snow2 = classify_type2(test2, threshold)
    
    pre_rain = pd.concat([pre_rain0, pre_rain1, pre_rain2])
    pre_snow = pd.concat([pre_snow0, pre_snow1, pre_snow2])   
    return pre_rain, pre_snow

def classify_type0(test):
    snow = test[test['tw']<=1.6]
    rain = test[test['tw']>1.6]
    return rain.wwflag, snow.wwflag

def type1_exp(x, m, t):
    return m * np.exp(t*x ) 

def classify_type1(test, threshold):
    '''
    Seperation for soundings with only one melting at the bottom
    '''
    coefs = {
             0.3: np.array([1.68332365, -0.1811878 ]),
             0.4: np.array([1.42235443, -0.22139454]),
             0.5: np.array([1.19237535, -0.29651954]),
             0.6: np.array([0.93126144, -0.42526325]),
             0.7: np.array([0.92178506, -1.31780889]),
             0.8: np.array([0.4477541 , -2.14061253])}
    popt = coefs[threshold]
    pre_snow = test[test['ti'] <= type1_exp(test['PA_ti'], *popt)]
    pre_rain = test[test['ti'] >  type1_exp(test['PA_ti'], *popt)]
    return pre_rain, pre_snow


def type2_tanh(x,  b, c, d):
    return -18*(np.tanh(b*x -c))+d

def classify_type2(test, threshold):
    '''
    Separation for soundings with a melting layer and a refreezing layer
    '''
    coefs = {0.3: np.array([ 0.11992169, -0.48402358,  7.40607351]),
             0.4: np.array([ 0.08766819,  0.0641349 , -3.59167715]),
             0.5: np.array([  0.14446052,   0.63535494, -14.0823996 ]),
             0.6: np.array([  0.21317597,   0.69167411, -16.86781474]),
             0.7: np.array([  0.29617753,   0.47365385, -18.36315865]),
             0.8: np.array([  0.39996308,  -0.11066325, -19.82783235])}

    ME = test['PA_ti'].values
    RE = abs(test['NA_ti'].values)
    
    pre_rain = test[test['ti']>  type2_tanh(np.log(3*ME/(2*RE)), *coefs[threshold])]
    pre_snow = test[test['ti']<= type2_tanh(np.log(3*ME/(2*RE)), *coefs[threshold])]
    return pre_rain, pre_snow