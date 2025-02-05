#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:07:40 2023

@author: sshi
"""
from function.watervapor import td2rh
#from function import prob
import sys
sys.path.append('/r1/sshi/sounding_phase/04-Temp_Threshold/nimbus/') 
# evalueate lapse rate and other methods

import numpy as np
def print_metrics(pre_rain, pre_snow, printflag):
    '''
    print_metrics(pre_rain, pre_snow, printflag)
    '''
    TP = sum(pre_snow==2)
    FP = sum(pre_snow==1)
    P = len(pre_snow)
    TN = sum(pre_rain==1)
    FN = sum(pre_rain==2)
    N = len(pre_rain)
    
    metrics = {}
    
    metrics['accuracy'] = (TP+TN)/(P+N)
    if TP+FN == 0:
        metrics['recall'] = np.nan
    else:
        metrics['recall'] = TP/(TP+FN)
    
    if TP+FP ==0:
        metrics['precision'] = np.nanS
        metrics['POFA'] = np.nan
    else:
        metrics['precision'] = np.divide(TP, (TP+FP))
        metrics['POFA'] = 1-metrics['precision']
        
    if FP+TN==0:
        metrics['POFD'] = np.nanS
    else:
        metrics['POFD'] = FP/(FP+TN)
        
    if TP==0:
        metrics['f1score'] = np.nan
    else:
        metrics['f1score'] = 1 /((TP+FN)/TP+(TP+FP)/TP) 
     
    metrics['POD'] = metrics['recall']
    
    if TP+FP+FN==0:
        metrics['CSI'] = np.nan
    else:
        metrics['CSI'] = TP/(TP+FP+FN)
      
    if ( (TP+FN)*(FN+TN) + (TP+FP)*(FP+TN) ) ==0:
        metrics['HSS'] = np.nan
    else:
        metrics['HSS'] = (2*(TP*TN-FP*FN)) / ( (TP+FN)*(FN+TN) + (TP+FP)*(FP+TN) )   
      
    metrics['TSS'] = metrics['POD'] - metrics['POFD']
    
    
    if printflag:
        strform = 'True positive: %d | False positive: %d | P_PRE:%d\n' +\
                  'False negative: %d | True negative: %d | N_PRE:%d \n' +\
                   'P_OBS: %d | N_OBS: %d\n | TOTAL: %d \n\n' +\
                  'Accuracy: %5.3f \n' +\
                  'Recall: %5.3f \n' +\
                  'Precision: %5.3f \n' +\
                  'F1Score: %5.3f \n' +\
                  'POD (Probability of Detection): %5.3f \n' +\
                  'POFA (False Alarm Ratio): %5.3f \n' +\
                  'POFD (Probability of false detection, False Alarm Rate): %5.3f \n' +\
                  'CSI (Critical Success Index): %5.3f \n' +\
                  'HSS (Heidke Skill Score): %5.3f \n' +\
                     'TSS (True Skill Statistics): %5.3f \n'
        print(strform %(TP, FP, TP+FP, FN, TN, TN+FN, TP+FN, FP+TN, P+N, 
                        metrics['accuracy'], metrics['recall'], 
                        metrics['precision'], metrics['f1score'],
                        metrics['POD'], metrics['POFA'], metrics['POFD'],
                        metrics['CSI'], metrics['HSS'], metrics['TSS']))
    return metrics


def evaluate_probsnow(test, value, t_tw):
    '''
    Evaluate the performance of probsnow scheme
    if we use the sow probability of "value" to separate rain and snow

    Parameters
    ----------
    test : dataframe, evaluation data set
    t_tw:  str, 't' or 'tw'. Use t or tw for probsnow scheme. Default to 'tw'
    value : threshold 

    Returns
    -------
    metrics: accuracy, recall, precision, f1score, POD, FAR, CSI, HSS

    '''
    
    if t_tw == 'tw':
        probcol, lrcol = 'probsnow_tw', 'lapse_rate_tw'
    elif t_tw == 't':
        probcol, lrcol = 'probsnow_t', 'lapse_rate_t'
        
        
    test[probcol] = np.nan
    for idx in test.index:
        tc = test.loc[idx, 't']
        tdc = test.loc[idx, 'td']
        pmb = test.loc[idx, 'p']
        rhp = td2rh(tc, tdc)
        lr = test.loc[idx, lrcol]
        
        if t_tw == 'tw':
            test.loc[idx, probcol] = prob.probsnow(tc, pmb, rhp, lr, -999.9, 1)
        else:
            test.loc[idx, probcol] = prob.probsnow(tc, pmb, -999.9, lr, -999.9, 1)
        
    pre_rain = test[test[probcol] <  value]['wwflag']
    pre_snow = test[test[probcol] >= value]['wwflag']
    metrics = print_metrics(pre_rain, pre_snow, True)
    return metrics

def evaluate_model(test, xvar, yvar, model, coef):
    pre_snow = test[test[yvar]<=model(test[xvar], *coef)]['wwflag']
    pre_rain = test[test[yvar]> model(test[xvar], *coef)]['wwflag']
    metrics = print_metrics(pre_rain, pre_snow, True)
    return metrics
