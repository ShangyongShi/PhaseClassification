# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:18:52 2023

@author: ssynj
"""
import numpy as np
import pandas as pd

def classify(test, threshold):
    '''
    test: the input dataframe. with columns "type_tw", "tw", "PA_tw", "NA_tw"
    threshold: the probability threshold to classify snow. 
        choose from: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    '''
    test0 = test[ test['type_tw']==0]
    test1 = test[ test['type_tw']==1]
    test2 = test[ test['type_tw']==2]
    
    pre_rain0, pre_snow0 = classify_type0(test0)
    pre_rain1, pre_snow1 = classify_type1(test1, threshold)
    pre_rain2, pre_snow2 = classify_type2(test2, threshold)
    
    pre_rain = pd.concat([pre_rain0, pre_rain1, pre_rain2])
    pre_snow = pd.concat([pre_snow0, pre_snow1, pre_snow2])   
    return pre_rain, pre_snow

def classify_type0(test):
    snow = test[test['tw']<=1.6]
    rain = test[test['tw']>1.6]
    return rain, snow

def type1_exp(x, m, t):
    return m * np.exp(t*x ) 

def classify_type1(test, threshold):
    '''
    Seperation for soundings with only one melting at the bottom
    '''
    coefs = {0.3: np.array([1.8352, -0.1688]),
            0.4: np.array([1.5738, -0.2053]),
            0.5: np.array([1.2957, -0.2748]),
            0.6: np.array([1.0060, -0.3635]),
            0.7: np.array([0.7358, -0.3635]),
            0.8: np.array([0.4, -0.4])
            }
    popt = coefs[threshold]
    pre_snow = test[test['tw'] <= type1_exp(test['PA_tw'], *popt)]
    pre_rain = test[test['tw'] >  type1_exp(test['PA_tw'], *popt)]
    return pre_rain, pre_snow


def type2_tanh(x,  b, c, d):
    return -5.38*(np.tanh(b*x -c))+d

def classify_type2(test, threshold):
    '''
    Separation for soundings with a melting layer and a refreezing layer
    '''
    coefs = {0.3: np.array([0.3216, -0.1059, 0.4968]),
            0.4: np.array([0.2770, -0.2366, -0.8472]),
            0.5: np.array([0.2819, -0.0287, -3.4724]),
            0.6: np.array([0.3112, -0.1421, -4.6392]),
            0.7: np.array([0.4217, -0.4273, -6.0914]),
            0.8: np.array([0.5647, -1.3234, -7.3206]) 
            }

    ME = test['PA_tw'].values
    RE = test['NA_tw'].values
    
    pre_rain = test[test['tw']>  type2_tanh(np.log(3*ME/(2*RE)), *coefs[threshold])]
    pre_snow = test[test['tw']<= type2_tanh(np.log(3*ME/(2*RE)), *coefs[threshold])]
    return pre_rain, pre_snow
    
def print_metrics(pre_rain, pre_snow, printflag):
    '''
    print_metrics(pre_rain, pre_snow, printflag)
    '''
    TP = sum(pre_snow.wwflag==2)
    FP = sum(pre_snow.wwflag==1)
    P = len(pre_snow)
    TN = sum(pre_rain.wwflag==1)
    FN = sum(pre_rain.wwflag==2)
    N = len(pre_rain)
    
    metrics = {}
    
    metrics['accuracy'] = (TP+TN)/(P+N)
    if TP+FN == 0:
        metrics['recall'] = np.nan
    else:
        metrics['recall'] = TP/(TP+FN)
    
    if TP+FP ==0:
        metrics['precision'] = np.nan
        metrics['POFA'] = np.nan
    else:
        metrics['precision'] = np.divide(TP, (TP+FP))
        metrics['POFA'] = 1-metrics['precision']
        
    if FP+TN==0:
        metrics['POFD'] = np.nan
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
