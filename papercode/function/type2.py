#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 16:23:09 2023

@author: sshi
"""
import pandas as pd
import numpy as np
from scipy import optimize as opt
from function.evaluate import print_metrics
PA = 'PA_tw'
NA = 'NA_tw'
xmax1, ymin, ymax = 500, 0, 500
global pa_intercept

def LDA_boundary_line(x_r, x_s):
    '''
    x_r, x_s: two columns indicate two groups
    Output: slope and intercept of the boundary line
    '''
    mu_r = x_r.mean(axis=0)
    mu_s = x_s.mean(axis=0)

    cov_r = np.cov(x_r, rowvar=False)
    cov_s = np.cov(x_s, rowvar=False)
    W = ((len(x_r)-1)*cov_r + (len(x_s)-1)*cov_s) / (len(x_r)+len(x_s)-2) # pooled within group covariance
    W_1 = np.linalg.inv(W)

    # T = np.cov(np.concatenate([x_r, x_s]), rowvar=False)
    # B = T-W
    # S = np.dot(W_1, B)

    # the boundary line is perpendicular to this line, crossing the midpoint of the centroids
    line = np.dot(W_1, (mu_s - mu_r))     
    midpoint = (mu_r+mu_s)/2
    
    slope = -line[0]/line[1]
    intercept = midpoint[1]-slope*midpoint[0]
    return slope, intercept

def fit_ln_metrics_loop_turning_point(cat):
    accuracy, recall, precision, f1score = np.zeros(50), np.zeros(50),np.zeros(50),np.zeros(50)
    POD, FAR, CSI, HSS = np.zeros(50), np.zeros(50),np.zeros(50),np.zeros(50)
    
    rain = cat[cat.wwflag==1]
    snow = cat[cat.wwflag==2]
    
    # loop different turning point between the two linear lines
    for i, xmin1 in enumerate(np.arange(1, 51, 1)):
        slope1, intercept1 = fit_LDA_for_greater_than(cat, xmin1)
        
        # find the best x intercept when must crossing (xmin1, f(xmin1) on LDA line)
        pa_intercept, aa, bb, cc, dd = find_x_intercept(cat, xmin1, slope1, intercept1)
    
        # linear function for x < xmin1
        slope2, intercept2 = fit_linear(x1=pa_intercept, y1=0, 
                                        x2=xmin1, y2=xmin1*slope1+intercept1)
        
        # Fit the two linear lines with ln function
        lnco = fit_ln_for_two_linear_lines(xmin1, pa_intercept, slope1, intercept1, slope2, intercept2)
        
        # test performance of this setting
        metrics = test_performance(cat, pa_intercept, lnco)
        accuracy[i], recall[i], precision[i], f1score[i], POD[i], FAR[i], CSI[i], HSS[i] = metrics.values()

    four = pd.DataFrame([accuracy, recall, precision, f1score, POD, FAR, CSI, HSS])
    df = pd.DataFrame(four.T.values, index=np.arange(1, 51, 1), 
                      columns=['accuracy', 'recall', 'precision', 'f1score',
                               'POD', 'FAR', 'CSI', 'HSS'])
    return df

def fit_LDA_for_greater_than(cat, xmin1):
    # LDA separation line for x>xmin1
    rain = cat[cat.wwflag==1]
    snow = cat[cat.wwflag==2]
    rain1 = rain[(rain[PA]<=xmax1) & (rain[PA]>=xmin1) & (rain[NA]>=ymin) & (rain[NA]<=ymax)]
    snow1 = snow[(snow[PA]<=xmax1) & (snow[PA]>=xmin1) & (snow[NA]>=ymin) & (snow[NA]<=ymax)]
    
    
    x_r1 = np.array((pd.concat([rain1[PA], rain1[NA]], axis=1)))
    x_s1 = np.array((pd.concat([snow1[PA], snow1[NA]], axis=1)))

    slope1, intercept1 = LDA_boundary_line(x_r1, x_s1)
    return slope1, intercept1

def find_x_intercept(cat, xmin1, slope1, intercept1):
    accuracy, recall, precision, f1score = np.zeros(3000), np.zeros(3000),np.zeros(3000),np.zeros(3000)

    for i, twPA in enumerate(np.arange(-10, 20, 0.01)):
        
        # linear function for x < xmin1
        slope2, intercept2 = fit_linear(x1=twPA, y1=0, 
                                        x2=xmin1, y2=xmin1*slope1+intercept1)

        pre_rain = cat[cat[NA]<cat[PA]*slope2+intercept2]['wwflag']
        pre_snow = cat[cat[NA]>=cat[PA]*slope2+intercept2]['wwflag']
        accuracy[i], recall[i], precision[i], f1score[i] = metrics(pre_rain, pre_snow)
    
    xs = np.arange(-10, 20, 0.01)
    # iacc = np.argmax(accuracy)
    iacc = np.argmax(accuracy+f1score) # edited 2022.10.20 
    x1 = xs[iacc]
    return x1, accuracy, recall, precision, f1score

def fit_linear(x1, y1, x2, y2):
    slope = (y2-y1) / (x2-x1)
    intercept = y1 - slope*x1
    return slope, intercept

def fit_ln_for_two_linear_lines(xmin1, pa_intercept, slope1, intercept1, slope2, intercept2):
    X1 = np.arange(xmin1, xmax1, 10) 
    Y1 = X1*slope1 + intercept1
    X2 = np.arange(pa_intercept, xmin1, 1) 
    Y2 = X2*slope2 + intercept2
    X = np.append(X2, X1)
    Y = np.append(Y2, Y1)
    def lnfunc(x, a, b, c, pa_intercept):
        return a+b*np.log(c*x+ np.exp(-a/b)-pa_intercept*c) # should cross (pa_intercept, 0)
    
    try:
        lnco, pcov = opt.curve_fit(lnfunc, X, Y)
    except RuntimeError:
        lnco = [np.nan, np.nan, np.nan]
    # y_pred = lnfunc(X, *lnco)

    # print('R2 score for fitting ln is %.4f' % r2_score(Y, y_pred))
    return lnco

def lnfunc(x, a, b, c, pa_intercept):
    return a+b*np.log(c*x+ np.exp(-a/b)-pa_intercept*c) # should cross (pa_intercept, 0)

def test_performance(test, pa_intercept, lnco):
    if sum(np.isnan(lnco))>1:
        return np.nan, np.nan, np.nan, np.nan
    else:
        pre_snow0 = test[test[PA] <= pa_intercept]['wwflag']
        test1 = test[test[PA] > pa_intercept]
        pre_snow1 = test1[test1[NA] >= lnfunc(test1[PA], *lnco)]['wwflag']
        pre_rain = test1[test1[NA] < lnfunc(test1[PA], *lnco)]['wwflag']
        pre_snow = pd.concat([pre_snow0, pre_snow1])

        metrics = print_metrics(pre_rain, pre_snow, False)
    return metrics

def metrics(pre_rain, pre_snow):
    TP = sum(pre_snow==2)
    FP = sum(pre_snow==1)
    P = len(pre_snow)
    TN = sum(pre_rain==1)
    FN = sum(pre_rain==2)
    N = len(pre_rain)
    accuracy = (TP+TN)/(P+N)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    f1score = 1/((TP+FN)/TP+(TP+FP)/TP)
    return accuracy, recall, precision, f1score



# ------------
def conp_1d(rain, snow, xvar, params):
    xmin, xmax, xbinsize=params
    num_rain = pd.DataFrame(data=0, index=[0],
                            columns=np.arange(xmin, xmax, xbinsize)+xbinsize/2)
    num_snow = pd.DataFrame(data=0, index=[0],
                            columns=np.arange(xmin, xmax, xbinsize)+xbinsize/2)
    for col in num_rain.columns:
        num_rain.loc[0, col] += ((rain[xvar]>col-xbinsize/2) & 
                                 (rain[xvar]<=col+xbinsize/2)).sum()
        num_snow.loc[0, col] += ((snow[xvar]>col-xbinsize/2) & 
                                 (snow[xvar]<=col+xbinsize/2)).sum()
    conp = num_snow/(num_rain+num_snow)
    return conp, num_rain, num_snow
    
def conp_2d(rain, snow, xvar, yvar, params):
    '''
    on the plot, column is x axis, row is y axis
    
    Input:
        rain
        snow
        xvar, yvar: str
        params:  xmin, xmax, xbinsize, ymin, ymax, ybinsize
    Output:
        conp, num_rain, num_snow
    '''
    xmin, xmax, xbinsize, ymin, ymax, ybinsize = params
    
    # column is x axis, row is y axis on the plot
    num_rain = pd.DataFrame(data=0, 
                            index=np.arange(ymin, ymax, ybinsize)+ybinsize/2, 
                            columns=np.arange(xmin, xmax, xbinsize)+xbinsize/2)
    num_snow = pd.DataFrame(data=0, 
                            index=np.arange(ymin, ymax, ybinsize)+ybinsize/2, 
                            columns=np.arange(xmin, xmax, xbinsize)+xbinsize/2)
    for row in num_rain.index:
        for col in num_rain.columns:
            num_rain.loc[row, col] += ((rain[yvar]>row-ybinsize/2) & 
                                       (rain[yvar]<=row+ybinsize/2) & 
                                       (rain[xvar]>col-xbinsize/2) & 
                                       (rain[xvar]<=col+xbinsize/2)).sum()
            num_snow.loc[row, col] += ((snow[yvar]>row-ybinsize/2) & 
                                       (snow[yvar]<=row+ybinsize/2) &
                                       (snow[xvar]>col-xbinsize/2) & 
                                       (snow[xvar]<=col+xbinsize/2)).sum()
    conp = num_snow/(num_snow+num_rain)
    return conp, num_rain, num_snow


def LDA_boundary_line(x_r, x_s):
    '''
    x_r, x_s: two columns indicate two groups
    Output: slope and intercept of the boundary line
    '''
    mu_r = x_r.mean(axis=0)
    mu_s = x_s.mean(axis=0)

    cov_r = np.cov(x_r, rowvar=False)
    cov_s = np.cov(x_s, rowvar=False)
    W = ((len(x_r)-1)*cov_r + (len(x_s)-1)*cov_s) / (len(x_r)+len(x_s)-2) # pooled within group covariance
    W_1 = np.linalg.inv(W)


    # the boundary line is perpendicular to this line,
    # crossing the midpoint of the centroids
    line = np.dot(W_1, (mu_s - mu_r))     
    midpoint = (mu_r+mu_s)/2
    
    slope = -line[0]/line[1]
    intercept = midpoint[1]-slope*midpoint[0]
    return slope, intercept

def fit_LDA_for_range(df, box):
    '''
    Input: df, box (xmin, xmax, ymin, ymax)
    Output: slope, intercept for LDA boundary line
    '''
    xmin, xmax, ymin, ymax = box
    
    # LDA separation line for x>xmin1
    rain = df[df.wwflag==1]
    snow = df[df.wwflag==2]
    rain1 = rain[(rain[PA]>=xmin) & (rain[PA]<=xmax) &
                 (rain[NA]>=ymin) & (rain[NA]<=ymax)]
    snow1 = snow[(snow[PA]>=xmin) & (snow[PA]<=xmax) &
                 (snow[NA]>=ymin) & (snow[NA]<=ymax)]
    
    x_r1 = np.array((pd.concat([rain1[PA], rain1[NA]], axis=1)))
    x_s1 = np.array((pd.concat([snow1[PA], snow1[NA]], axis=1)))

    slope1, intercept1 = LDA_boundary_line(x_r1, x_s1)
    return slope1, intercept1