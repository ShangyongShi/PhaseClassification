#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:02:44 2024

default to using Ti for computation
Given the input, output the snow probability

snow probability
snow/precipitation ratio

@author: ssynj
"""

import os
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------
'''



'''

#################################################################
# ------------------------- simple classification ------------------------- #
#################################################################
def classify_Ti( Tw, Ti, ME, RE, TYPE, threshold):
    '''
    Given the threshold, classify rain or snow.
    
    Return rain/snow mask
    
    
    '''

    snowp = snowprob_func(Tw, Ti, ME, RE, TYPE)

    snow = np.full_like(snowp, False)
    rain = np.full_like(snowp, False)

    snow[snowp >= threshold] = True
    rain[snowp <  threshold] = True

    return snow, rain




#################################################################
# ------------------------- Func method------------------------- #
#################################################################

def snowprob_func(Tw, Ti, ME, RE, TYPE):
    '''
    Inputs are 2d arrays.
    Use functions
    
    Input:
        T: ice-bulb temperature
        ME
        RE
        TYPE: 1 or 2
    '''
    snowp = np.full_like(Ti, np.nan, dtype=np.float64)
    
    # ------------------------- TYPE == 0 ------------------------- #
    mask_type0 = (TYPE == 0)
    snowp = np.where(mask_type0 & (Tw > 1.6), 0, 1)
    
    # ------------------------- TYPE == 1 ------------------------- #
    # Apply computations only where TYPE == 1
    mask_type1 = (TYPE == 1)
    snowp = compute_snowp(snowp, Ti, ME, RE, TYPE, mask_type1, 1)
            
    # ------------------------- TYPE == 2 ------------------------- #
    mask_type2 = (TYPE == 2)
    snowp = compute_snowp(snowp, Ti, ME, RE, TYPE, mask_type2, 2)
    
    return snowp


def type1_exp(x, a, b):
    return a * np.exp(b*x ) 


def type2_tanh(x,  b, c, d):
    return -5.38*(np.tanh(b*x -c))+d

coefs_type1 = {
            0.3: np.array([1.68332365, -0.1811878 ]),
            0.4: np.array([1.42235443, -0.22139454]),
            0.5: np.array([1.19237535, -0.29651954]),
            0.6: np.array([0.93126144, -0.42526325]),
            0.7: np.array([0.92178506, -1.31780889]),
            0.8: np.array([0.4477541 , -2.14061253])}
    

coefs_type2 = {0.3: np.array([0.3216, -0.1059, 0.4968]),
            0.4: np.array([0.2770, -0.2366, -0.8472]),
            0.5: np.array([0.2819, -0.0287, -3.4724]),
            0.6: np.array([0.3112, -0.1421, -4.6392]),
            0.7: np.array([0.4217, -0.4273, -6.0914]),
            0.8: np.array([0.5647, -1.3234, -7.3206]) 
        }


def compute_snowp(snowp, T, ME, RE, TYPE, mask, lut_model):
    """
    Update the input snowp array for grid points specified by 'mask', using either the TYPE-1 or TYPE-2 procedure.
    
    Parameters:
      snowp   : np.ndarray
                Initial snow probability array (will be updated at grid points in mask).
      T       : np.ndarray
                Temperature array (for TYPE==1, this is Ti; for TYPE==2, also Ti in our case).
      ME      : np.ndarray
                Melting energy array.
      RE      : np.ndarray
                Refreezing energy array.
      TYPE    : np.ndarray
                Array indicating the type at each grid point (not used here for processing since lut_model is provided).
      mask    : boolean np.ndarray
                Boolean mask indicating grid points to process.
      lut_model: int
                Indicator of which procedure to use: 1 for TYPE-1 and 2 for TYPE-2.
                
    Returns:
      np.ndarray:
          Updated snowp array (values clipped between 0 and 1) for grid points in mask.
    
    The function uses:
      - For lut_model==1: energy_ratio = ME, type1_exp, coefs_type1, thresholds [0.2, ..., 0.8].
      - For lut_model==2: energy_ratio = ln(3*ME/(2*|RE|)), type2_tanh, coefs_type2, thresholds [0.3, ..., 0.8].
    """
    # Ensure arrays are numpy arrays.
    T = np.asarray(T)
    ME = np.asarray(ME)
    RE = np.asarray(RE)
    TYPE = np.asarray(TYPE)
    mask = np.asarray(mask)
    
    # Choose parameters based on lut_model.
    if lut_model == 1:
        thresholds = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        model_func = type1_exp
        energy_ratio = ME  # For TYPE-1, energy ratio is simply ME.
        coefs = coefs_type1
    elif lut_model == 2:
        thresholds = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        model_func = type2_tanh
        energy_ratio = np.log((3 * ME) / (2 * np.abs(RE)))  # For TYPE-2.
        coefs = coefs_type2
    else:
        raise ValueError("lut_model must be 1 or 2.")
    
    # Compute predicted T values for each threshold.
    T_vals = {thr: model_func(energy_ratio, *coefs[thr]) for thr in thresholds}
    
    # Process warm-side extrapolation.
    warm_mask = mask & (T > T_vals[thresholds[0]])
    if np.any(warm_mask):
        # For warm-side, use thresholds[0] and threshold 0.5.
        T_low = T_vals[thresholds[0]]
        T_high = T_vals[0.5]
        denom = T_high - T_low
        ratio = np.zeros_like(T)
        valid = (denom != 0)
        ratio[valid] = (T[valid] - T_low[valid]) / denom[valid]
        extrap_val = thresholds[0] + ratio * (0.5 - thresholds[0])
        snowp = np.where(warm_mask, extrap_val, snowp)
    
    # Process cold-side extrapolation.
    cold_mask = mask & (T < T_vals[thresholds[-1]])
    if np.any(cold_mask):
        # For cold-side, use thresholds 0.5 and thresholds[-1].
        T_low = T_vals[0.5]
        T_high = T_vals[thresholds[-1]]
        denom = T_high - T_low
        ratio = np.zeros_like(T)
        valid = (denom != 0)
        ratio[valid] = (T[valid] - T_low[valid]) / denom[valid]
        extrap_val = 0.5 + ratio * (thresholds[-1] - 0.5)
        snowp = np.where(cold_mask, extrap_val, snowp)
    
    # Process intermediate values via interpolation.
    int_mask = mask & ~(warm_mask | cold_mask)
    if np.any(int_mask):
        for i in range(len(thresholds) - 1):
            thr1, thr2 = thresholds[i], thresholds[i+1]
            T1 = T_vals[thr1]
            T2 = T_vals[thr2]
            pair_mask = int_mask & (((T1 <= T) & (T < T2)) | ((T2 <= T) & (T < T1)))
            if np.any(pair_mask):
                denom = T2 - T1
                ratio = np.zeros_like(T)
                valid = (abs(denom)>1e-6 )
                ratio[valid] = (T[valid] - T1[valid]) / denom[valid]
                interp_val = thr1 + ratio * (thr2 - thr1)
                snowp = np.where(pair_mask, interp_val, snowp)
                
        # for i in range(len(thresholds) - 1):       
        #     p1, p2 = thresholds[i], thresholds[i + 1]  # Consecutive probability thresholds
        #     T1 = type1_exp(ME, *coefs_type1[p1])
        #     T2 = type1_exp(ME, *coefs_type1[p2])
        #     T1[T1<1e-6] = 0
        #     T2[T2<1e-6] = 0
        #     # Find where Ti falls between T1 and T2
        #     mask = int_mask & ((T1 <= T) & (T < T2) | (T2 <= T) & (T < T1))
            
        #     # Compute interpolation factor
        #     interfactor = np.where( abs(T1- T2)>1e-6, (T - T1) / (T2 - T1), 0)

        #     # Compute interpolated snowp value
        #     snowp = np.where(mask, p1 + interfactor * (p2 - p1), snowp)
    
    return np.clip(snowp, 0, 1)

    
    
    
# mask_type1 = (TYPE == 1)

# # Compute boundary conditions
# T_lowest = type1_exp(ME, *coefs_type1[0.7])
# T_highest = type1_exp(ME, *coefs_type1[0.2])

# # Apply boundary values
# snowp = np.where(mask_type1 & (Ti > T_highest), 0, snowp)
# snowp = np.where(mask_type1 & (Ti < T_lowest), 1, snowp)
# # if Ti > type1_exp(ME, coefs[0.3]):
# #     snowp = 0
# # if Ti < type1_exp(ME, coefs[0.8]):
# #     snowp = 1

# thresholds = sorted(coefs_type1.keys())  # Ensure thresholds are sorted in ascending order

# for i in range(len(thresholds) - 1):
#     p1, p2 = thresholds[i], thresholds[i + 1]  # Consecutive probability thresholds
#     T1 = type1_exp(ME, *coefs_type1[p1])
#     T2 = type1_exp(ME, *coefs_type1[p2])

#     # Find where Ti falls between T1 and T2
#     mask = mask_type1 & ((T1 <= Ti) & (Ti < T2) | (T2 <= Ti) & (Ti < T1))
    
#     # Compute interpolation factor
#     interfactor = np.where(T1 != T2, (Ti - T1) / (T2 - T1), 0)

#     # Compute interpolated snowp value
#     snowp = np.where(mask, p1 + interfactor * (p2 - p1), snowp)
    





    # # Compute log expression safely (avoid division by zero)
    # log_input = np.log(np.where(RE != 0, (3 * ME) / (2 * np.abs(RE)), np.nan))

    # # Compute boundary conditions for TYPE == 2
    # T_lowest_type2 = type2_tanh(log_input, *coefs_type2[0.8])
    # T_highest_type2 = type2_tanh(log_input, *coefs_type2[0.3])

    # # Apply boundary values for TYPE == 2
    # snowp = np.where(mask_type2 & (Ti > T_highest_type2), 0, snowp)
    # snowp = np.where(mask_type2 & (Ti < T_lowest_type2),  1, snowp)

    # # Compute intermediate probability values for TYPE == 2
    # for i in range(len(thresholds) - 1):
    #     p1, p2 = thresholds[i], thresholds[i + 1]
    #     T1 = type2_tanh(log_input, *coefs_type2[p1])
    #     T2 = type2_tanh(log_input, *coefs_type2[p2])

    #     # Find where Ti falls between T1 and T2
    #     mask = mask_type2 & ((T1 <= Ti) & (Ti < T2) | (T2 <= Ti) & (Ti < T1))
        
    #     # Compute interpolation factor
    #     interfactor = np.where(T1 != T2, (Ti - T1) / (T2 - T1), 0)

    #     # Compute interpolated snowp value
    #     snowp = np.where(mask, p1 + interfactor * (p2 - p1), snowp)
    
    
    
    
#################################################################
# ------------------------- LUT method------------------------- #
#################################################################
def snowprob_LUT(Tw, Ti, ME, RE, TYPE):
    """
    Lookup snow probability values from LUTs using array inputs.
   
    
    Parameters:
      T : np.ndarray
          Temperature array (Ti 
      ME : np.ndarray
          Melting energy array.
      RE : np.ndarray
          Refreezing energy array.
      TYPE : np.ndarray
          Array with values 1 or 0.
    
    Returns:
      np.ndarray
          Array of snow probability values (clipped between 0 and 1).
    """
    Ti = np.asarray(Ti)
    ME = np.asarray(ME)
    RE = np.asarray(RE)
    TYPE = np.asarray(TYPE)
    
    snowp = np.full(Ti.shape, np.nan, dtype=np.float64)
    
    LUT1 = read_LUT(pre_type=1) 
    LUT2 = read_LUT(pre_type=2) 
    
    mask0 = (TYPE==0)
    if np.any(mask0):
        T0 = Tw[mask0]  # For TYPE==0, T is the Tw value.
        # Apply new rule: if Tw > 1.6, then snowp = 0; otherwise, snowp = 1.
        snowp[mask0] = np.where(T0 > 1.6, 0, 1)
        
    # Mask for TYPE == 1 (energy_ratio = ME)
    mask1 = (TYPE == 1)
    if np.any(mask1):
        energy = ME[mask1]
        T1 = Ti[mask1]
        snowp[mask1] = lookup_LUT(LUT1, energy, T1)
    
    # Mask for TYPE == 2 (energy_ratio = ln(3*ME/(2*abs(RE))))
    mask2 = (TYPE == 2)
    if np.any(mask2):
        ratio = np.log((3 * ME[mask2]) / (2 * np.abs(RE[mask2])))
        T2 = Ti[mask2]
        snowp[mask2] = lookup_LUT(LUT2, ratio, T2)
    
    
    snowp = np.clip(snowp, 0.0, 1.0)
    snowp[np.isnan(Tw)] = np.nan
    return snowp




def read_LUT(pre_type):
    """
    Read the LUT file from ./LUTs and assign row/column labels.
    
    Parameters:
      pre_type : int
          (Here pre_type is assumed to be the same as TYPE; for TYPE==1, pre_type==1, for TYPE==0, pre_type==2.)
    
    Returns:
      LUT : pandas.DataFrame
          A DataFrame containing the LUT values with its index (energy or energy ratio) 
          and columns (temperature) set.
    """
    # Determine filename based on type:
    
    temp_str = "Ti"

    if pre_type == 1:
        pre_str = "Type1"
    elif pre_type == 2:
        pre_str = "Type2"
    else:
        raise ValueError("Invalid pre_type. Use 1 or 2.")

    # Get the absolute path of the current module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go one directory up (to project_root) and then to the LUT folder.
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    lut_dir = os.path.join(project_root, "LUT")

    filename = f"{lut_dir}/{temp_str}_{pre_str}_LUT.txt"
    
    # Read the LUT file 
    LUT = pd.read_csv(filename, delim_whitespace=True, header=None)
    nrow, ncol = LUT.shape

    # Define row and column labels 
    if  pre_type == 1:
        ratios = np.array([0.2 + i/10.0 for i in range(nrow)])
        ts = np.array([0.2 + i/10.0 for i in range(ncol)])
        
    else:# pre_type == 2:
        ratios = np.array([-10.0 + i/10.0 for i in range(nrow)])
        ts = np.array([-8.0 + i/10.0 for i in range(ncol)])

        
    LUT.index = ratios
    LUT.columns = ts
    return LUT

def nearest_index(sorted_array, values):
    """
    For each element in 'values', find the index of the nearest element in
    the sorted 1D array 'sorted_array'. Both inputs are numpy arrays.
    """
    sorted_array = np.asarray(sorted_array)
    values = np.asarray(values)
    idx = np.searchsorted(sorted_array, values, side='left')
    idx = np.where(idx >= len(sorted_array), len(sorted_array) - 1, idx)
    idx_minus = np.maximum(idx - 1, 0)
    diff1 = np.abs(values - sorted_array[idx_minus])
    diff2 = np.abs(sorted_array[idx] - values)
    nearest_idx = np.where(diff1 <= diff2, idx_minus, idx)
    return nearest_idx

def lookup_LUT(LUT, energy_ratios, Ts):
    """
    Given a LUT DataFrame (with numeric index and columns), and arrays of
    energy_ratios and temperatures Ts (of the same shape), perform a nearest-
    neighbor lookup and return an array of LUT values.
    """
    ratios = LUT.index.values   # 1D array of energy (or energy ratio) values
    temps = LUT.columns.values  # 1D array of temperature values
    
    energy_ratios = np.asarray(energy_ratios)
    Ts = np.asarray(Ts)
    
    flat_energy = energy_ratios.ravel()
    flat_T = Ts.ravel()
    
    row_idx = nearest_index(ratios, flat_energy)
    col_idx = nearest_index(temps, flat_T)
    
    result_flat = LUT.values[row_idx, col_idx]
    return result_flat.reshape(energy_ratios.shape)



# Example usage:
if __name__ == "__main__":
    # Create example 2D arrays (e.g., a grid)
    
    Tw = np.array([[ -2.0, -1.5], [-2.5, -3.0]])
    Ti = np.array([[ -2.0, -1.5], [-2.5, -3.0]])
    ME = np.array([[0.5, 0.6], [0.7, 0.8]])
    RE = np.array([[0.1, 0.2], [0.3, 0.4]])
    # TYPE: 1 indicates use Ti-based LUT; 0 indicates use Tw-based LUT.
    # Here, we'll create a combined TYPE array, for example:
    TYPE = np.array([[1, 0], [1, 0]])
    
    snowp = snowprob_LUT(Tw, Ti, ME, RE, TYPE)
    print("Snow probability (LUT lookup):")
    print(snowp)
    
    
    
    
    
    


    



    
    







#%%
# # old code below
# def classify(test, threshold, method):
#     '''
#     Input:
#         test: DataFrame, with columns of wet-bulb temperature, ice-bulb temperature
#                 melting and refreezing energy, and sounding type.
#             test.columns=['tw', 'ti', 'me', 're', 'type']
#         threshold: threshold for conditional probability of snow, 
#             available values: 0.3, 0.4, ..., 0.8
#         method: 1 for Ti and 2 for Tw

#     Output:
#         rain: DataFrame with rows predicted as rain
#         snow: DataFrame with rows predicted as snow
#     '''
#     test0 = test[ test['type']==0]
#     test1 = test[ test['type']==1]
#     test2 = test[ test['type']==2]
    
#     pre_rain0, pre_snow0 = classify_type0(test0)
#     pre_rain1, pre_snow1 = classify_type1(test1, threshold, method)
#     pre_rain2, pre_snow2 = classify_type2(test2, threshold, method)
    
#     pre_rain = pd.concat([pre_rain0, pre_rain1, pre_rain2])
#     pre_snow = pd.concat([pre_snow0, pre_snow1, pre_snow2])   
#     return pre_rain, pre_snow

# def classify_type0(test):
#     snow = test[test['tw']<=1.6]
#     rain = test[test['tw']>1.6]
#     return rain, snow



# def classify_type1(test, threshold, method):
#     '''
#     Seperation for soundings with only one melting at the bottom
#     '''
#     if method==1:
#         # Ti
#         coefs = {
#                 0.3: np.array([1.68332365, -0.1811878 ]),
#                 0.4: np.array([1.42235443, -0.22139454]),
#                 0.5: np.array([1.19237535, -0.29651954]),
#                 0.6: np.array([0.93126144, -0.42526325]),
#                 0.7: np.array([0.92178506, -1.31780889]),
#                 0.8: np.array([0.4477541 , -2.14061253])}
#     elif method==2:
#         coefs = {0.3: np.array([1.8352, -0.1688]),
#                 0.4: np.array([1.5738, -0.2053]),
#                 0.5: np.array([1.2957, -0.2748]),
#                 0.6: np.array([1.0060, -0.3635]),
#                 0.7: np.array([0.7358, -0.3635]),
#                 0.8: np.array([0.4, -0.4])
#                 }
#     else:
#         print('Input method=1 for Ti scheme, method=2 for Tw scheme')

#     popt = coefs[threshold]
#     pre_snow = test[test['ti'] <= type1_exp(test['me'], *popt)]
#     pre_rain = test[test['ti'] >  type1_exp(test['me'], *popt)]
#     return pre_rain, pre_snow




# def classify_type2(test, threshold, method):
#     '''
#     Separation for soundings with a melting layer and a refreezing layer
#     '''
#     if method==1:
#         coefs = {0.3: np.array([ 0.11992169, -0.48402358,  7.40607351]),
#                 0.4: np.array([ 0.08766819,  0.0641349 , -3.59167715]),
#                 0.5: np.array([  0.14446052,   0.63535494, -14.0823996 ]),
#                 0.6: np.array([  0.21317597,   0.69167411, -16.86781474]),
#                 0.7: np.array([  0.29617753,   0.47365385, -18.36315865]),
#                 0.8: np.array([  0.39996308,  -0.11066325, -19.82783235])}
#         def type2_tanh(x,  b, c, d):
#             return -18*(np.tanh(b*x -c))+d
#     elif method==2:
#         coefs = {0.3: np.array([0.3216, -0.1059, 0.4968]),
#                  0.4: np.array([0.2770, -0.2366, -0.8472]),
#                  0.5: np.array([0.2819, -0.0287, -3.4724]),
#                  0.6: np.array([0.3112, -0.1421, -4.6392]),
#                  0.7: np.array([0.4217, -0.4273, -6.0914]),
#                  0.8: np.array([0.5647, -1.3234, -7.3206]) 
#                 }
#         def type2_tanh(x,  b, c, d):
#             return -5.38*(np.tanh(b*x -c))+d
#     else:
#         print('Input method=1 for Ti scheme, method=2 for Tw scheme')

#     ME = test['me'].values
#     RE = abs(test['re'].values)
    
#     pre_rain = test[test['ti']>  type2_tanh(np.log(3*ME/(2*RE)), *coefs[threshold])]
#     pre_snow = test[test['ti']<= type2_tanh(np.log(3*ME/(2*RE)), *coefs[threshold])]
#     return pre_rain, pre_snow