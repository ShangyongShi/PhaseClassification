#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 20:30:56 2021

Select the CloudSat files that pass the region we are interested in 


This one is written by Song and I just modified the paths and regions.
See filter_DPR.py for my easier version...

@author: psong & modified by sshi
"""

import os
from pyhdf import HDF, VS, V
from pyhdf.HDF import *
from pyhdf.VS import *
from pyhdf.SD import SD, SDC
import numpy as np
import pandas as pd
    
def read_2CSNOW_info(file):
    '''
    read one CloudSat 2C-SNOW-PROFILE.P1_R05 file information
    
    Input: file with full path
    Output:
            info: DataFrame that stores lon, lat, elev and snow_status
                snow_status (snow retrieval status)  >= 4 is bad data
    '''
    f = HDF(file)
    vs = f.vstart()

    latid = vs.attach('Latitude')
    latid.setfields('Latitude')
    nrecs, _, _, _, _ = latid.inquire()
    latitude = latid.read(nRec=nrecs)
    latid.detach()

    lonid = vs.attach('Longitude')
    lonid.setfields('Longitude')
    nrecs, _, _, _, _ = lonid.inquire()
    longitude = lonid.read(nRec=nrecs)
    lonid.detach()
    
    dem_pid = vs.attach('DEM_elevation')
    dem_pid.setfields('DEM_elevation')
    nrecs, _, _, _, _ = dem_pid.inquire()
    dem = dem_pid.read(nRec=nrecs)
    dem_pid.detach()
        
    snow_status = vs.attach('snow_retrieval_status')[:]
    

    # transfer 1 column array to 1 row
    longitude = np.array(longitude).T[0]
    latitude = np.array(latitude).T[0]
    dem = np.array(dem).T[0]
    snow_status = np.array(snow_status).T[0]
    info = pd.DataFrame(columns=['lon', 'lat', 'elev', 'snow_status'])
    info['lon'] = longitude
    info['lat'] = latitude  
    info['elev'] = dem
    info['snow_status'] = snow_status
    
    info.loc[info.elev<-999, 'elev'] = np.nan
    return info

def read_2CSNOW_sf(file):
    '''
    read one CloudSat 2C-SNOW-PROFILE.P1_R05 file
    
    Input: file with full path
    Output:
            sfc: snowfall rate, 2D, (len(granule), len(lev)), 
                missing value -999 has been put into np.nan
    '''
       
    # read data
    hdf = SD(file, SDC.READ)
    dset = hdf.select('snowfall_rate')
    sf = dset[:, :]
    sf[sf<-990]=np.nan
    return  sf


def check_if_in_box(file, leftlon, rightlon, lowerlat, upperlat):
    '''
    Check if a CloudSat 2C-SNOW-PROFILE.P1_R05 or 2B GEOPROF
    is in given spatial box range
    If the file does not pass the desired region, return False
    
    Input:
        file | str with full path
        leftlon, rightlon  | longitude boundary, [-180, 180]
        lowerlat, upperlat | latitude boundary, [-90, 90]
    Output:
        inbox_status | Logical, True or False
        
    '''
    
    info = read_info(file)
    
    inbox = ((info.lon>=leftlon) & (info.lon<=rightlon) & 
             (info.lat>=lowerlat) & (info.lat<=upperlat))
    if inbox.sum() == 0:
        inbox_status = False
    else:
        inbox_status = True
    
    return inbox_status
    
# def read_2CSNOW_in_box(file, leftlon, rightlon, lowerlat, upperlat):
#     '''
#     Read yearly CloudSat 2C-SNOW-PROFILE.P1_R05 in given spatial box range
#     If the file does not pass the desired region, return an empty DataFrame
    
#     Input:
#         file | str with full path
#         leftlon, rightlon  | longitude boundary, [-180, 180]
#         lowerlat, upperlat | latitude boundary, [-90, 90]
#     Output:
#         box_snow | DataFrame with columns:
#                  | ['lon', 'lat', 'elev', 'snow_status','lowest_sf']
        
#     '''
#     info = read_2CSNOW_info(file)

#     inbox =  ((info.lon>=leftlon) & (info.lon<=rightlon) & 
#              (info.lat>=lowerlat) & (info.lat<=upperlat))
    
#     box_snow =info.loc[inbox, 'lon':'snow_status'].copy()
#     box_snow['lowest_sf'] = np.nan
    
#     sf = read_2CSNOW_sf(file)
    
    
    
#     # If there are non-missing snowfall in one location (125 lev),
#     # record the lowest available value
#     sf_inbox = sf[inbox, :]
#     for irow in range(len(sf_inbox)):
#         tmp = sf_inbox[irow, :]
#         if sum(~np.isnan(tmp))>0:
#             idx = info[inbox].index[irow]
#             box_snow.loc[idx, 'lowest_sf'] = tmp[~np.isnan(tmp)][-1]
     
#     # Quality control using the snow_retrieval_status flag
#     box_snow.loc[box_snow.snow_status>=4, 'lowest_sf'] = np.nan
    
    # return box_snow
              
def read_lowest_snowfall(file):
    '''
    Read yearly CloudSat 2C-SNOW-PROFILE.P1_R05 
    Input:
        file | str with full path
    Output:
        snow | DataFrame with columns:
                 | ['lon', 'lat', 'elev', 'snow_status','lowest_sf']
        
    '''
    info = read_2CSNOW_info(file)

    snow = info.loc[:, 'lon':'snow_status'].copy()
    snow['lowest_sf'] = np.nan
    
    sf = read_2CSNOW_sf(file)
    
    # If there are non-missing snowfall in one location (125 lev),
    # record the lowest available value
    for irow in range(len(sf)):
        tmp = sf[irow, :]
        if sum(~np.isnan(tmp))>0:
            idx = info.index[irow]
            snow.loc[idx, 'lowest_sf'] = tmp[~np.isnan(tmp)][-1]
     
    # Quality control using the snow_retrieval_status flag
    snow.loc[snow.snow_status>=4, 'lowest_sf'] = np.nan
    
    # save the datetime information in the DataFrame
    filename = file.split('/')[-1] # extract the filename
    snow = set_index_with_cloudsat_obs_datetime(filename, snow)
    
    return snow


def read_info(file):
    '''
    read one CloudSat file information
    
    Input: file with full path
    Output:
            info: DataFrame that stores lon, lat, elev 
    '''
    f = HDF(file)
    vs = f.vstart()

    latid = vs.attach('Latitude')
    latid.setfields('Latitude')
    nrecs, _, _, _, _ = latid.inquire()
    latitude = latid.read(nRec=nrecs)
    latid.detach()

    lonid = vs.attach('Longitude')
    lonid.setfields('Longitude')
    nrecs, _, _, _, _ = lonid.inquire()
    longitude = lonid.read(nRec=nrecs)
    lonid.detach()
    
    dem_pid = vs.attach('DEM_elevation')
    dem_pid.setfields('DEM_elevation')
    nrecs, _, _, _, _ = dem_pid.inquire()
    dem = dem_pid.read(nRec=nrecs)
    dem_pid.detach()
       
    # transfer 1 column array to 1 row
    longitude = np.array(longitude).T[0]
    latitude = np.array(latitude).T[0]
    dem = np.array(dem).T[0]
    info = pd.DataFrame(columns=['lon', 'lat', 'elev'])
    info['lon'] = longitude
    info['lat'] = latitude  
    info['elev'] = dem
    info.loc[info.elev<-9990, 'elev'] = np.nan
    
    return info

def read_reflectivity(file):
    '''
    read one CloudSat 2B-GEOPROF_P1_R05 file
    
    ref: https://www.hdfeos.org/zoo/OTHER/2010128055614_21420_CS_2B-GEOPROF_
    GRANULE_P_R04_E03.hdf.py
    [1] http://www.cloudsat.cira.colostate.edu/dataSpecs.php
    
    Input: file with full path
    Output:
            rf: reflectivity, 2D, (len(granule), len(lev)), 
                missing value -999 has been put into np.nan
    '''
       
    # read data
    hdf = SD(file, SDC.READ)
    dset = hdf.select('Radar_Reflectivity')
    data = dset[:, :]
    
    valid_min = -4000
    valid_max = 5000
    invalid = np.logical_or(data < valid_min, data > valid_max)
    dataf = data.astype(float)
    dataf[invalid] = np.nan
    
    
    # # Read attributes.
    # attrs = dset.attributes(full=1)
    
    # sfa=attrs["factor"]
    # scale_factor = sfa[0] 
    
    # vra=attrs["valid_range"]
    # valid_min = vra[0][0]        
    # valid_max = vra[0][1] 

    # # Process valid range.
    # invalid = np.logical_or(data < valid_min, data > valid_max)
    # dataf = data.astype(float)
    # dataf[invalid] = np.nan
    
    # Apply scale factor according to [1].
    scale_factor = 100
    dataf = dataf / scale_factor
    
    return  dataf

def read_surface_reflectivity(file):
    '''
    The surface is contaminated, so we search at the bin -7 above the 
    surface height bin. Record the data when there is value available
    '''
    info = read_info(file)

    reflectivity = info.loc[:, 'lon':'elev'].copy()
    reflectivity['surface_rf'] = np.nan
    
    rf = read_reflectivity(file)
    sbin = read_surfaceheightbin(file)
    
    # If there are non-missing snowfall in one location (125 lev),
    # record the lowest available value
    for irow in range(len(rf)):
        rf_row = rf[irow, :]
        bin_row = sbin[irow]
        surface = bin_row-7
        if (bin_row>-1) & (~np.isnan(rf_row[surface])):
            idx = info.index[irow]
            reflectivity.loc[idx, 'surface_rf'] = rf_row[surface]
    
    # save the datetime information in the DataFrame
    filename = file.split('/')[-1] # extract the filename
    reflectivity = set_index_with_cloudsat_obs_datetime(filename, reflectivity)
    return reflectivity
    
def read_surfaceheightbin(file):
    '''
    Return an array of SurfaceHeightBin
    '''
    hdf = HDF(file)
    vs = hdf.vstart()
    vd = vs.attach('SurfaceHeightBin')
    data = np.array(vd[:])
    vd.detach()
    vs.end()
    return data
                          
def set_index_with_cloudsat_obs_datetime(filename, var):
    start_time = convert_start_time_str_to_datetime(filename)
    nscan = len(var)
    scan_times = get_datetime_for_scans(start_time, nscan)    
    var.index = scan_times
    var.index.rename('datetime', inplace=True)
    return var
                    
def convert_start_time_str_to_datetime(filename):
    '''
    R05 files: YYYYDDDHHMMSS_NNNNN_CS_2B-TAU_GRANULE_S_RVV_EVV_F00.hdf
    
    YYYYDDDHHMMSS = Year, Julian day, hour, minute, second of the first data 
    contained in the file (UTC)
    
    convert to  pandas datetime format
    
    '''
    fulldate = filename.split('_')[0]
    year = fulldate[0:4]
    julian_day = int(fulldate[4:7])-1 # date= first day of year + ndays 
    hour = fulldate[7:9]
    minute = fulldate[9:11]
    second = fulldate[11:13]
    
    start_time = pd.to_datetime(year)+pd.to_timedelta(str(julian_day) + 'days'+
                                         hour + ':' + minute + ':' + second)
    return start_time

def get_datetime_for_scans(start_time, nscan):
    '''
    sampling interveal is 0.16 seconds
    '''
    time = start_time
    times = [time]
    for i in range(nscan-1):
        time += pd.to_timedelta('0.16s')
        times.append(time)
    return times


# import subprocess
# args = ('./h4toh5', file)
# p = subprocess.Popen(args, stdout=subprocess.PIPE)