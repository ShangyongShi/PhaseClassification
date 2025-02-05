
import pandas as pd
import numpy as np

def initiate_profiles(levels)    :
    colNames= ['ID', 'year', 'month','day','hour','reltime','numlev',
               'lat','lon'] + levels
    temp = pd.DataFrame(columns=colNames)
    rh   = pd.DataFrame(columns=colNames)
    wdir = pd.DataFrame(columns=colNames)
    wspd = pd.DataFrame(columns=colNames)
    dpdp = pd.DataFrame(columns=colNames)
    gph = pd.DataFrame(columns=colNames)
    return temp, rh, wdir, wspd, dpdp, gph
    
def read_IGRA(IGRA_ID, pre_datetime)  :
    levels = np.arange(0, 20)
    level_labels = [str(level) for level in levels]
    temp, rh, wdir, wspd, dpdp, gph = initiate_profiles(level_labels) 
    
    file = '/data/sshi/IGRA2/IGRA2_data/'+IGRA_ID+'-data.txt'
    with open(file, 'r') as f:
        record = 0
        while True:
            
            line = f.readline()
            if not line:
                break

            # read header
            header = line;
            NUMLEV = int(header[32:36])
                
            # read specific dates
            DATE = header[13:26]
            if (DATE[-2:] == '99') & (header[27:31]!='9999'):
                DATE = header[13:23] + header[27:29] # hour missing, use reltime
            if DATE not in pre_datetime:
                for i in range(NUMLEV):
                    line = f.readline()
                continue                 
                
            temp.loc[record, 'ID']      = header[1:12];
            temp.loc[record, 'year']    = int(header[13:17]);
            temp.loc[record, 'month']   = int(header[18:20]);
            temp.loc[record, 'day']     = int(header[21:23]);
            temp.loc[record, 'hour']    = int(header[24:26]);
            temp.loc[record, 'reltime'] = int(header[27:31]);
            temp.loc[record, 'numlev']  = int(header[32:36]);
            temp.loc[record, 'lat']     = int(header[55:62])/10000;
            temp.loc[record, 'lon']     = int(header[63:71])/10000;
            
            # add title
            rh.loc[record, 'ID':'lon'] = temp.loc[record, 'ID':'lon']
            dpdp.loc[record, 'ID':'lon'] = temp.loc[record, 'ID':'lon']
            wdir.loc[record, 'ID':'lon'] = temp.loc[record, 'ID':'lon']
            wspd.loc[record, 'ID':'lon'] = temp.loc[record, 'ID':'lon']
            gph.loc[record, 'ID':'lon'] = temp.loc[record, 'ID':'lon']
            
            for i in range(NUMLEV):
                line = f.readline()
                LVLTYPE= line[0:2]
                ETIME  = line[3:8]
                PRES   = int(line[9:15])/100 # hPa
                PFLAG  = line[15]
                GPH    = int(line[16:21])
                ZFLAG  = line[21]
                TEMP   = int(line[22:27])/10 # C
                TFLAG  = line[27]
                RH     = int(line[28:33])/10 # %
                DPDP   = int(line[34:39])/10 # C
                WDIR   = int(line[40:45])
                WSPD   = int(line[46:51])/10 # m/s
                
                    
                if PRES<-80:
                    PRES = np.nan
                if TEMP<-80:
                    TEMP = np.nan
                if RH <-80:
                    RH=np.nan
                if DPDP < -80:
                    DPDP = np.nan
                if WDIR < -80:
                    WDIR = np.nan
                if WSPD < -80:
                    WSPD = np.nan
                if GPH < -80:
                    GPH = np.nan
                    
                if (LVLTYPE[0]=='3') | (PRES<600): # non-pressure level
                    continue
                else:
                    temp.loc[record, str(i)] = str(PRES)+','+str(TEMP)
                    dpdp.loc[record,  str(i)] = str(PRES)+','+str(DPDP)
                    rh.loc[record,  str(i)] = str(PRES)+','+str(RH )
                    wdir.loc[record,  str(i)] = str(PRES)+','+str(WDIR)
                    wspd.loc[record,  str(i)] = str(PRES)+','+str(WSPD)
                    gph.loc[record,  str(i)] = str(PRES)+','+str(GPH)
                    
            record += 1 
      
    
    return temp, rh, dpdp, wdir, wspd, gph
