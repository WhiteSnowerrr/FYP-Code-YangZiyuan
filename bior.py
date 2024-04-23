#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 01:45:59 2023

@author: yangziyuan
"""

import biobase
import pandas as pd

def bioRdata():
    out = list()
    datas = biobase.biodata('all','all')
    a = ['mode_2023','driving_time', 'driving_distance', 'pt_time', 
         'pt_distance', 'pt_waiting', 'pt_walking', 'pt_transfer','block']
    c=['pt1_wait', 'pt1_walk', 'pt1_tt', 'pt1_cost', 
    'pt1_trans', 'pt1_crowd', 'cs2_walk', 'cs2_tt', 'cs2_cost', 'cs2_disin', 
    'rs3_wait', 'rs3_tt', 'rs3_cost', 'rs3_share']
    pt = 'publicTrans'
    cs = 'carSharing'
    rs = 'rideSharing'
    name = {'pt1_':pt,'cs2_':cs,'rs3_':rs}

    for data in datas:
        oldName = list(data.columns)
        newName = [i.strip().lower() for i in oldName]

        data = data.rename(columns = dict(zip(oldName,newName)))
        data = data.rename(columns = dict(zip(['cs_1','cs_2','num'], 
                                              ['choice_best','choice_worst','id'])))
        data.drop(columns=a, inplace = True)
        data['age_s'] = data['age']/10
        data['qoffice_3_s'] = data['qoffice_3']/1
        temp = pd.DataFrame()
        for i in data.columns:
            if True in [(x in i) for x in c]:
                temp = pd.concat([temp, data[i]], axis = 1)
                data.drop(columns=i, inplace = True)
                for j in ['pt1_','cs2_','rs3_']:
                    if j in i:
                        temp = temp.rename(columns = {i:i.split(j)[1] + '.' + name[j]})
        for i in temp.columns:
            j = [x.split('.')[1] for x in [x for x in temp.columns if (i.split('.')[0] in x)]]
            if not pt in j:
                temp[i.split('.')[0] + '.' + pt] = 0
            if not cs in j:
                temp[i.split('.')[0] + '.' + cs] = 0
            if not rs in j:
                temp[i.split('.')[0] + '.' + rs] = 0
        
        data = pd.concat([data, temp], axis = 1)
        data['ch.'+pt] = 0
        data['ch.'+cs] = 0
        data['ch.'+rs] = 0
        for i in data.index:
            if data.loc[i,'choice_best']==1:
                data.loc[i,'ch.'+pt] = 1
            elif data.loc[i,'choice_best']==2:
                data.loc[i,'ch.'+cs] = 1
            else:
                data.loc[i,'ch.'+rs] = 1
            if data.loc[i,'choice_worst']==1:
                data.loc[i,'ch.'+pt] = 3
            elif data.loc[i,'choice_worst']==2:
                data.loc[i,'ch.'+cs] = 3
            else:
                data.loc[i,'ch.'+rs] = 3
            if data.loc[i,'ch.'+pt] == 0:
                data.loc[i,'ch.'+pt] = 2
            elif data.loc[i,'ch.'+cs] == 0:
                data.loc[i,'ch.'+cs] = 2
            else:
                data.loc[i,'ch.'+rs] = 2
        
        data['cost_s.'+pt] = data['cost.'+pt]/1000*(1+1/(data['income_2']/10000))
        data['cost_s.'+cs] = data['cost.'+cs]/1000*(1+1/(data['income_2']/10000))
        data['cost_s.'+rs] = data['cost.'+rs]/1000*(1+1/(data['income_2']/10000))
        
        data['tt_cs_s.'+pt] = 0
        data['tt_cs_s.'+cs] = data['commuting_days']*4*(data['tt.'+pt]-data['tt.'+cs])/1000
        data['tt_cs_s.'+rs] = 0
        
        data['tt_rs_s.'+pt] = 0
        data['tt_rs_s.'+cs] = 0
        data['tt_rs_s.'+rs] = data['commuting_days']*4*(data['tt.'+pt]-data['tt.'+rs])/1000
        
        data['crowd_s.'+pt] = data['crowd.'+pt]/10
        data['crowd_s.'+cs] = 0
        data['crowd_s.'+rs] = 0
        
        data['trans_s.'+pt] = data['trans.'+pt]/1
        data['trans_s.'+cs] = 0
        data['trans_s.'+rs] = 0
        
        data['share_s.'+pt] = 0
        data['share_s.'+cs] = 0
        data['share_s.'+rs] = data['share.'+rs]/10
        
        data['disin_s.'+pt] = 0
        data['disin_s.'+cs] = data['disin.'+cs]/1
        data['disin_s.'+rs] = 0
        
        for i in data.index:
            for j in ['choice_best','choice_worst']:
                if data.loc[i,j]==1:
                    data.loc[i,j] = pt
                elif data.loc[i,j]==2:
                    data.loc[i,j] = cs
                else:
                    data.loc[i,j] = rs
        out.append(data)
    
    return(tuple(out))



def apoRdata():
    out = list()
    datas = biobase.biodata('all','all')
    for data in datas:
        oldName = list(data.columns)
        newName = [i.strip().lower() for i in oldName]

        data = data.rename(columns = dict(zip(oldName,newName)))
        data = data.rename(columns = dict(zip(['cs_1','cs_2'], 
                                              ['choice_best','choice_worst'])))
        count = 0
        for i in data.index:
            data.loc[i,'ID'] = count//8+1
            count += 1
            if data.loc[i,'mode_2023'] == 2:
                data.loc[i,'mode_2023'] = 0
        out.append(data)
    return(tuple(out))




def poRdata():
    out = list()
    datas = biobase.biodata('all','all')
    for data in datas:
        oldName = list(data.columns)
        newName = [i.strip().lower() for i in oldName]

        data = data.rename(columns = dict(zip(oldName,newName)))
        data = data.rename(columns = dict(zip(['cs_1','cs_2'], 
                                              ['choice_best','choice_worst'])))
        count = 0
        for i in data.index:
            if data.loc[i,'mode_2023'] == 2:
                data.loc[i,'mode_2023'] = 0
            data.loc[i,'ID'] = count//8+1
            
            if count%8!=0:
                data.drop(i, inplace = True)
            count += 1
            
        out.append(data)
    return(tuple(out))


