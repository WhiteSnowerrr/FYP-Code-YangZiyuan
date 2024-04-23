#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:06:18 2023

@author: yangziyuan
"""

#%% import biogeme 3.2.12 -must!!!

import pandas as pd
import numpy as np


#%% dataProcess

def trainTestSplit(train_data, test_size=0.2, random_state=None, shuffle=False):
    import random
    if random_state != None:
        random.seed(random_state)
    totalNum = int(len(train_data)/8)
    testNum = int(totalNum*test_size)
    totalList = []
    for i in train_data.index:
        if i not in totalList and i%8 == 0:
            totalList.append(i)
    testList = random.sample(totalList, testNum)
    testList.sort()
    
    train_data_out = train_data.copy()
    test_data_out = train_data.copy()
    for i in train_data.index:
        if i//8*8 in testList:
            train_data_out.drop(i, inplace = True)
        else:
            test_data_out.drop(i, inplace = True)
    
    if shuffle:
        from sklearn.utils import shuffle
        train_data_out = shuffle(train_data_out, random_state=random_state)
        test_data_out = shuffle(test_data_out, random_state=random_state)
    
    return(train_data_out,test_data_out)


def evenData(Data1,Data2,what=None):
    import random
    random.seed(20020828)
    if len(Data2)>len(Data1):
        h = 1
    else:
        h = -1
    if what == None:
        totalList = []
        for i in Data1.index:
            if i not in totalList and i%8 == 0:
                totalList.append(i)
        if h == 1:
            addList = random.sample(totalList, 
                                    int(round((len(Data2)/len(Data1)-1)*len(totalList),0)))
            Data3 = Data1.copy()
            for j in addList:
                temp3 = Data1.loc[j:j+7].copy()
                temp3.loc[:,'id'] = range(max(Data3.index)+1,max(Data3.index)+9)
                temp3.set_index('id', inplace=True, verify_integrity=False)
                Data3 = Data3._append(temp3, ignore_index = False)
            return(Data3)
        else:
            addList = random.sample(totalList, 
                                    int(round((len(Data2)/len(Data1))*len(totalList),0)))
            temp2 = 0
            for j in addList:
                temp3 = Data1.loc[j:j+7].copy()
                if temp2 == 0:
                    Data3 = temp3.copy()
                    temp2 = 1
                else:
                    Data3 = Data3._append(temp3, ignore_index = False)
                    
            return(Data3)
        
    Data = Data1[what]

    Fre=Data.value_counts()
    Fre_sort=Fre.sort_index(axis=0,ascending=True)
    Fre_df1=Fre_sort.reset_index()
    Fre_df1.columns=[what,'Fre1']
    Data = Data2[what]

    Fre=Data.value_counts()
    Fre_sort=Fre.sort_index(axis=0,ascending=True)
    Fre_df2=Fre_sort.reset_index()
    Fre_df2.columns=[what,'Fre2']
    Fre_df = pd.merge(Fre_df1,Fre_df2,how='outer',on=what)
    Fre_df.sort_values(by=what , inplace=True, ascending=True) 
    Fre_df.reset_index(drop=True, inplace=True)
    
    while True:
        ok = 1
        for i in Fre_df.index:
            if np.isnan(Fre_df['Fre2'][i]):
                ok = 0
                a = i-1

                if a>=0:
                    if not np.isnan(Fre_df['Fre2'][a]):
                        Fre_df.loc[a,'Fre2'] = Fre_df.loc[a,'Fre2']/2
                        a = Fre_df['Fre2'][a]
                    else:
                        a = -999
                else:
                    a = -999
                b = i+1
                if b<=max(Fre_df.index):
                    if not np.isnan(Fre_df['Fre2'][b]):
                        Fre_df.loc[b,'Fre2'] = Fre_df.loc[b,'Fre2']/2
                        b = Fre_df['Fre2'][b]
                    else:
                        b = -999
                else:
                    b = -999
                if a != -999 and b != -999:
                    Fre_df.loc[i,'Fre2'] = (a+b)
                elif a != -999:
                    Fre_df.loc[i,'Fre2'] = (a)
                elif b != -999:
                    Fre_df.loc[i,'Fre2'] = (b)
                
        if ok == 1:
            break
    
    
    for i in Fre_df.index:
        if np.isnan(Fre_df['Fre1'][i]):
            a = i-1
            aa = 1
            if a>=0:
                while np.isnan(Fre_df['Fre1'][a]) and a>=0:
                    a = a-1
                    if not a>=0:
                        aa = 0
                        break
            else:
                aa = 0
            b = i+1
            bb = 1
            if b<=max(Fre_df.index):
                while np.isnan(Fre_df['Fre1'][b]) and b<=max(Fre_df.index):
                    b = b+1
                    if not b<=max(Fre_df.index):
                        bb = 0
                        break
            else:
                bb = 0
            if aa == 1 and bb == 1:
                Fre_df.loc[a,'Fre2'] += Fre_df.loc[i,'Fre2']/2
                Fre_df.loc[b,'Fre2'] += Fre_df.loc[i,'Fre2']/2
            elif aa == 1:
                Fre_df.loc[a,'Fre2'] += Fre_df.loc[i,'Fre2']
            else:
                Fre_df.loc[b,'Fre2'] += Fre_df.loc[i,'Fre2']
            Fre_df.loc[i,'Fre2'] = 0
            
    Fre_df['d'] = Fre_df['Fre2']-Fre_df['Fre1']
    
    temp = Fre_df['d'].dropna()
    if h == 1:
        a = sum(temp[temp>=0])
    else:
        a = sum(temp[temp<=0])
    
    b = len(Data2)-len(Data1)
    
    Fre_df['d2'] = 0

    temp2 = b/a
    for i in Fre_df.index:
        if not np.isnan(Fre_df['d'][i]):
            if Fre_df['d'][i]*h >= 0:
                Fre_df.loc[i,'d2'] = round((Fre_df['d'][i]*temp2)/8,0)*8
    
    if h == 1:
        Data3 = Data1.copy()
        for i in Fre_df.index:
            if Fre_df['d2'][i] > 0:
                what2 = Fre_df[what][i]
                temp = Data1.loc[Data1[what] == what2]
                ilist = list(set((temp.index)-(temp.index)%8))
                for j in range(0,int(Fre_df['d2'][i]/8)):
                    temp2 = ilist[j%len(ilist)]
                    temp3 = Data1.loc[temp2:temp2+7].copy()
                    temp3.loc[:,'id'] = range(max(Data3.index)+1,max(Data3.index)+9)
                    temp3.set_index('id', inplace=True, verify_integrity=False)
                    Data3 = Data3._append(temp3, ignore_index = False)
        return(Data3)
    else:
        temp2 = 0
        for i in Fre_df.index:
            if not np.isnan(Fre_df['Fre1'][i]):
                what2 = Fre_df[what][i]
                temp = Data1.loc[Data1[what] == what2]
                ilist = list(set((temp.index)-(temp.index)%8))
                for j in random.sample(ilist, int((Fre_df['Fre1'][i]+Fre_df['d2'][i])/8)):
                    temp3 = Data1.loc[j:j+7].copy()
                    if temp2 == 0:
                        Data3 = temp3.copy()
                        temp2 = 1
                    else:
                        Data3 = Data3._append(temp3, ignore_index = False)
        return(Data3)
    

def outLinerRange(dataC):
    iQR = np.percentile(dataC,75) - np.percentile(dataC,25)
    lB = np.percentile(dataC,25) - 1.5*iQR
    uB = np.percentile(dataC,75) + 1.5*iQR
    return pd.Interval(lB,uB, closed='both')

def varname(p):
    import re
    import inspect
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)',line)
        if m:
            return m.group(1)

def cDF(Data):
    denominator=len(Data)#分母数量
    Fre=Data.value_counts()
    Fre_sort=Fre.sort_index(axis=0,ascending=True)
    Fre_df=Fre_sort.reset_index()#将Series数据转换为DataFrame
    Fre_df.columns=['Rds','count']
    Fre_df['count2']=Fre_df['count']/denominator#转换成概率
    Fre_df.columns=['Rds','count','Fre']
    Fre_df['cumsum']=np.cumsum(Fre_df['Fre'])
    return(Fre_df)

# obj
a=['wfh allowed ', 'wfh now ','age', 'time living ', 'gender', 'edu', 
'license', 'income_1', 'income_2', 'Qhousehold_1', 'Qhousehold_2', 
'Qhousehold_3', 'Qhousehold_4', 'Qhousehold_5', 'floor area', 
'Qfamily_1', 'Qfamily_2', 'Qfamily_3', 'Qfamily_4', 'Qfamily_5', 
'Qfamily_5.1', 'Qfamily_7', 'Qoffice_1', 'Qoffice_2', 'Qoffice_3', 
'Qoffice_4', 'private car#1_1', 'private car#1_2', 
'rs experience_1', 'rs experience_2', 'rs experience_3', 
'pt experience _1', 'pt experience _2', 'pt experience _3']

# subj
b=['cs rs#1_1', 'cs rs#1_2', 'att pt_1', 'att pt_2', 'att pt_3', 
'att pt_4', 'att cs_1', 'att cs_2', 'att cs_3', 'att shift rs_1', 
'att shift rs_2', 'att shift rs_3']

# sp
c=['Block', 'commuting_days', 'pt1_wait', 'pt1_walk', 'pt1_tt', 'pt1_cost', 
'pt1_trans', 'pt1_crowd', 'cs2_walk', 'cs2_tt', 'cs2_cost', 'cs2_disin', 
'rs3_wait', 'rs3_tt', 'rs3_cost', 'rs3_share', 'cs_1', 'cs_2'
]

# others
e=['driving_time', 'driving_distance', 'PT_time', 'PT_distance', 'PT_waiting',
   'PT_walking', 'PT_transfer']


#%%
# data sep -use rspt6
data_temp = pd.read_pickle('data/rspt6.pickle')

df2 = data_temp['df']
travel2 = data_temp['travel']
del data_temp


'''
plt.boxplot(df2['PT_waiting'],
            medianprops={'color': 'red', 'linewidth': '1.5'},
            meanline=True,
            showmeans=True,
            meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
            boxprops = {'facecolor':'#b0e686'},
            patch_artist = True,
            )
'''

#%%

# delete data
for i in [1744,1520,1992]: #here
    t = i - i%8
    for ii in range(t,t+8):
        df2.drop(ii, inplace = True)
        travel2.drop(ii, inplace = True)
del i,t,ii

ee=['driving_time', 'driving_distance', 'PT_time', 'PT_distance', 'PT_waiting',
   'PT_walking']

for i in ee:
    rsI = []
    ptI = []
    for j in df2.index:
        if df2['mode 2023 '][j] == 3:
            rsI.append(j)
        else:
            ptI.append(j)
    for k in (rsI,ptI):
        lR = outLinerRange(df2[i][k])
        c = 0
        for j in k:
            if not df2[i][j] in lR:
                c += 1
                df2.drop(j, inplace = True)
                travel2.drop(j, inplace = True)
        #print(i,c)

'''
for i in ee:
    lR = outLinerRange(df2[i])
    c = 0
    for j in df2.index:
        if not df2[i][j] in lR:
            c += 1
            df2.drop(j, inplace = True)
            travel2.drop(j, inplace = True)
    #print(i,c)
'''



for i in ee:
    rsI = []
    ptI = []
    for j in df2.index:
        if df2['mode 2023 '][j] == 3:
            rsI.append(j)
        else:
            ptI.append(j)
    for k in (rsI,ptI):
        c = 0
        Fre_df = cDF(df2[i][k])
        for l in Fre_df.index:
            if l != 0:
                Fre_df.loc[l,'drd'] = Fre_df['Rds'][l] - Fre_df['Rds'][l-1]
            else:
                Fre_df.loc[l,'drd'] = Fre_df['Rds'][l]
        Fre_df.loc[:,'fdrd'] = Fre_df['drd'] / (max(Fre_df['Rds'])-min(Fre_df['Rds']))
        temp2 = k.copy()
        for l in list(reversed(list(Fre_df.index))):
            t1 = 8*Fre_df['Fre'][l] <= Fre_df['fdrd'][l]
            t2 = 8*Fre_df['Fre'][l-1] <= Fre_df['fdrd'][l-1]
            t3 = 8*Fre_df['Fre'][l-2] <= Fre_df['fdrd'][l-2]
            if t1 or t2 or t3:
                temp = Fre_df['Rds'][l]
                temp3 = temp2.copy()
                for m in temp2:
                    if df2[i][m] == temp:
                        c += 1
                        temp3.remove(m)
                        #print(m)
                        
                        df2.drop(m, inplace = True)
                        travel2.drop(m, inplace = True)
                temp2 = temp3.copy()
            else:
                break
        #print(i,c)


'''
for i in df2.index:
    try:
        if df2['PT_time'][i] > 60:
            t = i-i%8
            for ii in range(t,t+8):
                df2.drop(ii, inplace = True)
                travel2.drop(ii, inplace = True)
    except:
        pass


for i in df2.index:
    try:
        if df2['PT_walking'][i] > 40:
            t = i-i%8
            for ii in range(t,t+8):
                df2.drop(ii, inplace = True)
                travel2.drop(ii, inplace = True)
    except:
        pass
'''


for i in df2.index:
    if i%8 == 0 and len(set(df2['cs_1'][range(i,i+8)])) == 1 and len(set(df2['cs_2'][range(i,i+8)])) == 1:
        for ii in range(i,i+8):
            df2.drop(ii, inplace = True)
            travel2.drop(ii, inplace = True)




#%%

for i in df2.index:
    try:
        if df2['license'][i] < 2:
            df2.loc[i,'license'] = 1
        else:
            df2.loc[i,'license'] = 0
    except:
        pass


for i in df2.index:
    for j in ['att pt_1', 'att pt_2', 'att pt_3', 'att pt_4']:
        df2.loc[i,j] -= 2

'''
df2['ATT_PT_5'] = df2['att pt_1'] + df2['att pt_2'] + df2['att pt_3'] + df2['att pt_4']

for i in df2.index:
    if df2['ATT_PT_5'][i] < -1:
        df2.loc[i,'ATT_PT_5'] = -1
    elif df2['ATT_PT_5'][i] < 2:
        df2.loc[i,'ATT_PT_5'] = 0
    else:
        df2.loc[i,'ATT_PT_5'] = 1

df2['ATT_PT_6'] = df2['att pt_1'] + df2['att pt_2']

for i in df2.index:
    if df2['ATT_PT_6'][i] < 1:
        df2.loc[i,'ATT_PT_6'] = 0
    else:
        df2.loc[i,'ATT_PT_6'] = 1

df2['ATT_PT_7'] = df2['att pt_3']

for i in df2.index:
    if df2['ATT_PT_7'][i] < 1:
        df2.loc[i,'ATT_PT_7'] = 1
    else:
        df2.loc[i,'ATT_PT_7'] = 2
'''

'''
ff=['pt1_tt', 'pt1_cost', 
'cs2_tt', 'cs2_cost', 
'rs3_tt', 'rs3_cost'
]


for i in ff:
    rsI = []
    ptI = []
    for j in df2.index:
        if df2['mode 2023 '][j] == 3:
            rsI.append(j)
        else:
            ptI.append(j)
    upi = i.upper()
    i1 = (np.percentile(df2[i][rsI],50) + np.percentile(df2[i][ptI],50))/2
    i2 = (np.percentile(df2[i][rsI],75) + np.percentile(df2[i][ptI],75))/2
    
    for j in df2.index:
        df2.loc[j,upi+'_1'] = min(df2[i][j],i1)
        df2.loc[j,upi+'_2'] = max(0,min((df2[i][j]-i1),i2-i1))
        df2.loc[j,upi+'_3'] = max((df2[i][j]-i2),0)
'''


'''
for i in df2.index:
    if i%8 == 0 and len(set(df2['cs_1'][range(i,i+8)])) == 1:
        for ii in range(i,i+8):
            df2.drop(ii, inplace = True)
            travel2.drop(ii, inplace = True)


for i in df2.index:
    if i%8 == 0 and len(set(df2['cs_2'][range(i,i+8)])) == 1:
        for ii in range(i,i+8):
            df2.drop(ii, inplace = True)
            travel2.drop(ii, inplace = True)
'''



#%%% biodata

def biodata(row,col,traveldata=False,df2=df2,travel2=travel2,a=a,b=b,c=c,e=e):  
#row-choose row data:rs/pt/all
#col-choose col data:obj/subj/sp/all
    if row == 'pt':
        d = 3
    elif row == 'rs':
        d = 4
    elif row == 'all':
        d = 0
    else:
        print('error row')
        
    if col == 'obj':
        dd = 0
    elif col == 'subj':
        dd = 1
    elif col == 'sp':
        dd = 2
    elif col == 'all':
        dd = 3
    else:
        print('error col')
    
    df = df2.copy()
    travel = travel2.copy()
    if dd == 0:
        df.drop(columns=b+c+e, inplace = True)
        df['RS_AV'] = 1
        df['PT_AV'] = 1
    elif dd==1:
        df.drop(columns=a+c+e, inplace = True)
        df['RS_AV'] = 1
        df['PT_AV'] = 1
    elif dd==2:
        df.drop(columns=a+b+e, inplace = True)
        '''
        df['CHOICE'] = 0
        for i in df.index:
            if df['cs_1'][i] == 1 and df['cs_2'][i] == 2:
                df.loc[i,'CHOICE'] = 1
            elif df['cs_1'][i] == 1 and (df['cs_2'][i] == 3 or df['cs_2'][i] == 4):
                df.loc[i,'CHOICE'] = 2
            elif df['cs_1'][i] == 2 and df['cs_2'][i] == 1:
                df.loc[i,'CHOICE'] = 3
            elif df['cs_1'][i] == 2 and (df['cs_2'][i] == 3 or df['cs_2'][i] == 4):
                df.loc[i,'CHOICE'] = 4
            elif (df['cs_1'][i] == 3 or df['cs_1'][i] == 4) and df['cs_2'][i] == 1:
                df.loc[i,'CHOICE'] = 5
            elif (df['cs_1'][i] == 3 or df['cs_1'][i] == 4) and df['cs_2'][i] == 2:
                df.loc[i,'CHOICE'] = 6
        del i
        
    elif dd==3:
        df['CHOICE'] = 0
        for i in df.index:
            if df['cs_1'][i] == 1 and df['cs_2'][i] == 2:
                df.loc[i,'CHOICE'] = 1
            elif df['cs_1'][i] == 1 and (df['cs_2'][i] == 3 or df['cs_2'][i] == 4):
                df.loc[i,'CHOICE'] = 2
            elif df['cs_1'][i] == 2 and df['cs_2'][i] == 1:
                df.loc[i,'CHOICE'] = 3
            elif df['cs_1'][i] == 2 and (df['cs_2'][i] == 3 or df['cs_2'][i] == 4):
                df.loc[i,'CHOICE'] = 4
            elif (df['cs_1'][i] == 3 or df['cs_1'][i] == 4) and df['cs_2'][i] == 1:
                df.loc[i,'CHOICE'] = 5
            elif (df['cs_1'][i] == 3 or df['cs_1'][i] == 4) and df['cs_2'][i] == 2:
                df.loc[i,'CHOICE'] = 6
        del i
    '''

    df.drop(columns=['Unnamed: 0',
        'home_2019_lat', 'home_2019_lng', 'work_2019_lat', 'work_2019_lng', 
        'home_now_lat', 'home_now_lng', 'work_now_lat', 'work_now_lng'], 
        inplace = True)
    
    oldName = list(df.columns)
    newName = [i.strip().upper().replace(' _','_').replace(' ','_').replace('#','_')
               for i in oldName]

    df = df.rename(columns = dict(zip(oldName,newName)))
    
    for i in df.index:
        if df['CS_1'][i] == 4:
            df.loc[i,'CS_1'] = 3
        if df['CS_2'][i] == 4:
            df.loc[i,'CS_2'] = 3
                    
    
    if d != 0:
        for i in travel.index:
            if df['MODE_2023'][i] == d:
                df.drop(i, inplace = True)
                travel.drop(i, inplace = True)
        for i in travel.index:
            if df['MODE_2023'][i] == 3:
                df.loc[i,'MODE_2023'] = 1 #rs
            else:
                df.loc[i,'MODE_2023'] = 2 #pt
        if traveldata:
            return (df,travel)
        else:
            return (df)
                
    else:
        dfrs = df.copy()
        dfpt = df.copy()
        for i in travel.index:
            if dfrs['MODE_2023'][i] == 4:
                dfrs.drop(i, inplace = True)
            else:
                dfrs.loc[i,'MODE_2023'] = 1
        for i in travel.index:
            if dfpt['MODE_2023'][i] == 3:
                dfpt.drop(i, inplace = True)
            else:
                dfpt.loc[i,'MODE_2023'] = 2
        del i
        for i in travel.index:
            if df['MODE_2023'][i] == 3:
                df.loc[i,'MODE_2023'] = 1 #rs
            else:
                df.loc[i,'MODE_2023'] = 2 #pt
        if traveldata:
            return (df,dfrs,dfpt,travel)
        else:
            return (df,dfrs,dfpt)
    

    
    
    
    