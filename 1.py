#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 16:10:10 2023

@author: yangziyuan
"""

#%% import biogeme 3.2.12 -must!!!


import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, bioNormalCdf, log, Elem
import matplotlib.pyplot as plt
import math as m
import numpy as np


# rewrite the output function to chose the output path
import biogeme.results
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
def newgetNewFileName(name, ext, path=''):
    fileName = name + '.' + ext
    theFile = Path(path+fileName)
    number = int(0)
    while theFile.is_file():
        fileName = f'{name}~{number:02d}.{ext}'
        theFile = Path(path+fileName)
        number += 1
    return fileName
def newwriteHtml(t, onlyRobust=True, path=''):
    """Write the results in an HTML file."""
    t.data.htmlFileName = newgetNewFileName(t.data.modelName, 'html', path)
    fileplace = path+t.data.htmlFileName
    with open(fileplace, 'w', encoding='utf-8') as f:
        f.write(t.getHtml(onlyRobust))
    logger.info(f'Results saved in file {t.data.htmlFileName}')    

del logger

import seaborn as sns
from scipy.stats import pearsonr
def reg_coef(x,y,label=None,color=None, **kwargs):
    ax = plt.gca()
    r,p = pearsonr(x,y)
    if p < 0.01:
        sig_level = '***'
    elif p < 0.05:
        sig_level = '**'
    elif p < 0.05:
        sig_level = '*'
    else:
        sig_level = ''
        
    ax.annotate('r = {:.2f} {}'.format(r, sig_level), xy=(0.5,0.5), xycoords='axes fraction', ha='center')
    ax.texts[0].set_size(16)
    ax.set_axis_off()

#%% read data -done


df = pd.read_excel(
    'data/FYP_RSPT.xlsx'
)



#%% caculating distance -done

'''
temp = []
for i in range(0,df.shape[0]):
    temp.append(6378.009 * m.acos(
    m.sin(eval(df['home 2019'][i])['0']['lng']) * m.sin(eval(df['work 2019'][i])['0']['lng']) + 
    m.cos(eval(df['home 2019'][i])['0']['lng']) * m.cos(eval(df['work 2019'][i])['0']['lng']) * 
    m.cos(eval(df['home 2019'][i])['0']['lat']-eval(df['work 2019'][i])['0']['lat'])
    ))
df['dis_2019'] = temp
temp = []
for i in range(0,df.shape[0]):
    temp.append(6371.009 * m.acos(
    m.sin(eval(df['home now'][i])['0']['lng']) * m.sin(eval(df['work now'][i])['0']['lng']) + 
    m.cos(eval(df['home now'][i])['0']['lng']) * m.cos(eval(df['work now'][i])['0']['lng']) * 
    m.cos(eval(df['home now'][i])['0']['lat']-eval(df['work now'][i])['0']['lat'])
    ))
df['dis_now'] = temp
del temp,i'''

'''
from geopy import distance
temp = []

for i in range(0,df.shape[0]):
    # Geopy库先lat后long
    old=(eval(df['home 2019'][i])['0']['lat'],eval(df['home 2019'][i])['0']['lng'])
    new=(eval(df['work 2019'][i])['0']['lat'],eval(df['work 2019'][i])['0']['lng'])
    d=distance.distance(old, new).km
    s=round(float(d),2)
    temp.append(s)
df['dis_2019'] = temp
temp = []
from geopy import distance
for i in range(0,df.shape[0]):
    # Geopy库先lat后long
    old=(eval(df['home now'][i])['0']['lat'],eval(df['home now'][i])['0']['lng'])
    new=(eval(df['work now'][i])['0']['lat'],eval(df['work now'][i])['0']['lng'])
    d=distance.distance(old, new).km
    s=round(float(d),2)
    temp.append(s)
df['dis_now'] = temp
del temp,i,old,new,d,s

df.drop(columns=[
    'home now', 'home 2019', 'work now', 'work 2019'], 
    inplace = True)'''


import googlemaps
import datetime

gmaps = googlemaps.Client(key = 'AIzaSyBPiy9MLjzSiF1E2l-RHzmDS7AaOyXaQtQ')
now = datetime.datetime.strptime('2023/08/28 8:00:00', '%Y/%m/%d %H:%M:%S')

driving2019 = []
pt2019 = []
for i in range(0,df.shape[0]):
    old=str(eval(df['home 2019'][i])['0']['lat']) + ',' + str(eval(df['home 2019'][i])['0']['lng'])
    new=str(eval(df['work 2019'][i])['0']['lat']) + ',' + str(eval(df['work 2019'][i])['0']['lng'])

    directions_result = gmaps.directions(old,new,mode = 'driving', 
                                         avoid = 'ferries',
                                         traffic_model="optimistic",
                                         departure_time = now)
    directions_result2 = gmaps.directions(old,new,mode = 'transit', 
                                         avoid = 'ferries',
                                         traffic_model="optimistic",
                                         departure_time = now)
    if directions_result == []:
        directions_result = gmaps.directions(old,new,mode = 'walking', 
                                             avoid = 'ferries',
                                             traffic_model="optimistic",
                                             departure_time = now)
    if directions_result2 == []:
        directions_result2 = gmaps.directions(old,new,mode = 'walking', 
                                             avoid = 'ferries',
                                             traffic_model="optimistic",
                                             departure_time = now)
    if i == 2250:
        print('25%')
    driving2019 = driving2019 + directions_result
    pt2019 = pt2019 + directions_result2
del i,old,new,directions_result,directions_result2
print('50%')
driving2023 = []
pt2023 = []
for i in range(0,df.shape[0]):
    old=str(eval(df['home now'][i])['0']['lat']) + ',' + str(eval(df['home now'][i])['0']['lng'])
    new=str(eval(df['work now'][i])['0']['lat']) + ',' + str(eval(df['work now'][i])['0']['lng'])

    directions_result = gmaps.directions(old,new,mode = 'driving', 
                                         avoid = 'ferries',
                                         traffic_model="optimistic",
                                         departure_time = now)
    directions_result2 = gmaps.directions(old,new,mode = 'transit', 
                                         avoid = 'ferries',
                                         traffic_model="optimistic",
                                         departure_time = now)
    if directions_result == []:
        directions_result = gmaps.directions(old,new,mode = 'walking', 
                                             avoid = 'ferries',
                                             traffic_model="optimistic",
                                             departure_time = now)
    if directions_result2 == []:
        directions_result2 = gmaps.directions(old,new,mode = 'walking', 
                                             avoid = 'ferries',
                                             traffic_model="optimistic",
                                             departure_time = now)
    if i == 2250:
        print('75%')
    driving2023 = driving2023 + directions_result
    pt2023 = pt2023 + directions_result2
del i,old,new,directions_result,directions_result2


import csv #调用数据保存文件
writer = pd.ExcelWriter('directions_result.xlsx')
test1=pd.DataFrame(data=pt2019)
test2=pd.DataFrame(data=driving2019)
test3=pd.DataFrame(data=pt2023)
test4=pd.DataFrame(data=driving2023)
test1.to_excel(writer,sheet_name='pt2019')
test2.to_excel(writer,sheet_name='driving2019')
test3.to_excel(writer,sheet_name='pt2023')
test4.to_excel(writer,sheet_name='driving2023')
writer.save()
writer.close()
del test1,test2,test3,test4,writer

travel = pd.DataFrame()
travel['driving_2019'] = driving2019
travel['pt_2019'] = pt2019
travel['driving_now'] = driving2023
travel['pt_now'] = pt2023
del driving2019,pt2019,driving2023,pt2023
''' travel['driving 2019'][0]['legs'][0]['distance']['value']
travel['driving 2019'][0]['legs'][0]['duration']['value']
'''


#%% Removing some observations -done
df.drop(columns=[
    'timer_First Click', 'timer_Last Click', 'timer_Page Submit', 
    'timer_Click Count', 'Duration (in seconds)', 'voluntary ', 
    'full-time 2019', 'full-time 2023', 'mode 2019 ', 
    'att shift cs_1', 'att shift cs_2', 'att shift cs_3', 
    'att shift cs_4', 'share', 'Choice situation', 
    'Number of commuting trips', 'pt.traveltime', 'pt.waittime', 
    'pt.walktime', 'pt.totalcost', 'pt.crowding', 'cs.traveltime', 
    'cs.walktime', 'cs.totalcost', 'cs.disinfection', 
    'rp.traveltime', 'rp.waittime', 'rp.totalcost', 
    'cs experience_1', 'cs experience_2', 'cs experience_3'
    'device'], 
    inplace = True)

# Fill nans
df.fillna(10, inplace = True)

# Removing some observations
df['home_2019_lat'] = ''
df['home_2019_lng'] = ''
df['work_2019_lat'] = ''
df['work_2019_lng'] = ''
df['home_now_lat'] = ''
df['home_now_lng'] = ''
df['work_now_lat'] = ''
df['work_now_lng'] = ''
for i in range(0,df.shape[0]):
    df['home_2019_lat'][i] = eval(df['home 2019'][i])['0']['lat']
    df['home_2019_lng'][i] = eval(df['home 2019'][i])['0']['lng']
    df['work_2019_lat'][i] = eval(df['work 2019'][i])['0']['lat']
    df['work_2019_lng'][i] = eval(df['work 2019'][i])['0']['lng']
    df['home_now_lat'][i] = eval(df['home now'][i])['0']['lat']
    df['home_now_lng'][i] = eval(df['home now'][i])['0']['lng']
    df['work_now_lat'][i] = eval(df['work now'][i])['0']['lat']
    df['work_now_lng'][i] = eval(df['work now'][i])['0']['lng']
df.drop(columns=[
    'home 2019', 'work 2019',
    'home now', 'work now'], 
    inplace = True)
del i



#%% data cleaning -use rspt5 -done

# show the travel time frequency distribution 
def tu(what, a=travel,oo = 3600):
    tt = []
    t = 0
    for i in a.index:
        t = t + 1
        if t%8 == 1:
            tt.append(a[what][i]['legs'][0]['duration']['value']/oo)
    plt.figure() #初始化一张图
    x = tt
    if oo == 30:
        width = 30
    else:
        width = 40
    n, bins, patches = plt.hist(x,bins = width,range=(0,width),color='#b0e686',alpha=0.5)
    plt.grid(alpha=0.5,linestyle='-.') #网格线，更好看 
    if oo == 30:
        plt.xlabel('Travel Time /Minutes')  
    else:
        plt.xlabel('Travel Time /Hours')  
    plt.ylabel('Number of Data'+'('+what+')')  
    plt.title(r'Travel time frequency distribution histogram of survey data') #+citys[i])  
    plt.xticks(np.arange(0,width,2))
    plt.plot(bins[0:width]+((bins[1]-bins[0])/2.0),n,color='red')#利用返回值来绘制区间中点连线
    plt.show()
    
# boxplot picture
def bp(oo):
    labels = 'pt_2019', 'pt_now', 'driving_2019', 'driving_now'
    A = []
    B = []
    C = []
    D = []
    t = 0
    for i in travel.index:
        t = t + 1
        if t%8 == 1:
            A.append(travel['pt_2019'][i]['legs'][0]['duration']['value']/60)
            B.append(travel['pt_now'][i]['legs'][0]['duration']['value']/60)
            C.append(travel['driving_2019'][i]['legs'][0]['duration']['value']/60)
            D.append(travel['driving_now'][i]['legs'][0]['duration']['value']/60)
    plt.grid(True)  # 显示网格
    plt.boxplot([A, B, C, D],
                medianprops={'color': 'red', 'linewidth': '1.5'},
                meanline=True,
                showmeans=True,
                meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
                boxprops = {'facecolor':'#b0e686'},
                patch_artist = True,
                labels=labels)
    plt.yticks(np.arange(0, oo, 30))
    plt.ylabel('Travel Time /Minutes')  
    plt.title(r'Travel time frequency distribution boxplot of survey data')
    plt.show()

# 0:rspt5
df.to_csv('/Users/yangziyuan/Documents/学/大四上/fyp/data/0.csv')

'''
# delete invalid location: same travel start/end
for i in travel.index:
    if travel['driving_2019'][i]['legs'][0]['start_location'] == travel['driving_2019'][i]['legs'][0]['end_location']:
        print('!!!!!')
    if travel['pt_2019'][i]['legs'][0]['start_location'] == travel['pt_2019'][i]['legs'][0]['end_location']:
        print('!!!!!')
    if travel['driving_now'][i]['legs'][0]['start_location'] == travel['driving_now'][i]['legs'][0]['end_location']:
        print('!!!!!')
    if travel['pt_now'][i]['legs'][0]['start_location'] == travel['pt_now'][i]['legs'][0]['end_location']:
        print('!!!!!')
'''

tu('pt_2019')

# 1:delete invalid location: not in singapore
for i in travel.index:
    t1 = travel['driving_2019'][i]['legs'][0]['start_address']
    t2 = travel['driving_2019'][i]['legs'][0]['end_address']
    t3 = travel['driving_now'][i]['legs'][0]['start_address']
    t4 = travel['driving_now'][i]['legs'][0]['end_address']
    if ('Malaysia' in t1) or ('Malaysia' in t2) or ('Malaysia' in t3) or ('Malaysia' in t4):
        df.drop(i, inplace = True)
        travel.drop(i, inplace = True)
del i,t1,t2,t3,t4
df.to_csv('/Users/yangziyuan/Documents/学/大四上/fyp/data/1.csv')

# 2:delete invalid location: in the water
for i in [71,1911,2055,2063,511]: #here
    t = i - i%8
    for ii in range(t,t+8):
        df.drop(ii, inplace = True)
        travel.drop(ii, inplace = True)
del i,t,ii
df.to_csv('/Users/yangziyuan/Documents/学/大四上/fyp/data/2.csv')
tu('pt_2019')

tu('pt_2019',oo=30)
# 3:delete invalid location: same home/work
for i in df.index:
    if (df['home_2019_lat'][i] == df['work_2019_lat'][i]) and (df['home_2019_lng'][i] == df['work_2019_lng'][i]):
        df.drop(i, inplace = True)
        travel.drop(i, inplace = True)
        continue
    if (df['home_now_lat'][i] == df['work_now_lat'][i]) and (df['home_now_lng'][i] == df['work_now_lng'][i]):
        df.drop(i, inplace = True)
        travel.drop(i, inplace = True)
        continue
del i
tu('pt_2019',oo=30)
df.to_csv('/Users/yangziyuan/Documents/学/大四上/fyp/data/3.csv')

bp(oo=210) #outliners: pt time > 90mins
# 4:delete outliners
for i in travel.index:
    t1 = travel['pt_2019'][i]['legs'][0]['duration']['value']/60
    t2 = travel['pt_now'][i]['legs'][0]['duration']['value']/60
    if (t1 > 90) or (t2 > 90):
        df.drop(i, inplace = True)
        travel.drop(i, inplace = True)
del i,t1,t2
bp(oo = 120)
df.to_csv('/Users/yangziyuan/Documents/学/大四上/fyp/data/4.csv')


'''
travel['driving_2019'][0]['legs'][0]['start_location']
driving2019.drop(1)
'''




#%% data sep -use rspt6
d = 0 # choose row data:0=all,4=rs,3=pt
dd = 1 # choose col data:0=obj,1=subj,2=sp,3=all

# delete data
for i in [1744]: #here
    t = i - i%8
    for ii in range(t,t+8):
        df.drop(ii, inplace = True)
        travel.drop(ii, inplace = True)
del i,t,ii

#%%%
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
'rs3_wait', 'rs3_tt', 'rs3_cost', 'rs3_share', 'cs_1', 'cs_2']

# others
e=['driving_time', 'driving_distance', 'PT_time', 'PT_distance', 'PT_waiting',
   'PT_walking', 'PT_transfer']


if dd == 0:
    df.drop(columns=b+c+e, inplace = True)
elif dd==1:
    df.drop(columns=a+c+e, inplace = True)
elif dd==2:
    df.drop(columns=a+b+e, inplace = True)
del dd,a,b,c,e

df.drop(columns=['Unnamed: 0',
    'home_2019_lat', 'home_2019_lng', 'work_2019_lat', 'work_2019_lng', 
    'home_now_lat', 'home_now_lng', 'work_now_lat', 'work_now_lng'], 
    inplace = True)

df = df.rename(columns = {'wfh allowed ':'WFH_ALLOWED', 'wfh now ':'WFH_NOW', 
    'age':'AGE', 'time living ':'TIME_LIVING', 'gender':'GENDER', 'edu':'EDU', 
    'license':'LICENSE', 'income_1':'INCOME_1', 'income_2':'INCOME_2', 
    'Qhousehold_1':'QHOUSEHOLD_1', 'Qhousehold_2':'QHOUSEHOLD_2', 
    'Qhousehold_3':'QHOUSEHOLD_3', 'Qhousehold_4':'QHOUSEHOLD_4', 
    'Qhousehold_5':'QHOUSEHOLD_5', 'floor area':'FLOOR_AREA', 
    'Qfamily_1':'QFAMILY_1', 'Qfamily_2':'QFAMILY_2', 'Qfamily_3':'QFAMILY_3', 
    'Qfamily_4':'QFAMILY_4', 'Qfamily_5':'QFAMILY_5', 'Qfamily_7':'QFAMILY_7', 
    'Qfamily_5.1':'QFAMILY_5_1', 'Qoffice_1':'QOFFICE_1', 
    'Qoffice_2':'QOFFICE_2', 'Qoffice_3':'QOFFICE_3', 'Qoffice_4':'QOFFICE_4', 
    'cs rs#1_1':'CSRS_1_1', 'cs rs#1_2':'CSRS_1_2', 
    'private car#1_1':'PRIVATE_CAR_1_1', 'private car#1_2':'PRIVATE_CAR_1_2', 
    'rs experience_1':'RS_EXPERIENCE_1', 'rs experience_2':'RS_EXPERIENCE_2', 
    'rs experience_3':'RS_EXPERIENCE_3', 
    'pt experience _1':'PT_EXPERIENCE_1', 'pt experience _2':'PT_EXPERIENCE_2', 
    'pt experience _3':'PT_EXPERIENCE_3', 
    'att pt_1':'ATT_PT_1', 'att pt_2':'ATT_PT_2', 'att pt_3':'ATT_PT_3', 
    'att pt_4':'ATT_PT_4', 'att cs_1':'ATT_CS_1', 'att cs_2':'ATT_CS_2', 
    'att cs_3':'ATT_CS_3', 'att shift rs_1':'ATT_SHIFT_RS_1', 
    'att shift rs_2':'ATT_SHIFT_RS_2', 'att shift rs_3':'ATT_SHIFT_RS_3', 
    'driving_time':'DRIVING_TIME', 'driving_distance':'DRIVING_DISTANCE', 
    'PT_time':'PT_TIME', 'PT_distance':'PT_DISTANCE', 'PT_waiting':'PT_WAITING', 
    'PT_walking':'PT_WALKING', 'PT_transfer':'PT_TRANSFER', 'Block':'BLOCK', 
    'commuting_days':'COMMUTING_DAYS', 'pt1_wait':'PT1_WAIT', 'pt1_walk':'PT1_WALK', 
    'pt1_tt':'PT1_TT', 'pt1_cost':'PT1_COST', 'pt1_trans':'PT1_TRANS', 
    'pt1_crowd':'PT1_CROWD', 'cs2_walk':'CS2_WALK', 'cs2_tt':'CS2_TT', 
    'cs2_cost':'CS2_COST', 'cs2_disin':'CS2_DISIN', 'rs3_wait':'RS3_WAIT', 
    'rs3_tt':'RS3_TT', 'rs3_cost':'RS3_COST', 'rs3_share':'RS3_SHARE', 
    'cs_1':'CS_1', 'cs_2':'CS_2', 'num':'NUM', 'mode 2023 ':'MODE_2023'})

if d != 0:
    for i in travel.index:
        if df['MODE_2023'][i] == d:
            df.drop(i, inplace = True)
            travel.drop(i, inplace = True)
else:
    dfrs = df.copy()
    dfpt = df.copy()
    for i in travel.index:
        if dfrs['MODE_2023'][i] == 4:
            dfrs.drop(i, inplace = True)
    for i in travel.index:
        if dfpt['MODE_2023'][i] == 3:
            dfpt.drop(i, inplace = True)
    del i
del d

#%%%
database = db.Database('rspt', df)

WFH_ALLOWED = Variable('WFH_ALLOWED')
WFH_NOW = Variable('WFH_NOW')
AGE = Variable('AGE')
TIME_LIVING = Variable('TIME_LIVING')
GENDER = Variable('GENDER')
EDU = Variable('EDU')
LICENSE = Variable('LICENSE')
INCOME_1 = Variable('INCOME_1')
INCOME_2 = Variable('INCOME_2')
QHOUSEHOLD_1 = Variable('QHOUSEHOLD_1')
QHOUSEHOLD_2 = Variable('QHOUSEHOLD_2')
QHOUSEHOLD_3 = Variable('QHOUSEHOLD_3')
QHOUSEHOLD_4 = Variable('QHOUSEHOLD_4')
QHOUSEHOLD_5 = Variable('QHOUSEHOLD_5')
FLOOR_AREA = Variable('FLOOR_AREA')
QFAMILY_1 = Variable('QFAMILY_1')
QFAMILY_2 = Variable('QFAMILY_2')
QFAMILY_3 = Variable('QFAMILY_3')
QFAMILY_4 = Variable('QFAMILY_4')
QFAMILY_5 = Variable('QFAMILY_5')
QFAMILY_5_1 = Variable('QFAMILY_5_1')
QFAMILY_7 = Variable('QFAMILY_7')
QOFFICE_1 = Variable('QOFFICE_1')
QOFFICE_2 = Variable('QOFFICE_2')
QOFFICE_3 = Variable('QOFFICE_3')
QOFFICE_4 = Variable('QOFFICE_4')
CSRS_1_1 = Variable('CSRS_1_1') 
CSRS_1_2 = Variable('CSRS_1_2')
PRIVATE_CAR_1_1 = Variable('PRIVATE_CAR_1_1')
PRIVATE_CAR_1_2 = Variable('PRIVATE_CAR_1_2')
RS_EXPERIENCE_1 = Variable('RS_EXPERIENCE_1')
RS_EXPERIENCE_2 = Variable('RS_EXPERIENCE_2')
RS_EXPERIENCE_3 = Variable('RS_EXPERIENCE_3')
PT_EXPERIENCE_1 = Variable('PT_EXPERIENCE_1')
PT_EXPERIENCE_2 = Variable('PT_EXPERIENCE_2') 
PT_EXPERIENCE_3 = Variable('PT_EXPERIENCE_3')
ATT_PT_1 = Variable('ATT_PT_1')
ATT_PT_2 = Variable('ATT_PT_2')
ATT_PT_3 = Variable('ATT_PT_3')
ATT_PT_4 = Variable('ATT_PT_4')
ATT_CS_1 = Variable('ATT_CS_1')
ATT_CS_2 = Variable('ATT_CS_2')  
ATT_CS_3 = Variable('ATT_CS_3')
ATT_SHIFT_RS_1 = Variable('ATT_SHIFT_RS_1')  
ATT_SHIFT_RS_2 = Variable('ATT_SHIFT_RS_2')
ATT_SHIFT_RS_3 = Variable('ATT_SHIFT_RS_3')  
DRIVING_TIME = Variable('DRIVING_TIME')
DRIVING_DISTANCE = Variable('DRIVING_DISTANCE') 
PT_TIME = Variable('PT_TIME')
PT_DISTANCE = Variable('PT_DISTANCE')  
PT_WAITING = Variable('PT_WAITING')
PT_WALKING = Variable('PT_WALKING')  
PT_TRANSFER = Variable('PT_TRANSFER')
BLOCK = Variable('BLOCK')  
COMMUTING_DAYS = Variable('COMMUTING_DAYS')
PT1_WAIT = Variable('PT1_WAIT') 
PT1_WALK = Variable('PT1_WALK')
PT1_TT = Variable('PT1_TT')  
PT1_COST = Variable('PT1_COST')
PT1_TRANS = Variable('PT1_TRANS')  
PT1_CROWD = Variable('PT1_CROWD')
CS2_WALK = Variable('CS2_WALK')  
CS2_TT = Variable('CS2_TT')
CS2_COST = Variable('CS2_COST') 
CS2_DISIN = Variable('CS2_DISIN')
RS3_WAIT = Variable('RS3_WAIT')  
RS3_TT = Variable('RS3_TT')
RS3_COST = Variable('RS3_COST')  
RS3_SHARE = Variable('RS3_SHARE')
CS_1 = Variable('CS_1')  
CS_2 = Variable('CS_2')
NUM = Variable('NUM') 
MODE_2023 = Variable('MODE_2023') 



#%% data summery & visualize

# summary table
print(df.head())
describe = df.describe()
print(describe)
describe.to_excel(
    '/Users/yangziyuan/Documents/学/大四上/fyp/FYP CODE/data/describe.xlsx')


# scattermatrix
sns.pairplot(df, kind="scatter", hue='MODE_2023', 
             markers=["o", "s"], palette="Set2")
plt.show()


# 交叉频数分析1
ct = pd.crosstab(df['ATT_PT_2'], df['MODE_2023'], normalize=True)
sns.heatmap(ct, cmap='YlOrRd', annot=True, mask=ct<0.1) #图1
from statsmodels.graphics.mosaicplot import mosaic
props = lambda key: {"color": "0.45"} if '4' in key else {"color": "#C6E2FF"}
mosaic(ct.stack(), properties=props) #图2


# 交叉频数分析2 频率分布直方图
sns.distplot(dfrs['ATT_PT_2'], color="skyblue", label="rs", kde=True)
sns.distplot(dfpt['ATT_PT_2'], color="red", label="pt", kde=True)
plt.legend() 
plt.show()

#
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
t = -1
for i in ['CSRS_1_1', 'CSRS_1_2', 'ATT_PT_1', 'ATT_PT_2', 'ATT_PT_3', 'ATT_PT_4']:
    t = t+1
    sns.distplot(dfrs[i], color="skyblue", label="rs", kde=True, ax=axs[t//3, t%3])
    sns.distplot(dfpt[i], color="red", label="pt", kde=True, ax=axs[t//3, t%3])
plt.legend() 
plt.show()
del t,i,fig,axs

#
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
t = -1
for i in ['ATT_SHIFT_RS_1', 'ATT_SHIFT_RS_2', 'ATT_SHIFT_RS_3']:
    t = t+1
    sns.distplot(dfrs[i], color="skyblue", label="rs", kde=True, ax=axs[t%3])
plt.legend() 
plt.show()
del t,i,fig,axs


'''# 分面展示多个直方图
# 设置画板
fig, axs = plt.subplots(2, 2, figsize=(7, 7))

# 分别绘制多个直方图
sns.distplot(df["sepal_length"], kde=True, color="skyblue", ax=axs[0, 0])
sns.distplot(df["sepal_width"], kde=True, color="olive", ax=axs[0, 1])
sns.distplot(df["petal_length"], kde=True, color="gold", ax=axs[1, 0])
sns.distplot(df["petal_width"], kde=True, color="teal", ax=axs[1, 1])

plt.show()'''


# 散点图
import random
dftem = df.copy()
for i in dftem.index:
    for j in df.columns:
        t = 10**len(str(dftem[j][i]))*0.0001
        dftem[j][i] = dftem[j][i] + random.uniform(-t, t)
del i,j,t

plt.plot( 'ATT_PT_2', 'MODE_2023', 
         data=dftem, linestyle='', marker='o', 
         markersize=3, alpha=0.05, color="red")
plt.xlabel('Value of X') # 设置x轴标签
plt.ylabel('Value of Y') # 设置y轴标签
plt.title('Overplotting looks like that:', loc='left') # 设置标题


# Create the plot
columns = list(df.columns)
columns.remove('NUM')
#columns.remove('MODE_2023')
g = sns.PairGrid(data=df, vars=columns, hue=None)
g.map_upper(reg_coef)
g = g.map_lower(sns.regplot, scatter_kws={"edgecolor": "white"})
g = g.map_diag(sns.histplot, kde=True)
plt.show()
del columns



#%% default modeling

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# Definition of new variables
SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)
CAR_AV_SP = CAR_AV * (SP != 0)
TRAIN_AV_SP = TRAIN_AV * (SP != 0)
TRAIN_TT_SCALED = TRAIN_TT / 100
TRAIN_COST_SCALED = TRAIN_COST / 100
SM_TT_SCALED = SM_TT / 100
SM_COST_SCALED = SM_COST / 100
CAR_TT_SCALED = CAR_TT / 100
CAR_CO_SCALED = CAR_CO / 100

# Definition of the utility functions
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'default' #change the model name here

'''# Calculate the null log likelihood for reporting.
print('Null log likelihood:',the_biogeme.calculateNullLoglikelihood(av))'''

# Estimate the parameters
the_biogeme.generate_html = False
the_biogeme.generate_pickle = False
the_biogeme.save_iterations = False
results = the_biogeme.estimate()
print(results.short_summary())

# Get the results in a pandas table
pandasResults = results.getEstimatedParameters()
print(pandasResults)

# Get the results in Html
newwriteHtml(results,
    path='/Users/yangziyuan/Documents/学/大四上/fyp/results/'
)






