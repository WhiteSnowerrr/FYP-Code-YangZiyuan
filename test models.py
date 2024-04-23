#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 07:51:23 2023

@author: yangziyuan
"""

import biobase
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from biogeme.expressions import Variable, Beta
from biogeme import models
import biogeme.biogeme as bio
import biogeme.database as db
import math
#row-choose row data:rs/pt/all
#col-choose col data:obj/subj/sp/all
df0,dfrs0,dfpt0 = biobase.biodata('all', 'all')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import bioplot

#%%

bioplot.cDFP(df0['CS2_TT'])
biobase.cDFP(df0['CS2_WALK'])

biobase.cDFP(df0['RS3_TT'])
biobase.cDFP(df0['RS3_WAIT'])

biobase.cDFP(df0['PT1_COST'])
biobase.cDFP(df0['CS2_COST'])
biobase.cDFP(df0['RS3_COST'])


bioplot.cDFP(df0['PT_TIME'],'all')
biobase.cDFP(dfrs0['PT_TIME'],'rs')
biobase.cDFP(dfpt0['PT_TIME'],'pt')

biobase.cDFP(df0['PT_WAITING'],'all')
biobase.cDFP(dfrs0['PT_WAITING'],'rs')
biobase.cDFP(dfpt0['PT_WAITING'],'pt')

biobase.cDFP(df0['PT_WALKING'],'all')
biobase.cDFP(dfrs0['PT_WALKING'],'rs')
biobase.cDFP(dfpt0['PT_WALKING'],'pt')

plot=plt.figure()
ax1=plot.add_subplot(1,1,1)
ax=sns.histplot(dfrs0['PT1_TT'], color="skyblue", label="RS",
                stat='density', discrete=True, shrink=.5, kde=True, 
                kde_kws={'cut':1.8}, edgecolor='dodgerblue')
ax=sns.histplot(dfpt0['PT1_TT'], color="red", label="PT", 
                stat='density', discrete=True, shrink=.5, kde=True, 
                kde_kws={'cut':1.8}, edgecolor='orangered')
plt.legend()

temp = biobase.evenData(dfrs0, dfpt0, 'PT_TIME')

plot=plt.figure()
ax1=plot.add_subplot(1,1,1)
ax=sns.histplot(temp['PT1_TT'], color="skyblue", label="RS",
                stat='density', discrete=True, shrink=.5, kde=True, 
                kde_kws={'cut':1.8}, edgecolor='dodgerblue')
ax=sns.histplot(dfpt0['PT1_TT'], color="red", label="PT", 
                stat='density', discrete=True, shrink=.5, kde=True, 
                kde_kws={'cut':1.8}, edgecolor='orangered')
plt.legend()

temp2 = biobase.evenData(dfpt0, dfrs0, 'PT_TIME')

plot=plt.figure()
ax1=plot.add_subplot(1,1,1)
ax=sns.histplot(dfrs0['PT1_TT'], color="skyblue", label="RS",
                stat='density', discrete=True, shrink=.5, kde=True, 
                kde_kws={'cut':1.8}, edgecolor='dodgerblue')
ax=sns.histplot(temp2['PT1_TT'], color="red", label="PT", 
                stat='density', discrete=True, shrink=.5, kde=True, 
                kde_kws={'cut':1.8}, edgecolor='orangered')
plt.legend()



describe = dfpt0.describe()
print(describe)

biobase.cDFP(dfrs0['DRIVING_TIME'],'rs')
biobase.cDFP(dfpt0['DRIVING_TIME'],'pt')
biobase.cDFP(dfrs0['DRIVING_DISTANCE'],'rs')
biobase.cDFP(dfpt0['DRIVING_DISTANCE'],'pt')
biobase.cDFP(dfrs0['PT_TIME'],'rs')
biobase.cDFP(dfpt0['PT_TIME'],'pt')
biobase.cDFP(dfrs0['PT_DISTANCE'],'rs')
biobase.cDFP(dfpt0['PT_DISTANCE'],'pt')
biobase.cDFP(dfrs0['PT_WAITING'],'rs')
biobase.cDFP(dfpt0['PT_WAITING'],'pt')
biobase.cDFP(dfrs0['PT_WALKING'],'rs')
biobase.cDFP(dfpt0['PT_WALKING'],'pt')

bioplot.scatterplot(dfrs0, 'CS2_DISIN', 'CS_1')
bioplot.scatterplot(dfpt0, 'CS2_DISIN', 'CS_1')




sns.histplot(df0['AGE'], color="skyblue")
sns.histplot(df0['GENDER'], color="skyblue")
sns.histplot(df0['EDU'], color="skyblue")
sns.histplot(df0['TIME_LIVING'], color="skyblue")
sns.histplot(df0['CS_RS_1_1'], color="skyblue")
sns.histplot(df0['CS_RS_1_2'], color="skyblue")
sns.histplot(df0['QOFFICE_3'], color="skyblue")
sns.histplot(df0['QOFFICE_4'], color="skyblue")
sns.histplot(df0['QHOUSEHOLD_3'], color="skyblue")
sns.histplot(df0['INCOME_2'], color="skyblue")
sns.histplot(df0['PRIVATE_CAR_1_2'], color="skyblue")
t = df0['INCOME_2']-df0['INCOME_1']
sns.histplot(t, color="skyblue")

sns.histplot(df0['QFAMILY_5'], color="skyblue")


#%% modeling for rs
df = df0.copy()
dfrs = dfrs0.copy()
dfpt = dfpt0.copy()

for i in df.index:
    if df['CS_1'][i] == 4:
        df.loc[i,'CS_1'] = 3
    if df['CS_2'][i] == 4:
        df.loc[i,'CS_2'] = 3
    if df.loc[i,'MODE_2023'] == 2:
        df.loc[i,'MODE_2023'] = 0

for i in dfrs.index:
    if dfrs['CS_1'][i] == 4:
        dfrs.loc[i,'CS_1'] = 3
    if dfrs['CS_2'][i] == 4:
        dfrs.loc[i,'CS_2'] = 3
    if dfrs.loc[i,'MODE_2023'] == 2:
        dfrs.loc[i,'MODE_2023'] = 0

for i in dfpt.index:
    if dfpt['CS_1'][i] == 4:
        dfpt.loc[i,'CS_1'] = 3
    if dfpt['CS_2'][i] == 4:
        dfpt.loc[i,'CS_2'] = 3
    if dfpt.loc[i,'MODE_2023'] == 2:
        dfpt.loc[i,'MODE_2023'] = 0

#a = pd.concat([dfrs,dfrs,dfpt])
database0 = db.Database('SP', df)
database = db.Database('RS_SP', dfrs)
database2 = db.Database('PT_SP', dfpt)
# Parameters to be estimated

B_WaitT = Beta('B_WaitT', 0, None, None, 0)
B_WalkT = Beta('B_WalkT', 0, None, None, 0)
B_TT = Beta('B_TT', 0, None, None, 0)
B_Cost = Beta('B_Cost', 0, None, None, 0)


B_WaitT_PT = Beta('B_WaitT_PT', 0, None, None, 0)
B_WalkT_PT = Beta('B_WalkT_PT', 0, None, None, 0)
B_TT_PT = Beta('B_TT_PT', 0, None, None, 0)
B_Cost_PT = Beta('B_Cost_PT', 0, None, None, 0)

B_WaitT_CS = Beta('B_WaitT_CS', 0, None, None, 0)
B_WalkT_CS = Beta('B_WalkT_CS', 0, None, None, 0)
B_TT_CS = Beta('B_TT_CS', 0, None, None, 0)
B_Cost_CS = Beta('B_Cost_CS', 0, None, None, 0)

B_WaitT_RS = Beta('B_WaitT_RS', 0, None, None, 0)
B_WalkT_RS = Beta('B_WalkT_RS', 0, None, None, 0)
B_TT_RS = Beta('B_TT_RS', 0, None, None, 0)
B_Cost_RS = Beta('B_Cost_RS', 0, None, None, 0)

B_Trans = Beta('B_Trans', 0, None, None, 0)
B_Crowd = Beta('B_Crowd', 0, None, None, 0)
B_Disin = Beta('B_Disin', 0, None, None, 0)
B_Share = Beta('B_Share', 0, None, None, 0)
B_Lic = Beta('B_Lic', 0, None, None, 0)




for i in dfrs.columns:
    globals()[i] = Variable(i) 
del i
# Scaleing variables
COMMUTING_DAYS = COMMUTING_DAYS*4
PT1_COST = PT1_COST / 1000
CS2_COST = CS2_COST / 1000
RS3_COST = RS3_COST / 1000
PT1_TRANS = PT1_TRANS / 1
PT1_CROWD = PT1_CROWD / 10
CS2_DISIN = CS2_DISIN / 10
RS3_SHARE = RS3_SHARE / 10

PT1_WAIT = PT1_WAIT / 1000
PT1_WALK = PT1_WALK / 1000
PT1_TT = PT1_TT / 1000

CS2_WALK = CS2_WALK / 1000
CS2_TT = CS2_TT / 1000

RS3_WAIT = RS3_WAIT / 1000
RS3_TT = RS3_TT / 1000

ASC_PT = Beta('ASC_PT', 0, None, None, 1)
ASC_CS = Beta('ASC_CS', 0, None, None, 0)
ASC_RS = Beta('ASC_RS', 0, None, None, 0)

B_In = Beta('B_In', 0, None, None, 0)

B_TT_1 = Beta('B_TT_1', 0, None, None, 0)
B_TT_2 = Beta('B_TT_2', 0, None, None, 0)
B_TT_3 = Beta('B_TT_3', 0, None, None, 0)
B_TT_4 = Beta('B_TT_4', 0, None, None, 0)
B_TT_5 = Beta('B_TT_5', 0, None, None, 0)
B_TT_6 = Beta('B_TT_6', 0, None, None, 0)
B_TT_7 = Beta('B_TT_7', 0, None, None, 0)

AGE = AGE / 10
B_PTA = Beta('B_PTA', 0, None, None, 0)
B_CSA = Beta('B_CSA', 0, None, None, 0)
B_RSA = Beta('B_RSA', 0, None, None, 1)

B_PTO = Beta('B_PTO', 0, None, None, 1)
B_CSO = Beta('B_CSO', 0, None, None, 0)
B_RSO = Beta('B_RSO', 0, None, None, 0)

INCOME_2 = INCOME_2 / 10000
B_PTI = Beta('B_PTI', 0, None, None, 1)
B_CSI = Beta('B_CSI', 0, None, None, 0)
B_RSI = Beta('B_RSI', 0, None, None, 0)

PT1_TT_1 = PT1_TT_1 / 100
PT1_TT_2 = PT1_TT_2 / 100
PT1_TT_3 = PT1_TT_3 / 100
'''
PT1_WALK_1 = PT1_WALK_1 / 100
PT1_WALK_2 = PT1_WALK_2 / 100
PT1_WALK_3 = PT1_WALK_3 / 100
'''
RS3_TT_1 = RS3_TT_1 / 100
RS3_TT_2 = RS3_TT_2 / 100


#pt
PT = (ASC_PT 
    
      
      )

#cs
CS = (ASC_CS 
      
      + (B_TT_1 + B_TT_2*MODE_2023)*CS2_DISIN 
      
      )

#rs
RS = (ASC_RS 
     
      
      )

V1 = PT
V2 = CS
V3 = RS

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, None, CS_1)

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'default_sp_rs' #change the model name here

# Estimate the parameters
the_biogeme.generate_html = False
the_biogeme.generate_pickle = False
the_biogeme.save_iterations = False
results = the_biogeme.estimate()


# Create the Biogeme object
the_biogeme2 = bio.BIOGEME(database2, logprob)
the_biogeme2.modelName = 'default_sp_pt' #change the model name here
the_biogeme2.generate_html = False
the_biogeme2.generate_pickle = False
the_biogeme2.save_iterations = False
results2 = the_biogeme2.estimate()


# Create the Biogeme object
the_biogeme0 = bio.BIOGEME(database0, logprob)
the_biogeme0.modelName = 'default_sp' #change the model name here

# Estimate the parameters
the_biogeme0.generate_html = False
the_biogeme0.generate_pickle = False
the_biogeme0.save_iterations = False
results0 = the_biogeme0.estimate()


pandasResults = results.getEstimatedParameters()
pandasResults2 = results2.getEstimatedParameters()
pandasResults0 = results0.getEstimatedParameters()


print('RS')
for i in pandasResults.index:
    if pandasResults['Rob. p-value'][i] >=0.05:
        print(i)
print(pandasResults)

print('\nPT')
for i in pandasResults2.index:
    if pandasResults2['Rob. p-value'][i] >=0.05:
        print(i)
print(pandasResults2)


print('\nALL')
for i in pandasResults0.index:
    if pandasResults0['Rob. p-value'][i] >=0.05:
        print(i)
print(pandasResults0)
del i



#%% ok model

#1
#pt
PT = (ASC_PT + B_WaitT*(PT1_WAIT)*(PT1_TRANS+1) + B_TT*(PT1_TT+PT1_WALK) + 
      B_Cost*(PT1_COST-5*CS2_COST)/COMMUTING_DAYS + B_Crowd*PT1_CROWD*(ATT_PT_1+ATT_PT_2-1) + 
      B_PTE*EDU
      )

#cs
CS = (ASC_CS + B_TT_CS*(CS2_TT) + 
      B_Disin*CS2_DISIN*(ATT_PT_1+ATT_PT_2-1) + 
      B_CSE*EDU
      )

#rs
RS = (ASC_RS + B_WaitT*RS3_WAIT + B_TT*(RS3_TT) + 
      B_Cost*(RS3_COST-5*CS2_COST)/COMMUTING_DAYS + B_Share*RS3_SHARE*(ATT_PT_1+ATT_PT_2-1) + 
      B_RSE*EDU
      )




#pt
PT = (ASC_PT 
      + B_Cost*PT1_COST 
      + B_Crowd*PT1_CROWD + B_Trans*PT1_TRANS


      )

#cs
CS = (ASC_CS 
      + B_Cost*CS2_COST
      + B_TT_CS*(PT1_TT-CS2_TT)*COMMUTING_DAYS

      
      )

#rs
RS = (ASC_RS 
      + B_Cost*RS3_COST
      + B_TT_RS*(PT1_TT-RS3_TT)*COMMUTING_DAYS
      + B_Share*RS3_SHARE

      
      )
