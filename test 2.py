#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:45:33 2023

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
df0,dfrs0,dfpt0 = biobase.biodata('all', 'sp')


## modeling for rs
dftemp = df0.copy()
dfrs = dfrs0.copy()
dfpt = dfpt0.copy()
for i in dftemp.index:
    if dftemp['CS_1'][i] == 4:
        dftemp.loc[i,'CS_1'] = 3
    if dftemp['CS_2'][i] == 4:
        dftemp.loc[i,'CS_2'] = 3

for i in dfrs.index:
    if dfrs['CS_1'][i] == 4:
        dfrs.loc[i,'CS_1'] = 3
    if dfrs['CS_2'][i] == 4:
        dfrs.loc[i,'CS_2'] = 3

for i in dfpt.index:
    if dfpt['CS_1'][i] == 4:
        dfpt.loc[i,'CS_1'] = 3
    if dfpt['CS_2'][i] == 4:
        dfpt.loc[i,'CS_2'] = 3
del i

df, df_test = biobase.trainTestSplit(dftemp, test_size=0.2, random_state=20011207)


database0 = db.Database('SP', df)
database = db.Database('RS_SP', dfrs)
database2 = db.Database('PT_SP', dfpt)
# Parameters to be estimated
ASC_PT = Beta('ASC_PT', 0, None, None, 0)
ASC_CS = Beta('ASC_CS', 0, None, None, 0)
ASC_RS = Beta('ASC_RS', 0, None, None, 1)

B_WaitT = Beta('B_WaitT', 0, None, 0, 0)
B_WalkT = Beta('B_WalkT', 0, None, 0, 0)
B_TT = Beta('B_TT', 0, None, None, 0)
B_Cost = Beta('B_Cost', 0, None, None, 0)
B_TC = Beta('B_TC', 0, None, None, 0)

B_WaitT_PT = Beta('B_WaitT_PT', 0, None, 0, 0)
B_WalkT_PT = Beta('B_WalkT_PT', 0, None, 0, 0)
B_TT_PT = Beta('B_TT_PT', 0, None, 0, 0)
B_Cost_PT = Beta('B_Cost_PT', 0, None, None, 0)

B_WalkT_CS = Beta('B_WalkT_CS', 0, None, 0, 0)
B_TT_CS = Beta('B_TT_CS', 0, None, 0, 0)
B_Cost_CS = Beta('B_Cost_CS', 0, None, None, 0)

B_WaitT_RS = Beta('B_WaitT_RS', 0, None, 0, 0)
B_TT_RS = Beta('B_TT_RS', 0, None, 0, 0)
B_Cost_RS = Beta('B_Cost_RS', 0, None, 0, 0)

B_Trans = Beta('B_Trans', 0, None, None, 0)
B_Crowd = Beta('B_Crowd', 0, None, None, 0)
B_Disin = Beta('B_Disin', 0, None, None, 0)
B_Share = Beta('B_Share', 0, None, None, 0)



for i in dfrs.columns:
    globals()[i] = Variable(i) 
del i
# Scaleing variables
COMMUTING_DAYS = COMMUTING_DAYS / 10


MODE_2023 = MODE_2023
PT1_COST_S = PT1_COST / 10000
CS2_COST_S = CS2_COST / 10000
RS3_COST_S = RS3_COST / 10000
PT1_TRANS = PT1_TRANS
PT1_CROWD = PT1_CROWD / 10
CS2_DISIN = CS2_DISIN / 100
RS3_SHARE = RS3_SHARE / 10

PT1_WAIT_S = PT1_WAIT / 100
PT1_WALK_S = PT1_WALK / 100
PT1_TT_S = PT1_TT / 100

CS2_WALK = CS2_WALK / 100
CS2_TT_S = CS2_TT / 100

RS3_WAIT_S = RS3_WAIT / 100
RS3_TT_S = RS3_TT / 100


# Definition of the utility functions
#pt
PT = (ASC_PT + B_WaitT*(PT1_WAIT_S)*(PT1_TRANS+1) + B_TT*(PT1_TT_S+PT1_WALK_S) + 
      B_Cost*PT1_COST_S

      )

#cs
CS = (ASC_CS + B_TT_CS*(CS2_TT_S) + 
      B_Cost*CS2_COST_S

      )

#rs
RS = (ASC_RS + B_WaitT*RS3_WAIT_S + B_TT*(RS3_TT_S) + 
      B_Cost*RS3_COST_S

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
the_biogeme0 = bio.BIOGEME(database0, logprob)
the_biogeme0.modelName = 'default_sp' #change the model name here

# Estimate the parameters
the_biogeme0.generate_html = False
the_biogeme0.generate_pickle = False
the_biogeme0.save_iterations = False
results0 = the_biogeme0.estimate()
pandasResults0 = results0.getEstimatedParameters()
print(pandasResults0)



#%% predict
prob_PT = models.logit(V, None, 1)
prob_CS = models.logit(V, None, 2)
prob_RS = models.logit(V, None, 3)

simulate ={'Prob. PT':  prob_PT ,
           'Prob. CS':  prob_CS ,
           'Prob. RS':  prob_RS ,}

database_test = db.Database('SP_test', df_test)
the_biogeme_test = bio.BIOGEME(database_test, simulate)
the_biogeme_test.modelName = "default_sp_test"

betaValues = results0.getBetaValues()

simulatedValues = the_biogeme_test.simulate(betaValues)
print(simulatedValues.head())

prob_max = simulatedValues.idxmax(axis=1)
prob_max = prob_max.replace({'Prob. PT': 1, 'Prob. CS': 2, 'Prob. RS': 3})

data = {'y_Actual':    df_test['CS_1'],
        'y_Predicted': prob_max
        }

AP = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(AP['y_Actual'], AP['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

confusion_matrix

accuracy = np.diagonal(confusion_matrix.to_numpy()).sum()/confusion_matrix.to_numpy().sum()
print('Global accuracy of the model:', accuracy)


#%% market
prob_PT = models.logit(V, None, 1)
prob_CS = models.logit(V, None, 2)
prob_RS = models.logit(V, None, 3)

simulate ={'Prob. PT':  prob_PT ,
           'Prob. CS':  prob_CS ,
           'Prob. RS':  prob_RS ,
           'Revenue public transportation': prob_PT * PT1_COST}

the_biogeme_market = bio.BIOGEME(database0, simulate)
the_biogeme_market.modelName = "default_sp_market"
betaValues = results0.getBetaValues()

betas = the_biogeme_market.free_beta_names()

simulatedValues = the_biogeme_market.simulate(betaValues)
b = results0.getBetasForSensitivityAnalysis(betas , size=100, useBootstrap=False)

left, right = the_biogeme_market.confidenceIntervals(b, .9)

marketShare_PT = simulatedValues['Prob. PT'].mean()
marketShare_PT_left = left['Prob. PT'].mean()
marketShare_PT_right = right['Prob. PT'].mean()

marketShare_CS = simulatedValues['Prob. CS'].mean()
marketShare_CS_left = left['Prob. CS'].mean()
marketShare_CS_right = right['Prob. CS'].mean()

marketShare_RS = simulatedValues['Prob. RS'].mean()
marketShare_RS_left = left['Prob. RS'].mean()
marketShare_RS_right = right['Prob. RS'].mean()

print(f"Market Share for PT :  {100*marketShare_PT:.1f}%  [{100*marketShare_PT_left:.1f} % , {100*marketShare_PT_right:.1f} %]")
print(f"Market Share for CS :  {100*marketShare_CS:.1f}%  [{100*marketShare_CS_left:.1f} % , {100*marketShare_CS_right:.1f} %]")
print(f"Market Share for RS :  {100*marketShare_RS:.1f}%  [{100*marketShare_RS_left:.1f} % , {100*marketShare_RS_right:.1f} %]")

revenues_pt = ( simulatedValues['Revenue public transportation']).sum()
revenues_pt_left = (left['Revenue public transportation']).sum()
revenues_pt_right = ( right ['Revenue public transportation']).sum()

print( f"Revenues for PT : {revenues_pt:.3f} [{revenues_pt_left:.3f}, {revenues_pt_right:.3f}]")


#%% revenue-scenarios

def scenario(scale):
    RS3_COST_Scenario = RS3_COST_S * scale
    
    # Definition of the utility functions
    #pt
    PT0 = (ASC_PT + B_WaitT*(PT1_WAIT_S)*(PT1_TRANS+1) + B_TT*(PT1_TT_S+PT1_WALK_S) + 
          B_Cost*PT1_COST_S

          )

    #cs
    CS0 = (ASC_CS + B_TT_CS*(CS2_TT_S) + 
          B_Cost*CS2_COST_S

          )

    #rs
    RS0 = (ASC_RS + B_WaitT*RS3_WAIT_S + B_TT*(RS3_TT_S) + 
          B_Cost*RS3_COST_Scenario ##change here!!!!!!

          )

    V10 = PT0
    V20 = CS0
    V30 = RS0

    # Associate utility functions with the numbering of alternatives
    V0 = {1: V10, 2: V20, 3: V30}
    prob_RS = models.logit(V0, None, 3)
    simulate ={'Prob. RS':  prob_RS ,
               'Revenue RS': prob_RS * RS3_COST_Scenario}
    the_biogeme_revenue_scenarios = bio.BIOGEME(database0, simulate)
    betas = the_biogeme_revenue_scenarios.free_beta_names()
    betaValues = results0.getBetaValues(betas)
    simulatedValues = the_biogeme_revenue_scenarios.simulate(betaValues)
    revenues_rs = (simulatedValues['Revenue RS']).sum()
    return revenues_rs


scales = np.arange(0.0,10.0,0.1)
revenues = [scenario(s) for s in scales]
plt.plot(scales,revenues)
plt.xlabel("Modification of the price of RS")
plt.ylabel("Revenues")
plt.show()



#%% elasticities
from biogeme.expressions import Derive
prob_PT = models.logit(V, None, 1)
prob_CS = models.logit(V, None, 2)
prob_RS = models.logit(V, None, 3)


# Direct Elasticities
direct_elas_pt_time  = Derive(prob_PT,'PT1_TT') * PT1_TT / prob_PT
direct_elas_pt_cost  = Derive(prob_PT,'PT1_COST') * PT1_COST / prob_PT
direct_elas_cs_time  = Derive(prob_CS,'CS2_TT') * CS2_TT / prob_CS
direct_elas_cs_cost  = Derive(prob_CS,'CS2_COST') * CS2_COST / prob_CS
direct_elas_rs_time  = Derive(prob_RS,'RS3_TT') * RS3_TT / prob_RS
direct_elas_rs_cost  = Derive(prob_RS,'RS3_COST') * RS3_COST / prob_RS

simulate ={'Prob. PT':  prob_PT ,
           'Prob. CS':  prob_CS ,
           'Prob. RS':  prob_RS ,
           'direct_elas_pt_time':direct_elas_pt_time,
           'direct_elas_pt_cost':direct_elas_pt_cost,
           'direct_elas_cs_time':direct_elas_cs_time,
           'direct_elas_cs_cost':direct_elas_cs_cost,
           'direct_elas_rs_time':direct_elas_rs_time,
           'direct_elas_rs_cost':direct_elas_rs_cost
            }

the_biogeme_elasticities = bio.BIOGEME(database0, simulate)
the_biogeme_elasticities.modelName = "default_sp_elasticities_direct"
betaValues = results0.getBetaValues()
simulatedValues = the_biogeme_elasticities.simulate(betaValues)

denominator_pt = simulatedValues['Prob. PT'].sum()
denominator_cs = simulatedValues['Prob. CS'].sum()
denominator_rs = simulatedValues['Prob. RS'].sum()

direct_elas_term_pt_time = (simulatedValues['Prob. PT'] * simulatedValues['direct_elas_pt_time'] / denominator_pt).sum()
direct_elas_term_pt_cost = (simulatedValues['Prob. PT'] * simulatedValues['direct_elas_pt_cost'] / denominator_pt).sum()
direct_elas_term_cs_time = (simulatedValues['Prob. CS'] * simulatedValues['direct_elas_cs_time'] / denominator_cs).sum()
direct_elas_term_cs_cost = (simulatedValues['Prob. CS'] * simulatedValues['direct_elas_cs_cost'] / denominator_cs).sum()
direct_elas_term_rs_time = (simulatedValues['Prob. RS'] * simulatedValues['direct_elas_rs_time'] / denominator_rs).sum()
direct_elas_term_rs_cost = (simulatedValues['Prob. RS'] * simulatedValues['direct_elas_rs_cost'] / denominator_rs).sum()

print(f"Aggregate direct elasticity of pt wrt time: {direct_elas_term_pt_time:.3g}")
print(f"Aggregate direct elasticity of pt wrt cost: {direct_elas_term_pt_cost:.3g}")
print(f"Aggregate direct elasticity of cs wrt time: {direct_elas_term_cs_time:.3g}")
print(f"Aggregate direct elasticity of cs wrt cost: {direct_elas_term_cs_cost:.3g}")
print(f"Aggregate direct elasticity of rs wrt time: {direct_elas_term_rs_time:.3g}")
print(f"Aggregate direct elasticity of rs wrt cost: {direct_elas_term_rs_cost:.3g}")


# Cross Elasticities
cross_elas_pt_time = Derive(prob_PT,'RS3_TT') * RS3_TT / prob_PT
cross_elas_pt_cost = Derive(prob_PT,'RS3_COST') * RS3_COST / prob_PT
cross_elas_rs_time = Derive(prob_RS,'PT1_TT') * PT1_TT / prob_RS
cross_elas_rs_cost = Derive(prob_RS,'PT1_COST') * PT1_COST / prob_RS

simulate ={'Prob. PT':  prob_PT ,
           'Prob. RS':  prob_RS ,
           'cross_elas_pt_time':cross_elas_pt_time,
           'cross_elas_pt_cost':cross_elas_pt_cost,
           'cross_elas_rs_time':cross_elas_rs_time,
           'cross_elas_rs_cost':cross_elas_rs_cost
            }

the_biogeme_elasticities = bio.BIOGEME(database0, simulate)
the_biogeme_elasticities.modelName = "default_sp_elasticities_cross"
betaValues = results0.getBetaValues()
simulatedValues = the_biogeme_elasticities.simulate(betaValues)

denominator_pt = simulatedValues['Prob. PT'].sum()
denominator_rs = simulatedValues['Prob. RS'].sum()

cross_elas_term_pt_time = (simulatedValues['Prob. PT'] * simulatedValues['cross_elas_pt_time'] / denominator_pt).sum()
cross_elas_term_pt_cost = (simulatedValues['Prob. PT'] * simulatedValues['cross_elas_pt_cost'] / denominator_pt).sum()
cross_elas_term_rs_time = (simulatedValues['Prob. RS'] * simulatedValues['cross_elas_rs_time'] / denominator_rs).sum()
cross_elas_term_rs_cost = (simulatedValues['Prob. RS'] * simulatedValues['cross_elas_rs_cost'] / denominator_rs).sum()

print(f"Aggregate cross elasticity of PT wrt RS time: {cross_elas_term_pt_time:.3g}")
print(f"Aggregate cross elasticity of PT wrt RS cost: {cross_elas_term_pt_cost:.3g}")
print(f"Aggregate cross elasticity of RS wrt PT time: {cross_elas_term_rs_time:.3g}")
print(f"Aggregate cross elasticity of RS wrt PT cost: {cross_elas_term_rs_cost:.3g}")


# Arc Elasticities

def scenario(delta_cost):
    RS3_COST_Scenario = (RS3_COST + delta_cost) / 10000
    
    # Definition of the utility functions
    #pt
    PT0 = (ASC_PT + B_WaitT*(PT1_WAIT_S)*(PT1_TRANS+1) + B_TT*(PT1_TT_S+PT1_WALK_S) + 
          B_Cost*PT1_COST_S

          )

    #cs
    CS0 = (ASC_CS + B_TT_CS*(CS2_TT_S) + 
          B_Cost*CS2_COST_S

          )

    #rs
    RS0 = (ASC_RS + B_WaitT*RS3_WAIT_S + B_TT*(RS3_TT_S) + 
          B_Cost*RS3_COST_Scenario ##change here!!!!!!

          )

    V10 = PT0
    V20 = CS0
    V30 = RS0

    # Associate utility functions with the numbering of alternatives
    V0 = {1: V10, 2: V20, 3: V30}
    prob_RS = models.logit(V, None, 3)
    prob_RS_after = models.logit(V0, None, 3)
    direct_elas_RS_cost = (prob_RS_after - prob_RS) * RS3_COST / (prob_RS * delta_cost)
    simulate ={'Prob. RS':  prob_RS ,
               'Prob. RS after':  prob_RS_after ,
               'direct_elas_RS_cost': direct_elas_RS_cost}
    the_biogeme_elasticities = bio.BIOGEME(database0, simulate)
    the_biogeme_elasticities.modelName = "default_sp_elasticities_arc"
    betaValues = results0.getBetaValues()
    simulatedValues = the_biogeme_elasticities.simulate(betaValues)
    denominator_rs = simulatedValues['Prob. RS'].sum()
    direct_elas_RS_cost = (simulatedValues['Prob. RS'] * simulatedValues['direct_elas_RS_cost'] / denominator_rs).sum()
    
    #print(f"Aggregate direct elasticity of RS wrt cost: {direct_elas_RS_cost:.3g}")
    return direct_elas_RS_cost


scales = range(0,100,2)
revenues = [scenario(s) for s in scales]
plt.plot(scales,revenues)
plt.xlabel("Arc Elasticities of the cost of RS")
plt.ylabel("elasticities")
plt.show()


#%% wtp
from biogeme.expressions import Derive
prob_PT = models.logit(V, None, 1)
prob_RS = models.logit(V, None, 3)
WTP_PT_TIME = Derive(prob_PT,'PT1_TT') / Derive(prob_PT,'PT1_COST')
WTP_RS_TIME = Derive(prob_RS,'RS3_TT') / Derive(prob_RS,'RS3_COST')

simulate = {'WTP PT time': WTP_PT_TIME,
            'WTP RS time': WTP_RS_TIME}

the_biogeme_wtp = bio.BIOGEME(database0, simulate, removeUnusedVariables=False)
the_biogeme_wtp.modelName = "default_sp_wtp"
betas = the_biogeme_wtp.free_beta_names()
betaValues = results0.getBetaValues()
simulatedValues = the_biogeme_wtp.simulate(betaValues)

wtprs = (60 * simulatedValues['WTP RS time']).mean()

b = results0.getBetasForSensitivityAnalysis(betas,size=100, useBootstrap=False)
left,right = the_biogeme_wtp.confidenceIntervals(b,0.9)
wtprs_left = (60 * left['WTP RS time']).mean()
wtprs_right = (60 * right['WTP RS time']).mean()

print(f"Average WTP for RS: {wtprs:.3g} CI:[{wtprs_left:.3g},{wtprs_right:.3g}]")

print("Unique values: ", [f"{i:.3g}" for i in 60 * simulatedValues['WTP RS time'].unique()])


def wtpForSubgroup(filter):
    sim = simulatedValues[filter]
    wtprs = (60 * sim['WTP RS time']).mean()
    wtprs_left = (60 * left[filter]['WTP RS time']).mean()
    wtprs_right = (60 * right[filter]['WTP RS time']).mean()
    return wtprs, wtprs_left,wtprs_right

filter = database0.data['CS_1'] == 1
w,l,r = wtpForSubgroup(filter)
print(f"WTP car for PT people: {w:.3g} CI:[{l:.3g},{r:.3g}]")


