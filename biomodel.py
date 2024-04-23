#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 07:51:23 2023

@author: yangziyuan
"""

from biogeme.expressions import Variable, Beta
from biogeme import models
import biogeme.biogeme as bio
import biogeme.database as db


#%% modeling

# Parameters to be estimated
ASC_PT = Beta('ASC_PT', 0, None, None, 1)
ASC_CS = Beta('ASC_CS', 0, None, None, 0)
ASC_RS = Beta('ASC_RS', 0, None, None, 0)

B_TT_CS = Beta('B_TT_CS', 0, None, None, 0)
B_TT_RS = Beta('B_TT_RS', 0, None, None, 0)

B_Cost = Beta('B_Cost', 0, None, None, 0)
B_Trans = Beta('B_Trans', 0, None, None, 0)
B_Crowd = Beta('B_Crowd', 0, None, None, 0)
B_Share = Beta('B_Share', 0, None, None, 0)

B_PTA = Beta('B_PTA', 0, None, None, 0)
B_CSA = Beta('B_CSA', 0, None, None, 0)
B_RSA = Beta('B_RSA', 0, None, None, 1)
B_PTO = Beta('B_PTO', 0, None, None, 1)
B_CSO = Beta('B_CSO', 0, None, None, 0)
B_RSO = Beta('B_RSO', 0, None, None, 0)

# Scaleing variables
CS_1 = Variable('CS_1')
COMMUTING_DAYS = Variable('COMMUTING_DAYS')
PT1_COST = Variable('PT1_COST')
CS2_COST = Variable('CS2_COST')
RS3_COST = Variable('RS3_COST')
PT1_TRANS = Variable('PT1_TRANS')
PT1_CROWD = Variable('PT1_CROWD')
RS3_SHARE = Variable('RS3_SHARE')
PT1_TT = Variable('PT1_TT')
CS2_TT = Variable('CS2_TT')
RS3_TT = Variable('RS3_TT')
AGE = Variable('AGE')
QOFFICE_3 = Variable('QOFFICE_3')
INCOME_2 = Variable('INCOME_2')
 
CS_1 = CS_1
COMMUTING_DAYS_S = COMMUTING_DAYS*4
PT1_COST_S = PT1_COST / 1000
CS2_COST_S = CS2_COST / 1000
RS3_COST_S = RS3_COST / 1000
PT1_TRANS_S = PT1_TRANS / 1
PT1_CROWD_S = PT1_CROWD / 10
RS3_SHARE_S = RS3_SHARE / 10
PT1_TT_S = PT1_TT / 1000
CS2_TT_S = CS2_TT / 1000
RS3_TT_S = RS3_TT / 1000
AGE_S = AGE / 10
QOFFICE_3_S = QOFFICE_3 / 1
INCOME_2_S = INCOME_2 / 10000

#pt
PT = (ASC_PT 
      + B_Cost*PT1_COST_S*(1+1/INCOME_2_S)
      + B_PTA*AGE_S + B_PTO*QOFFICE_3_S
      + B_Crowd*PT1_CROWD_S
      + B_Trans*PT1_TRANS_S
      
      )

#cs
CS = (ASC_CS 
      + B_Cost*CS2_COST_S*(1+1/INCOME_2_S)
      + B_TT_CS*(PT1_TT_S-CS2_TT_S)*COMMUTING_DAYS_S
      + B_CSA*AGE_S + B_CSO*QOFFICE_3_S
      
      )

#rs
RS = (ASC_RS 
      + B_Cost*RS3_COST_S*(1+1/INCOME_2_S)
      + B_TT_RS*(PT1_TT_S-RS3_TT_S)*COMMUTING_DAYS_S
      + B_RSA*AGE_S + B_RSO*QOFFICE_3_S
      + B_Share*RS3_SHARE_S
      
      )

V1 = PT
V2 = CS
V3 = RS

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, None, CS_1)

def modelResult(data):
    database = db.Database('Data', data)
    
    # Create the Biogeme object
    the_biogeme = bio.BIOGEME(database, logprob)
    the_biogeme.modelName = 'default' #change the model name here

    # Estimate the parameters
    the_biogeme.generate_html = False
    the_biogeme.generate_pickle = False
    the_biogeme.save_iterations = False
    results = the_biogeme.estimate()
    return(the_biogeme,results)
    


