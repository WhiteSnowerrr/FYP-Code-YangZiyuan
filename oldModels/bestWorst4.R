# ################################################################# #
#### LOAD LIBRARY AND DEFINE CORE SETTINGS                       ####
# ################################################################# #

### Clear memory
rm(list = ls())

### Load Apollo library
library(apollo)

### Initialise code
apollo_initialise()

### Set core controls
apollo_control = list(
  modelName       = "MNL_bestWorst",
  modelDescr      = "Best Worst model",
  indivID         = "ID", 
  #outputDirectory = "output",
  nCores = parallel::detectCores() - 1
)

# ################################################################# #
#### LOAD DATA AND APPLY ANY TRANSFORMATIONS                     ####
# ################################################################# #

### Loading data from package
source_python('bior.py')
database = apoRdata()[[1]]

### Create new variable with average income
database$mean_income = mean(database$income_2)
database$commuting_days_s = database$commuting_days*4
database$pt1_cost_s = database$pt1_cost/1000
database$cs2_cost_s = database$cs2_cost/1000
database$rs3_cost_s = database$rs3_cost/1000
database$pt1_trans_s = database$pt1_trans/1
database$pt1_crowd_s = database$pt1_crowd/10
database$rs3_share_s = database$rs3_share/10
database$pt1_tt_s = database$pt1_tt/1000
database$cs2_tt_s = database$cs2_tt/1000
database$rs3_tt_s = database$rs3_tt/1000
database$age_s = database$age/10
database$qoffice_3_s = database$qoffice_3/1


# ################################################################# #
#### ANALYSIS OF CHOICES                                         ####
# ################################################################# #

### Define settings for analysis of choice data to be conducted prior to model estimation
choiceAnalysis_settings <- list(
  alternatives = c(pt=1, cs=2, rs=3),
  avail        = 1,
  choiceVar    = database$choice_best,
  explanators  = database[,c("age","qoffice_3","income_2")],
  rows         = (database$mode_2023==1) #rs
)

### Run function to analyse choice data
apollo_choiceAnalysis(choiceAnalysis_settings, apollo_control, database)

# ################################################################# #
#### DEFINE MODEL PARAMETERS                                     ####
# ################################################################# #

### Vector of parameters, including any that are kept fixed in estimation
apollo_beta=c(asc_pt                = 0,
              asc_cs                = 0,
              asc_rs                = 0,
             
              b_tt_cs               = 0,
              b_tt_rs               = 0,
              b_tt_shift            = 0,
              
              b_cost                = 0,
              b_cost_shift          = 0,
              b_trans               = 0,
              b_trans_shift         = 0,
              b_crowd               = 0,
              b_crowd_shift         = 0,
              b_share               = 0,
              b_share_shift         = 0,
              
              b_pta                 = 0,
              b_pta_shift           = 0,
              b_csa                 = 0,
              b_csa_shift           = 0,
              b_rsa                 = 0,
              b_rsa_shift           = 0,
              cost_income_elast     = 1,
              
              scale_best            = 1,
              scale_worst           = 1
              )

### Vector with names (in quotes) of parameters to be kept fixed at their starting value in apollo_beta, use apollo_beta_fixed = c() if none
apollo_fixed = c("asc_pt","scale_best","b_rsa","b_rsa_shift")

### Read in starting values for at least some parameters from existing model output file
#apollo_beta = apollo_readBeta(apollo_beta, apollo_fixed, "MNL_base", overwriteFixed=FALSE)

# ################################################################# #
#### GROUP AND VALIDATE INPUTS                                   ####
# ################################################################# #

apollo_inputs = apollo_validateInputs()

# ################################################################# #
#### DEFINE MODEL AND LIKELIHOOD FUNCTION                        ####
# ################################################################# #

apollo_probabilities=function(apollo_beta, apollo_inputs, functionality="estimate"){
  
  ### Attach inputs and detach after function exit
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))
  
  ### Create list of probabilities P
  P = list()
  
  ### Create alternative specific constants and coefficients using interactions with socio-demographics
  b_tt_cs_value   = (b_tt_cs + b_tt_shift * mode_2023) * commuting_days_s
  b_tt_rs_value   = (b_tt_rs + b_tt_shift * mode_2023) * commuting_days_s
  b_cost_value    = (b_cost + b_cost_shift * mode_2023) * (mean_income / income_2) ^ cost_income_elast
  b_pta_value     = b_pta + b_pta_shift * mode_2023
  b_csa_value     = b_csa + b_csa_shift * mode_2023
  b_rsa_value     = b_rsa + b_rsa_shift * mode_2023


  b_share_value   = b_share + b_share_shift * mode_2023
  b_crowd_value   = b_crowd + b_crowd_shift * mode_2023
  b_trans_value   = b_trans + b_trans_shift * mode_2023
  
  ### List of utilities: these must use the same names as in mnl_settings, order is irrelevant
  V = list()
  V[["pt"]]  = asc_pt + b_cost_value * pt1_cost_s +                                       + b_pta_value * age_s + b_crowd_value * pt1_crowd_s + b_trans_value * pt1_trans_s
  V[["cs"]]  = asc_cs + b_cost_value * cs2_cost_s + b_tt_cs_value * (pt1_tt_s - cs2_tt_s) + b_csa_value * age_s
  V[["rs"]]  = asc_rs + b_cost_value * rs3_cost_s + b_tt_rs_value * (pt1_tt_s - rs3_tt_s) + b_rsa_value * age_s + b_share_value * rs3_share_s
  
  
  ### Compute probabilities for "best" choice using MNL model
  mnl_settings_best = list(
    alternatives = c(pt=1, cs=2, rs=3),
    avail        = 1,
    choiceVar    = choice_best,
    utilities    = list(pt  = scale_best*V[["pt"]],
                        cs  = scale_best*V[["cs"]],
                        rs  = scale_best*V[["rs"]]),
    componentName = "best"
  )
  P[["choice_best"]] = apollo_mnl(mnl_settings_best, functionality)
  
  ### Compute probabilities for "worst" choice using MNL model
  mnl_settings_worst = list(
    alternatives = c(pt=1, cs=2, rs=3),
    avail        = list(pt=(choice_best!=1), cs=(choice_best!=2), rs=(choice_best!=3)),
    choiceVar    = choice_worst,
    utilities    = list(pt  = -scale_worst*V[["pt"]],
                        cs  = -scale_worst*V[["cs"]],
                        rs  = -scale_worst*V[["rs"]])
  )
  
  P[["choice_worst"]] = apollo_mnl(mnl_settings_worst, functionality)
  
  ### Combined model
  P = apollo_combineModels(P, apollo_inputs, functionality)
  
  ### Take product across observation for same individual
  P = apollo_panelProd(P, apollo_inputs, functionality)
  
  ### Prepare and return outputs of function
  P = apollo_prepareProb(P, apollo_inputs, functionality)
  
  return(P)
}

# ################################################################# #
#### MODEL ESTIMATION                                            ####
# ################################################################# #

estimate_settings = list(#constraints = c("b_tt_cs > 0", "b_tt_rs > 0"),
                         writeIter = FALSE)

model = apollo_estimate(apollo_beta, apollo_fixed, apollo_probabilities, apollo_inputs, estimate_settings)

# ################################################################# #
#### MODEL OUTPUTS                                               ####
# ################################################################# #

# ----------------------------------------------------------------- #
#---- FORMATTED OUTPUT (TO SCREEN)                               ----
# ----------------------------------------------------------------- #

modelOutput_settings = list(printPVal=2
                            ,printClassical=FALSE
                            #,printOutliers=TRUE
                            )

apollo_modelOutput(model, modelOutput_settings)

# ----------------------------------------------------------------- #
#---- FORMATTED OUTPUT (TO FILE, using model name)               ----
# ----------------------------------------------------------------- #

apollo_saveOutput(model)

# ################################################################# #
##### POST-PROCESSING                                            ####
# ################################################################# #

### Print outputs of additional diagnostics to new output file (remember to close file writing when complete)
apollo_sink()

# ----------------------------------------------------------------- #
#---- LR TEST AGAINST SIMPLE MNL MODEL                           ----
# ----------------------------------------------------------------- #

### Example syntax with both models loaded from file
apollo_lrTest("MNL_SP", "MNL_SP_covariates")

### Example syntax with one model in memory
apollo_lrTest("MNL_SP", model)

# ----------------------------------------------------------------- #
#---- MODEL PREDICTIONS AND ELASTICITY CALCULATIONS              ----
# ----------------------------------------------------------------- #

### Use the estimated model to make predictions
predictions_base = apollo_prediction(model, apollo_probabilities, apollo_inputs, prediction_settings=list(runs=30))

### Now imagine the cost for rail increases by 1%
database$pt1_cost_s = 1.5*database$pt1_cost_s

### Rerun predictions with the new data
apollo_inputs = apollo_validateInputs()
predictions_new = apollo_prediction(model, apollo_probabilities, apollo_inputs)

### Return to original data
database$pt1_cost_s = 1/1.5*database$pt1_cost_s
apollo_inputs = apollo_validateInputs()

### work with predictions at estimates
predictions_base=predictions_base[["at_estimates"]]

### Compute change in probabilities
change=(predictions_new-predictions_base)/predictions_base

### Not interested in chosen alternative now, so drop last column
change=change[,-ncol(change)]

### First two columns (change in ID and task) also not needed
change=change[,-c(1,2)]

### Summary of changes (possible presence of NAs for unavailable alternatives)
summary(change)

### Look at mean changes for subsets of the data, ignoring NAs
colMeans(change,na.rm=TRUE)
colMeans(subset(change,database$mode_2023==0),na.rm=TRUE)
colMeans(subset(change,database$mode_2023==1),na.rm=TRUE)
colMeans(subset(change,(database$income_2<quantile(database$income_2,0.25))),na.rm=TRUE)
colMeans(subset(change,(database$income_2>=quantile(database$income_2,0.25))|(database$income_2<=quantile(database$income_2,0.75))),na.rm=TRUE)
colMeans(subset(change,(database$income_2>quantile(database$income_2,0.75))),na.rm=TRUE)

### Compute own elasticity for rail:
log(sum(predictions_new[,3])/sum(predictions_base[,3]))/log(1.5)

### Compute cross-elasticities for other modes
log(sum(predictions_new[,4])/sum(predictions_base[,4]))/log(1.5)
log(sum(predictions_new[,5])/sum(predictions_base[,5]))/log(1.5)

# ----------------------------------------------------------------- #
#---- RECOVERY OF SHARES FOR ALTERNATIVES IN DATABASE            ----
# ----------------------------------------------------------------- #

sharesTest_settings = list()
sharesTest_settings[["alternatives"]] = c(pt=1, cs=2, rs=3)
sharesTest_settings[["choiceVar"]]    = database$choice_best
sharesTest_settings[["subsamples"]]   = list(pt=(database$mode_2023==0),
                                             rs=(database$mode_2023==1))

apollo_sharesTest(model, apollo_probabilities, apollo_inputs, sharesTest_settings)

# ----------------------------------------------------------------- #
#---- MODEL PERFORMANCE IN SUBSETS OF DATABASE                   ----
# ----------------------------------------------------------------- #

fitsTest_settings = list()

fitsTest_settings[["subsamples"]] = list()
fitsTest_settings$subsamples[["pt"]] = database$mode_2023==0
fitsTest_settings$subsamples[["rs"]] = database$mode_2023==1

apollo_fitsTest(model,apollo_probabilities,apollo_inputs,fitsTest_settings)

# ----------------------------------------------------------------- #
#---- FUNCTIONS OF MODEL PARAMETERS                              ----
# ----------------------------------------------------------------- #

deltaMethod_settings=list(expression=c(VTT_car_min="b_tt_car/b_cost",
                                       VTT_car_hour="60*b_tt_car/b_cost",
                                       b_tt_diff_car_rail="b_tt_car-b_tt_rail"))
apollo_deltaMethod(model, deltaMethod_settings)

# ----------------------------------------------------------------- #
#---- switch off writing to file                                 ----
# ----------------------------------------------------------------- #

apollo_sink()











