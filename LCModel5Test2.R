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
  modelName       = "LC_tt22",
  modelDescr      = "LC model",
  indivID         = "ID", 
  outputDirectory = "output",
  nCores = parallel::detectCores() - 1
)


# ################################################################# #
#### LOAD DATA, APPLY ANY TRANSFORMATIONS, DEFINE MODEL PARAMETERS, DEFINE LATENT CLASS COMPONENTS, GROUP AND VALIDATE INPUTS ####
# ################################################################# #

### Loading data from package
source("Rbase.R")
database = RDataBase()

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
database$cs2_disin_s = database$cs2_disin
database$pt1_walk_s = database$pt1_walk/1000
database$cs2_walk_s = database$cs2_walk/1000
database$pt1_wait_s = database$pt1_wait/1000
database$rs3_wait_s = database$rs3_wait/1000



### Vector of parameters, including any that are kept fixed in estimation
apollo_beta=c(asc_pt                = 0,
              asc_cs                = 0,
              asc_rs                = 0,
             
              b_tt                  = 0,
              b_tt_shift            = 0,
              b_tt_shift2            = 0,
              
              b_cost                = 0,
              b_cost_shift          = 0,
              b_cost_shift2          = 0,
              
              b_trans               = 0,
              b_trans_shift         = 0,
              b_trans_shift2         = 0,
              b_crowd               = 0,
              b_crowd_shift         = 0,
              b_crowd_shift2         = 0,
              b_disin               = 0,
              b_disin_shift         = 0,
              b_disin_shift2         = 0,
              b_share               = 0,
              b_share_shift         = 0,
              b_share_shift2         = 0,
              
              b_pta                 = 0,
              b_pta_shift           = 0,
              b_pta_shift2           = 0,
              b_csa                 = 0,
              b_csa_shift           = 0,
              b_csa_shift2           = 0,
              b_rsa                 = 0,
              b_rsa_shift           = 0,
              b_rsa_shift2           = 0,
              
              delta_b               = -3,
              delta_c               = 0
              
              )

### Vector with names (in quotes) of parameters to be kept fixed at their starting value in apollo_beta, use apollo_beta_fixed = c() if none
apollo_fixed = c("asc_rs",
                 "b_rsa","b_rsa_shift","b_rsa_shift2"
                 )

### Read in starting values for at least some parameters from existing model output file
#apollo_beta = apollo_readBeta(apollo_beta, apollo_fixed, "LC_tt22", overwriteFixed=FALSE)



apollo_lcPars=function(apollo_beta, apollo_inputs){
  lcpars = list()
  lcpars[["b_tt_value"]] = list(b_tt, b_tt_shift, b_tt_shift2)
  lcpars[["b_cost_value"]] = list(b_cost, b_cost_shift, b_cost_shift2)
  lcpars[["b_trans_value"]] = list(b_trans, b_trans_shift, b_trans_shift2)
  lcpars[["b_crowd_value"]] = list(b_crowd, b_crowd_shift, b_crowd_shift2)
  lcpars[["b_disin_value"]] = list(b_disin, b_disin_shift, b_disin_shift2)
  lcpars[["b_share_value"]] = list(b_share, b_share_shift, b_share_shift2)
  lcpars[["b_pta_value"]] = list(b_pta, b_pta_shift, b_pta_shift2)
  lcpars[["b_csa_value"]] = list(b_csa, b_csa_shift, b_csa_shift2)
  lcpars[["b_rsa_value"]] = list(b_rsa, b_rsa_shift, b_rsa_shift2)
  

  ### Utilities of class allocation model
  V=list()
  V[["class_a"]] = 0
  V[["class_b"]] = delta_b
  V[["class_c"]] = delta_c
  
  ### Settings for class allocation models
  classAlloc_settings = list(
    classes      = c(class_a=1, class_b=2, class_c=3), 
    utilities    = V  
  )
  
  lcpars[["pi_values"]] = apollo_classAlloc(classAlloc_settings)
  
  return(lcpars)
}



apollo_inputs = apollo_validateInputs()


# ################################################################# #
#### DEFINE MODEL AND LIKELIHOOD FUNCTION, MODEL ESTIMATION ####
# ################################################################# #

apollo_probabilities=function(apollo_beta, apollo_inputs, functionality="estimate"){
  
  ### Attach inputs and detach after function exit
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))
  
  ### Create list of probabilities P
  P = list()
  
  ### Define settings for MNL model component
  mnl_settings = list(
    alternatives = c(pt=1, cs=2, rs=3),
    avail        = 1,
    choiceVar    = choice_best
  )
  
  ### Loop over classes
  for(s in 1:3){
    
    ### Compute class-specific utilities
    V=list()
    
    V[["pt"]]  = asc_pt + b_cost_value[[s]] * pt1_cost_s  + b_tt_value[[s]] * commuting_days_s * pt1_tt_s + b_pta_value[[s]] * age_s + b_crowd_value[[s]] * pt1_crowd_s + b_trans_value[[s]] * pt1_trans_s
    V[["cs"]]  = asc_cs + b_cost_value[[s]] * cs2_cost_s  + b_tt_value[[s]] * commuting_days_s * cs2_tt_s + b_csa_value[[s]] * age_s + b_disin_value[[s]] * cs2_disin_s
    V[["rs"]]  = asc_rs + b_cost_value[[s]] * rs3_cost_s  + b_tt_value[[s]] * commuting_days_s * rs3_tt_s + b_rsa_value[[s]] * age_s + b_tt_value[[s]] * b_share_value[[s]] * rs3_share_s
    
    
    
    mnl_settings$utilities     = V
    mnl_settings$componentName = paste0("Class_",s)
    
    ### Compute within-class choice probabilities using MNL model
    P[[paste0("Class_",s)]] = apollo_mnl(mnl_settings, functionality)
    
    ### Take product across observation for same individual
    P[[paste0("Class_",s)]] = apollo_panelProd(P[[paste0("Class_",s)]], apollo_inputs ,functionality)
    
  }
  
  ### Compute latent class model probabilities
  lc_settings  = list(inClassProb = P, classProb=pi_values)
  P[["model"]] = apollo_lc(lc_settings, apollo_inputs, functionality)
  
  ### Prepare and return outputs of function
  P = apollo_prepareProb(P, apollo_inputs, functionality)
  return(P)
}

con = c()
for (i in names(apollo_beta)){
  if (i %in% apollo_fixed) {}else{
  con = c(con,paste0(i," > -30"))
  con = c(con,paste0(i," < 30"))}
}

estimate_settings = list(constraints = con,
  maxIterations = 1000,
                         writeIter = FALSE)

model = apollo_estimate(apollo_beta, apollo_fixed, apollo_probabilities, apollo_inputs, estimate_settings)
#apollo_saveOutput(model)


# ################################################################# #
#### MODEL OUTPUTS                                               ####
# ################################################################# #

predictions_base = apollo_prediction(model, apollo_probabilities, apollo_inputs, prediction_settings=list(runs=30))
predictions_base=predictions_base[["model"]]
predictions_base=predictions_base[["at_estimates"]]
predictions_base[predictions_base$chosen>=predictions_base$pt & predictions_base$chosen>=predictions_base$cs & predictions_base$chosen>=predictions_base$rs, "chosen"]=1
predictions_base[predictions_base$chosen!=1, "chosen"]=0
sum(predictions_base$chosen)/nrow(predictions_base)

modelOutput_settings = list(printPVal=1
                            ,printClassical=FALSE
                            #,printOutliers=TRUE
                            )

apollo_modelOutput(model, modelOutput_settings)



# ################################################################# #
#### POST-PROCESSING                                            ####
# ################################################################# #

# ----------------------------------------------------------------- #
#---- LC PLOT                                                    ----
# ----------------------------------------------------------------- #

v = c('qhousehold_3', 
      'income_2', 
      'att_pt_1',  'att_pt_3', 'att_pt_4','qfamily_2'
      #, 'income_d'
) ###

#v = c('mode_2023', 'qoffice_3', 'qhousehold_3', 
#     'income_2', 'private_car', 'cs_rs_1_1', 
#    'qfamily_3', 'license', 'att_pt_5', 'att_pt_6', 
#   'age', 'gender', 'edu', 'cs_rs_1_2', 
#  'private_car_1_1', 'private_car_1_2', 
# 'qoffice_4', 'qfamily_2', 'qfamily_5' 
#, 'income_d'
#) ###


probs = list()
predictions_base = apollo_prediction(model, apollo_probabilities, apollo_inputs)

temp1 = database$income_2
temp2 = database$age
temp3 = database$income_d
database$income_2 = l2f2(database$income_2)
database$age = l2f2(database$age)
database$income_d = l2f2(database$income_d)

count = 1
del = c('delta_a'=0)
for (i in names(model[["estimate"]])){
  if (grepl("delta", i)) {
    del = c(del, model[["estimate"]][i])
    count = count + 1
  }
}

pc = c()
for (i in 1:length(del)){
  temp = exp(del[i])/sum(exp(del))
  names(temp) = paste0("Class_",i)
  pc = c(pc, temp)
}

pc2 = list()
for (i in 1:count){
  pc2[[paste0("Class_",i)]]=list()
}
for (i in 1:nrow(database)){
  for (j in 1:count){
    pc2[[paste0("Class_",j)]][i]=predictions_base[[paste0("Class_",j)]][["chosen"]][i]*pc[j]
  }
}

t = c()
for (i in 1:count){
  a = 0
  names(a) = paste0("Class_",i)
  t = c(t,a)
}
for (i in 1:nrow(database)){
  if (i%%8==1){
    for (j in 1:count){
      t[paste0("Class_",j)] = 0
    }
  }
  for (j in 1:count){
    t[paste0("Class_",j)] = t[paste0("Class_",j)] + pc2[[paste0("Class_",j)]][[i]]
  }
  if (i%%8==0){
    for (jj in (i-7):i){
      for (j in 1:count){
        pc2[[paste0("Class_",j)]][jj] = t[paste0("Class_",j)]/sum(t)
      }
    }
  }
}

for (i in v){
  rownames = c()
  for (j in 1:count){
    rownames = c(rownames, paste0("Class_",j))
  }
  colnames = paste0("Pr(",sort(unique(database[[i]])),")")
  table = matrix(0, nrow = count, ncol = length(sort(unique(database[[i]]))), dimnames=list(rownames, colnames))
  t = sort(unique(database[[i]]))
  for (j in 1:nrow(database)){
    for (jj in 1:count){
      table[jj,which(t==database[[i]][j])] = table[jj,which(t==database[[i]][j])] + pc2[[paste0("Class_",jj)]][[j]]
    }
  }
  for (j in 1:count){
    table[j,] = table[j,]/sum(table[j,])
  }
  probs[[i]] = table
}



library(ggplot2)
lcmodel <- reshape2::melt(probs, level=2)
zp1 <- ggplot(lcmodel,aes(x = L2, y = value, fill = Var2))
zp1 <- zp1 + geom_bar(stat = "identity", position = "stack")
zp1 <- zp1 + facet_grid(Var1 ~ .) 
zp1 <- zp1 + scale_fill_brewer(type="seq", palette="Greys") +theme_bw()
#zp1 <- zp1 + labs(x = "Fragebogenitems",y="Anteil der Item-\nAntwortkategorien", fill ="Antwortkategorien")
zp1 <- zp1 + theme( axis.text.y=element_blank(),
                    axis.ticks.y=element_blank(),  
                    axis.text.x = element_text(angle = 30,vjust = 0.85,hjust = 0.75),
                    panel.grid.major.y=element_blank())
zp1 <- zp1 + guides(fill = guide_legend(reverse=TRUE))
print(zp1)

wh = "age"
library(ggplot2)
lcmodel <- reshape2::melt(probs[wh], level=2)
lcmodel$L2 = wh
lcmodel$ly = 0
for (i in 1:count){
  t = lcmodel[lcmodel$Var1==paste0("Class_",i),]
  t = t[nrow(t):1,]
  for (j in 1:nrow(t)){
    if (j != 1){
    lcmodel[lcmodel$Var1==t$Var1&lcmodel$Var2==t$Var2[j],'ly'] = (sum(t$value[1:j])+sum(t$value[1:j-1]))/2
    }else{
      lcmodel[lcmodel$Var1==t$Var1&lcmodel$Var2==t$Var2[j],'ly'] = (sum(t$value[1:j])+0)/2
    }
  }
}
zp1 <- ggplot(lcmodel,aes(x = L2, y = value, fill = Var2))
zp1 <- zp1 + geom_bar(stat = "identity") + coord_polar('y')
zp1 <- zp1 + facet_grid(Var1 ~ .) 
zp1 <- zp1 +theme_bw()
zp1 <- zp1 + labs(fill = wh)
zp1 <- zp1 + theme(axis.title=element_blank(),axis.text=element_blank(),axis.ticks=element_blank())
zp1 <- zp1 + guides(fill = guide_legend(reverse=TRUE))  +
  geom_text(aes(x=L2,y=ly,label=paste(round(value*100,2),"%")),size=2)
print(zp1)



database$income_2 = temp1
database$age = temp2
database$income_d = temp3

probs$qhousehold_3
sum(probs$income_2[1,]*sort(unique(database[['income_2']])))
sum(probs$income_2[2,]*sort(unique(database[['income_2']])))


# ----------------------------------------------------------------- #
#---- LR TEST AGAINST SIMPLE MNL MODEL                           ----
# ----------------------------------------------------------------- #

### Example syntax with both models loaded from file
apollo_lrTest("MNL_SP", "MNL_SP_covariates")

### Example syntax with one model in memory
apollo_lrTest("MNL_SP", model)


# ----------------------------------------------------------------- #
#---- VALUE DIFFERENCE                                           ----
# ----------------------------------------------------------------- #

### The value of travel time
deltaMethod_settings=list(expression=c(VTT_Class_1="b_tt/b_cost",
                                       VTT_Class_2="(b_tt_shift)/(b_cost_shift)"))
apollo_deltaMethod(model, deltaMethod_settings)


# ----------------------------------------------------------------- #
#---- ELASTICITY DIFFERENCE PLOT                                 ----
# ----------------------------------------------------------------- #

x  = "pt1_cost_s" ###
scale = 1000 ###
dy    = 0.1 ###
y     = seq(0.1,10,dy) ###
z  = 'pt' ###


count = 1
for (i in names(model[["estimate"]])){
  if (grepl("delta", i)) {
    count = count + 1
  }
}

out = list()
for (i in 1:count){
  out[[paste0("Class_",i)]]=c(1)
}


predictions_base = apollo_prediction(model, apollo_probabilities, apollo_inputs, prediction_settings=list(runs=30))
for (i in y){
  database[,x] = database[,x]*i
  apollo_inputs = apollo_validateInputs()
  predictions_new = apollo_prediction(model, apollo_probabilities, apollo_inputs)
  database[,x] = database[,x]/i
  apollo_inputs = apollo_validateInputs()
  
  for (s in 1:count){
    
    base=predictions_base[[paste0("Class_",s)]][["at_estimates"]]
    new=predictions_new[[paste0("Class_",s)]]
    change=(new-base)/base
    change=change[,-ncol(change)]
    change=change[,-c(1,2)]
    
    out[[paste0("Class_",s)]] = c(out[[paste0("Class_",s)]],unname(colMeans(new,na.rm=TRUE)[z]))
    }
}
for (i in 1:count){
  out[[paste0("Class_",i)]]=out[[paste0("Class_",i)]][2:length(out[[paste0("Class_",i)]])]
}

library(ggplot2)

out[['x']]=y*scale ###
maX = max(out$x)
miX = min(out$x)

maY = -10000
miY = 10000
for (i in 1:count){
  maY = max(maY,max(out[[paste0("Class_",i)]]))
  miY = min(miY,min(out[[paste0("Class_",i)]]))
}
xscale = maX-miX
yscale = maY-miY

plotdata = data.frame(x=c(),y=c())
for (i in 1:count){
  plotdata = rbind(plotdata, data.frame(x=out[['x']],y=out[[paste0("Class_",i)]]))
}
t=c()
for (i in 1:count){
  t = c(t,paste0("Class_",i))
}
# Make the plot

plotdata$Group <- rep(t, each = length(out[['x']]))

p = ggplot(plotdata, aes(x = x, y = y, color = Group, linetype = Group)) + 
  geom_line(size = 1) +

  scale_linetype_manual(name = "Group",
                        values = rep(1, times=count), 
                        labels = t) +
  labs(x = paste0("Change in ",toupper(x)), y = paste0("Market share (",toupper(z),")")) + ### Change the labels
  xlim(miX*1.05, maX*1.05) +
  ylim(miY*0.95,maY*1.05) + 
  
  theme_bw() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.border = element_blank(), axis.line = element_line(colour = "black"))
# add the default price point on the line
#geom_vline(xintercept = 0.5, linetype = "dashed", color = "red") +

p


# ----------------------------------------------------------------- #
#---- REVENUE DIFFERENCE PLOT                                    ----
# ----------------------------------------------------------------- #

x     = "rs3_cost_s" ###
scale = 1000 ###
dy    = 0.1 ###
y     = seq(0.1,15,dy) ###
z     = 'rs' ###



count = 1
for (i in names(model[["estimate"]])){
  if (grepl("delta", i)) {
    count = count + 1
  }
}

out = list()
for (i in 1:count){
  out[[paste0("Class_",i)]]=c(1)
}

predictions_base = apollo_prediction(model, apollo_probabilities, apollo_inputs, prediction_settings=list(runs=30))
for (i in y){
  database[,x] = database[,x]*i
  apollo_inputs = apollo_validateInputs()
  predictions_new = apollo_prediction(model, apollo_probabilities, apollo_inputs)
  database[,x] = database[,x]/i
  apollo_inputs = apollo_validateInputs()
  
  for (s in 1:count){
    revenue = predictions_new[[paste0("Class_",s)]][,z]*database[,x]*scale*i
    base=predictions_base[[paste0("Class_",s)]][["at_estimates"]]
    new=predictions_new[[paste0("Class_",s)]]
    change=(new-base)/base
    change=change[,-ncol(change)]
    change=change[,-c(1,2)]
    
    out[[paste0("Class_",s)]] = c(out[[paste0("Class_",s)]],mean(revenue,na.rm=TRUE))
  }
}

for (i in 1:count){
  out[[paste0("Class_",i)]]=out[[paste0("Class_",i)]][2:length(out[[paste0("Class_",i)]])]
  s=out[[paste0("Class_",i)]][y==1]
  out[[paste0("Class_",i)]]=out[[paste0("Class_",i)]]/s
}


library(ggplot2)
out[['x']]=y ###
maX = max(out$x)
miX = min(out$x)

maY = -10000
miY = 10000
for (i in 1:count){
  maY = max(maY,max(out[[paste0("Class_",i)]]))
  miY = min(miY,min(out[[paste0("Class_",i)]]))
}
xscale = maX-miX
yscale = maY-miY

plotdata = data.frame(x=c(),y=c())
for (i in 1:count){
  plotdata = rbind(plotdata, data.frame(x=out[['x']],y=out[[paste0("Class_",i)]]))
}
t=c()
for (i in 1:count){
  t = c(t,paste0("Class_",i))
}
# Make the plot
plotdata$Group <- rep(t, each = length(out[['x']]))

p = ggplot(plotdata, aes(x = x, y = y, color = Group, linetype = Group)) + 
  geom_line(size = 1) +

  scale_linetype_manual(name = "Group",
                        values = rep(1, times=count),
                        labels = t) +
  labs(x = paste0("Change in ",toupper(x)," (scale)"), y = "Revenue") + ### Change the labels
  xlim(miX, maX*1.05) +
  coord_cartesian(clip = 'off',ylim=c(miY,maY*1.05))+
  theme_bw() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.border = element_blank(), axis.line = element_line(colour = "black")) +
  geom_vline(xintercept=1, linetype = 'dashed') + 
  annotate('text',x=1,y=miY-yscale*0.1,label='x=1',color='black')
# add the default price point on the line
#geom_vline(xintercept = 0.5, linetype = "dashed", color = "red") +

p

for (i in 1:count){
  p = p + geom_point(aes(x=out[['x']][out[[paste0("Class_",i)]]==max(out[[paste0("Class_",i)]])],y=out[[paste0("Class_",i)]][out[[paste0("Class_",i)]]==max(out[[paste0("Class_",i)]])]), color = 'red') 
}

p




p = p +
  geom_point(aes(x=A[A$y==max(A$y), "x"],y=A[A$y==max(A$y), "y"]), color = 'red') +
  geom_point(aes(x=B[B$y==max(B$y), "x"],y=B[B$y==max(B$y), "y"]), color = 'red') +
  annotate("text", x = A[A$y==max(A$y), "x"]+xscale*0.08, y = A[A$y==max(A$y), "y"]+yscale*0.05, label = la1) +
  annotate("text", x = B[B$y==max(B$y), "x"]+xscale*0.08, y = B[B$y==max(B$y), "y"]+yscale*0.05, label = la2)

p


# ----------------------------------------------------------------- #
#---- ELASTICITY DIFFERENCE                                      ----
# ----------------------------------------------------------------- #

x  = "rs3_cost_s"
z  = 'pt'
z2 = 'cs'
z3 = 'rs'

predictions_base = apollo_prediction(model, apollo_probabilities, apollo_inputs, prediction_settings=list(runs=30))

database[,x] = database[,x]*1.01 ### Now imagine the cost for rail increases by 1%
apollo_inputs = apollo_validateInputs()
predictions_new = apollo_prediction(model, apollo_probabilities, apollo_inputs, prediction_settings=list(runs=30))
database[,x] = database[,x]/1.01
apollo_inputs = apollo_validateInputs()

for (s in 1:3){
  base=predictions_base[[paste0("Class_",s)]][["at_estimates"]]
  new=predictions_new[[paste0("Class_",s)]][["at_estimates"]]
  
  print(paste0("Class_",s))
  print(log(sum(new[,z])/sum(base[,z]))/log(1.01)) ### Compute elasticities for rail
  print(log(sum(new[,z2])/sum(base[,z2]))/log(1.01)) ### Compute cross-elasticities for other modes
  print(log(sum(new[,z3])/sum(base[,z3]))/log(1.01))
  
  }


# ----------------------------------------------------------------- #
#---- RECOVERY OF SHARES FOR ALTERNATIVES IN DATABASE            ----
# ----------------------------------------------------------------- #

# test if ASCs need to shift
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


