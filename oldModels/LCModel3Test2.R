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
  modelName       = "LC_2",
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
database$commuting_days_s = database$commuting_days*4
database$pt1_cost_s = database$pt1_cost/1000
database$cs2_cost_s = database$cs2_cost/1000
database$rs3_cost_s = database$rs3_cost/1000
database$pt1_trans_s = database$pt1_trans/1
database$pt1_crowd_s = database$pt1_crowd/10
database$rs3_share_s = database$rs3_share/10
database$pt1_tt_s = database$pt1_tt/1000
database$pt1_wait_s = database$pt1_wait/1000
database$pt1_walk_s = database$pt1_walk/1000
database$cs2_tt_s = database$cs2_tt/1000
database$cs2_walk_s = database$cs2_walk/1000
database$rs3_tt_s = database$rs3_tt/1000
database$rs3_wait_s = database$rs3_wait/1000
database$cs2_disin_s = database$cs2_disin
database$income_2 = database$income_2/1000
database$mean_income = mean(database$income_2)


### Vector of parameters, including any that are kept fixed in estimation
apollo_beta=c(asc_pt                = 0,
              asc_cs                = 0,
              asc_rs                = 0,
             
              b_tt                  = 0,
              b_tt_shift            = 0,
              
              b_cost                = 0,
              b_cost_shift          = 0,
              
              b_trans               = 0,
              b_trans_shift         = 0,
              b_crowd               = 0,
              b_crowd_shift         = 0,
              b_disin               = 0,
              b_disin_shift         = 0,
              b_share               = 0,
              b_share_shift         = 0,
              
              #cost_income_elast     = 1,

              
              delta_a               = 0,
              #7 g_off3_a              = 0,
              #2 g_off4_a              = 0,
              g_hou_a               = 0,
              #g_incn_a              = 0,
              #4 g_incd_a              = 0,
              #g_car_a               = 0,
              #g_csrs1_a             = 0,
              #g_csrs2_a             = 0,
              #5 g_fam2_a              = 0,
              #3 g_fam3_a              = 0,
              #6 g_fam5_a              = 0,
              #g_lic_a               = 0,
              g_att1_a              = 0,
              g_att2_a              = 0
              #g_age_a               = 0,
              #1 g_gen_a               = 0,
              #g_edu_a               = 0
              )

### Vector with names (in quotes) of parameters to be kept fixed at their starting value in apollo_beta, use apollo_beta_fixed = c() if none
apollo_fixed = c("asc_pt"
                 )

### Read in starting values for at least some parameters from existing model output file
apollo_beta = apollo_readBeta(apollo_beta, apollo_fixed, "LC_2", overwriteFixed=FALSE)



apollo_lcPars=function(apollo_beta, apollo_inputs){
  lcpars = list()
  lcpars[["b_tt_value"]]    = list(b_tt, b_tt+b_tt_shift)
  lcpars[["b_cost_value"]]  = list(b_cost, b_cost+b_cost_shift)
  lcpars[["b_trans_value"]] = list(b_trans, b_trans+b_trans_shift)
  lcpars[["b_crowd_value"]] = list(b_crowd, b_crowd+b_crowd_shift)
  lcpars[["b_disin_value"]] = list(b_disin, b_disin+b_disin_shift)
  lcpars[["b_share_value"]] = list(b_share, b_share+b_share_shift)

  
  

  ### Utilities of class allocation model
  V=list()
  V[["class_a"]] = (delta_a + 
                      #g_off3_a*qoffice_3 + 
                      #g_off4_a*qoffice_4 + 
                      g_hou_a*qhousehold_3 + 
                      #g_incn_a*income_2 + 
                      #g_incd_a*income_d + 
                      #g_car_a*private_car + g_csrs1_a*cs_rs_1_1 + g_csrs2_a*cs_rs_1_2 + 
                      #g_fam2_a*qfamily_2 + 
                      #g_fam3_a*qfamily_3 + 
                      #g_fam5_a*qfamily_5 + 
                      #g_lic_a*license + 
                      g_att1_a*att_pt_5 + g_att2_a*att_pt_6 
                    #+ g_age_a*age 
                      #g_gen_a*gender + 
                      #g_edu_a*edu
                      )     
  V[["class_b"]] = 0
  
  ### Settings for class allocation models
  classAlloc_settings = list(
    classes      = c(class_a=1, class_b=2), 
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
  for(s in 1:2){
    
    ### Compute class-specific utilities
    V=list()
    
    V[["pt"]]  = asc_pt + b_cost_value[[s]] * pt1_cost_s  + b_tt_value[[s]] * commuting_days_s * (pt1_tt_s+pt1_walk_s+pt1_wait_s) + b_crowd_value[[s]] * pt1_crowd_s + b_trans_value[[s]] * pt1_trans_s
    V[["cs"]]  = asc_cs + b_cost_value[[s]] * cs2_cost_s  + b_tt_value[[s]] * commuting_days_s * (cs2_tt_s+cs2_walk_s) + b_disin_value[[s]] * cs2_disin_s
    V[["rs"]]  = asc_rs + b_cost_value[[s]] * rs3_cost_s  + b_tt_value[[s]] * commuting_days_s * (rs3_tt_s+rs3_wait_s) + b_share_value[[s]] * rs3_share_s
    
    
    
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
  con = c(con,paste0(i," > -100"))
  con = c(con,paste0(i," < 100"))}
}

estimate_settings = list(#constraints = con,
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

v = c('mode_2023', 'qhousehold_3', 
      'income_2', 'private_car', 'cs_rs_1_1', 
       'license', 'att_pt_5', 'att_pt_6', 
      'age', 'edu', 'cs_rs_1_2'
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
pc1 = (predictions_base[["model"]][,'pt']-predictions_base[["Class_2"]][,'pt'])/(predictions_base[["Class_1"]][,'pt']-predictions_base[["Class_2"]][,'pt'])
pc2 = 1-pc1

temp1 = database$income_2
temp2 = database$age
temp3 = database$income_d
database$income_2 = l2f2(database$income_2)
database$age = l2f2(database$age)
database$income_d = l2f2(database$income_d)

for (i in v){
  rownames = c("Class_1", "Class_2")
  colnames = paste0("Pr(",sort(unique(database[[i]])),")")
  table = matrix(0, nrow = 2, ncol = length(sort(unique(database[[i]]))), dimnames=list(rownames, colnames))
  t = sort(unique(database[[i]]))
  for (j in 1:nrow(database)){
      table[1,which(t==database[[i]][j])] = table[1,which(t==database[[i]][j])] + pc1[j]
      table[2,which(t==database[[i]][j])] = table[2,which(t==database[[i]][j])] + pc2[j]
  }
  table[1,] = table[1,]/sum(table[1,])
  table[2,] = table[2,]/sum(table[2,])
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

x  = "pt1_tt_s" ###
scale = 1000 ###
dy = 1 ###
y  = seq(-30,30,dy)/scale ###
z  = 'pt' ###


out1=c()
out2=c()
predictions_base = apollo_prediction(model, apollo_probabilities, apollo_inputs, prediction_settings=list(runs=30))
for (i in y){
  database[,x] = database[,x]+i
  apollo_inputs = apollo_validateInputs()
  predictions_new = apollo_prediction(model, apollo_probabilities, apollo_inputs)
  database[,x] = database[,x]-i
  apollo_inputs = apollo_validateInputs()
  
  for (s in 1:2){
    pi = (predictions_new[["model"]][,z]-predictions_new[[paste0("Class_",ifelse(s == 2, 1, 2))]][,z])/(predictions_new[[paste0("Class_",s)]][,z]-predictions_new[[paste0("Class_",ifelse(s == 2, 1, 2))]][,z])
    base=predictions_base[[paste0("Class_",s)]][["at_estimates"]]
    new=predictions_new[[paste0("Class_",s)]]
    change=(new-base)/base
    change=change[,-ncol(change)]
    change=change[,-c(1,2)]
    
    if (s==1){
      out1 = c(out1,unname(colMeans(new,na.rm=TRUE)[z]))
    } else {
      out2 = c(out2,unname(colMeans(new,na.rm=TRUE)[z]))
    }}
}

library(ggplot2)

A = data.frame(x=y*scale,y=out1)
B = data.frame(x=y*scale,y=out2)
maX = max(max(A$x),max(B$x))
maY = max(max(A$y),max(B$y))
miX = min(min(A$x),min(B$x))
miY = min(min(A$y),min(B$y))
xscale = maX-miX
yscale = maY-miY
la1 = toString(c(0,round(A[A$x==0, "y"],4)))
la1 = paste("(",la1,")",sep="")
la2 = toString(c(0,round(B[B$x==0, "y"],4)))
la2 = paste("(",la2,")",sep="")

#Calculate the intersection point of two lines A and B
inte = 1
if (length(unique((A$y-B$y)>=0))==1){
  inte = 0
}else{
  tt = (A$y-B$y)>=0
  
  i = 2
  while (i<=length(tt)){
    if (tt[i]!=tt[i-1]){
      break
    }
    i = i+1
  }
  t=A$x[i-1]
  e=0.00001*dy
  while (t<A$x[i]){
    temp = abs(approx(A$x, A$y, xout = t)$y - approx(B$x, B$y, xout = t)$y)
    t=t+e
    if (temp < 0.000001|temp < abs(approx(A$x, A$y, xout = t)$y - approx(B$x, B$y, xout = t)$y)){
      break
    }
  }
  intersectionX=t
  intersectionY=approx(A$x, A$y, xout = intersectionX)$y
  la = toString(c(round(intersectionX,4),round(intersectionY,4)))
  la = paste("(",la,")",sep="")
}

# Make the plot
plotdata = rbind(A,B)
plotdata$group <- rep(c("c1", "c2"), each = nrow(A))

p = ggplot(plotdata, aes(x = x, y = y, color = group, linetype = group)) + 
  geom_line(size = 1) +
  scale_color_manual(name = "Group",
                     values = c("c1" = 'blue', "c2" = 'green'), 
                     labels = c('Class 1', 'Class 2')) + 
  scale_linetype_manual(name = "Group",
                        values = c("c1" = 1, "c2" = 1), 
                        labels = c('Class 1', 'Class 2')) +
  labs(x = paste0("Change in ",toupper(x)), y = paste0("Market share (",toupper(z),")")) + ### Change the labels
  xlim(miX*1.05, maX*1.05) +
  ylim(miY*0.95,maY*1.05) + 
  
  theme_bw() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.border = element_blank(), axis.line = element_line(colour = "black"))
# add the default price point on the line
#geom_vline(xintercept = 0.5, linetype = "dashed", color = "red") +

if (inte==1){
  p = p + annotate("text", x = intersectionX+xscale*0.10, y = intersectionY+yscale*0.10, label = la) + 
    geom_line(aes(x=rep(c(intersectionX,intersectionX), each = nrow(A)),y=rep(c(miY*0.95,intersectionY), each = nrow(A))), linetype = 'dashed', color='black') +
    geom_line(aes(x=rep(c(miX*1.05,intersectionX), each = nrow(A)),y=rep(c(intersectionY,intersectionY), each = nrow(A))), linetype = 'dashed', color='black') +
    geom_point(aes(x=intersectionX,y=intersectionY), color = 'red')
}

p


# ----------------------------------------------------------------- #
#---- REVENUE DIFFERENCE PLOT                                    ----
# ----------------------------------------------------------------- #

x     = "rs3_cost_s" ###
scale = 1000 ###
dy    = 0.1 ###
y     = seq(0.1,15,dy) ###
z     = 'rs' ###


out=c()
out1=c()
out2=c()
predictions_base = apollo_prediction(model, apollo_probabilities, apollo_inputs, prediction_settings=list(runs=30))
for (i in y){
  database[,x] = database[,x]*i
  apollo_inputs = apollo_validateInputs()
  predictions_new = apollo_prediction(model, apollo_probabilities, apollo_inputs)
  database[,x] = database[,x]/i
  apollo_inputs = apollo_validateInputs()
  
  for (s in 1:2){
    pi = (predictions_new[["model"]][,z]-predictions_new[[paste0("Class_",ifelse(s == 2, 1, 2))]][,z])/(predictions_new[[paste0("Class_",s)]][,z]-predictions_new[[paste0("Class_",ifelse(s == 2, 1, 2))]][,z])
    revenue = predictions_new[[paste0("Class_",s)]][,z]*database[,x]*scale*i
    base=predictions_base[[paste0("Class_",s)]][["at_estimates"]]
    new=predictions_new[[paste0("Class_",s)]]
    change=(new-base)/base
    change=change[,-ncol(change)]
    change=change[,-c(1,2)]
    
    if (s==1){
      out1 = c(out1,mean(revenue,na.rm=TRUE))
    } else {
      out2 = c(out2,mean(revenue,na.rm=TRUE))
    }}
  revenue = predictions_new[["model"]][,z]*database[,x]*scale*i
  out = c(out,mean(revenue,na.rm=TRUE))
}

library(ggplot2)

s = out[y==1]
out = out/s
s1 = out1[y==1]
out1 = out1/s1
s2 = out2[y==1]
out2 = out2/s2

A = data.frame(x=y,y=out1)
B = data.frame(x=y,y=out2)
maX = max(max(A$x),max(B$x))
maY = max(max(A$y),max(B$y))
miX = min(min(A$x),min(B$x))
miY = min(min(A$y),min(B$y))
xscale = maX-miX
yscale = maY-miY
la1 = toString(c(round(A[A$y==max(A$y), "x"],2),round(A[A$y==max(A$y), "y"],2)))
la1 = paste("(",la1,")",sep="")
la2 = toString(c(round(B[B$y==max(B$y), "x"],2),round(B[B$y==max(B$y), "y"],2)))
la2 = paste("(",la2,")",sep="")
C = data.frame(x=y,y=out)
la = toString(c(round(C[C$y==max(C$y), "x"],2),round(C[C$y==max(C$y), "y"],2)))
la=paste('(',la,')',sep='')

# Make the plot
plotdata = rbind(A,B)
plotdata$group <- rep(c("c1", "c2"), each = nrow(A))

p = ggplot(plotdata, aes(x = x, y = y, color = group, linetype = group)) + 
  geom_line(size = 1) +
  scale_color_manual(name = "Group",
                     values = c("c1" = 'blue', "c2" = 'green'), 
                     labels = c('Class 1', 'Class 2')) + 
  scale_linetype_manual(name = "Group",
                        values = c("c1" = 1, "c2" = 1), 
                        labels = c('Class 1', 'Class 2')) +
  labs(x = paste0("Change in ",toupper(x)," (scale)"), y = "Revenue") + ### Change the labels
  xlim(miX, maX*1.05) +
  coord_cartesian(clip = 'off',ylim=c(miY,maY*1.05))+
  theme_bw() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.border = element_blank(), axis.line = element_line(colour = "black"))
# add the default price point on the line
#geom_vline(xintercept = 0.5, linetype = "dashed", color = "red") +

p = p + geom_vline(xintercept=1, linetype = 'dashed') + 
  annotate('text',x=1,y=min(A$y)-yscale*0.1,label='x=1',color='black') +
  geom_point(aes(x=A[A$y==max(A$y), "x"],y=A[A$y==max(A$y), "y"]), color = 'red') +
  geom_point(aes(x=B[B$y==max(B$y), "x"],y=B[B$y==max(B$y), "y"]), color = 'red') +
  annotate("text", x = A[A$y==max(A$y), "x"]+xscale*0.08, y = A[A$y==max(A$y), "y"]+yscale*0.05, label = la1) +
  annotate("text", x = B[B$y==max(B$y), "x"]+xscale*0.08, y = B[B$y==max(B$y), "y"]+yscale*0.05, label = la2)

p


# ----------------------------------------------------------------- #
#---- ELASTICITY DIFFERENCE                                      ----
# ----------------------------------------------------------------- #

x  = "pt1_cost_s"
z  = 'pt'
z2 = 'cs'
z3 = 'rs'

predictions_base = apollo_prediction(model, apollo_probabilities, apollo_inputs, prediction_settings=list(runs=30))

database[,x] = database[,x]*1.01 ### Now imagine the cost for rail increases by 1%
apollo_inputs = apollo_validateInputs()
predictions_new = apollo_prediction(model, apollo_probabilities, apollo_inputs, prediction_settings=list(runs=30))
database[,x] = database[,x]/1.01
apollo_inputs = apollo_validateInputs()

for (s in 1:2){
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


