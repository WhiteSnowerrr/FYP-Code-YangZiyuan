

####  1  ####

rm(list = ls())
library(poLCA)
library(reticulate)

source_python('bior.py')
database = poRdata()[[1]]

l2f=function(t){
  out = t
  qup = unname(quantile(t, 0.75))
  qlow = unname(quantile(t, 0.25))
  for (i in 1:length(t)){
    if (t[i] > qup){
      out[i] = 3
    }else if (t[i] < qlow){
      out[i] = 1
    }else{
      out[i] = 2
    }
  }
  return(out)}

database$mode_2023 = database$mode_2023 + 1 #1 pt, 2rs
database$wfh_now = database$wfh_now + 1 
database$license = database$license + 1
database$qhousehold_1 = database$qhousehold_1 + 1
database$qhousehold_2 = database$qhousehold_2 + 1
database$qhousehold_3 = database$qhousehold_3 + 1
database$qhousehold_5 = database$qhousehold_5 + 1
database[database$qfamily_2==10,'qfamily_2'] = 0
database$qfamily_2 = database$qfamily_2 + 1
database[database$qfamily_3==10,'qfamily_3'] = 0
database$qfamily_3 = database$qfamily_3 + 1
database$qfamily_4 = database$qfamily_4 + 2
database[database$qfamily_4==12,'qfamily_4'] = 1
database[database$qfamily_5==10,'qfamily_5'] = 0
database$qfamily_5 = database$qfamily_5 + 1
database$qfamily_5.1 = database$qfamily_5.1 + 2
database[database$qfamily_5.1==12,'qfamily_5.1'] = 1
database$qfamily_7 = database$qfamily_7 + 2
database[database$qfamily_7==12,'qfamily_7'] = 1
database$qoffice_1 = database$qoffice_1 + 1
database$qoffice_2 = database$qoffice_2 + 1
database$qoffice_3 = database$qoffice_3 + 1
database$qoffice_4 = database$qoffice_4 + 1
database$att_pt_1 = database$att_pt_1 + 2
database$att_pt_2 = database$att_pt_2 + 2
database$att_pt_3 = database$att_pt_3 + 2
database$att_pt_4 = database$att_pt_4 + 2
database$pt_experience_1 = database$pt_experience_1 + 2
database[database$pt_experience_1==12,'pt_experience_1'] = 1
database$pt_experience_2 = database$pt_experience_2 + 2
database[database$pt_experience_2==12,'pt_experience_2'] = 1
database$pt_experience_3 = database$pt_experience_3 + 2
database[database$pt_experience_3==12,'pt_experience_3'] = 1
database$rs_experience_1 = database$rs_experience_1 + 2
database[database$rs_experience_1==12,'rs_experience_1'] = 1
database$rs_experience_2 = database$rs_experience_2 + 2
database[database$rs_experience_2==12,'rs_experience_2'] = 1
database$rs_experience_3 = database$rs_experience_3 + 2
database[database$rs_experience_3==12,'rs_experience_3'] = 1

database$income_1 = l2f(database$income_1)
database$income_2 = l2f(database$income_2)
database$floor_area = l2f(database$floor_area)
database$time_living = database$time_living - 2016

database[database$att_cs_1==10,'att_cs_1'] = 0
database$att_cs_1 = database$att_cs_1 + 1
database[database$att_cs_2==10,'att_cs_2'] = 0
database$att_cs_2 = database$att_cs_2 + 1
database[database$att_cs_3==10,'att_cs_3'] = 0
database$att_cs_3 = database$att_cs_3 + 1

database[database$att_shift_rs_1==10,'att_shift_rs_1'] = 0
database$att_shift_rs_1 = database$att_shift_rs_1 + 1
database[database$att_shift_rs_2==10,'att_shift_rs_2'] = 0
database$att_shift_rs_2 = database$att_shift_rs_2 + 1
database[database$att_shift_rs_3==10,'att_shift_rs_3'] = 0
database$att_shift_rs_3 = database$att_shift_rs_3 + 1

database$att_pt = 1
database[database$att_pt_1==3|database$att_pt_2==3|database$att_pt_3==3|database$att_pt_4==3,'att_pt'] = 2

database$att_shift_rs = 1
database[database$att_shift_rs_1==4|database$att_shift_rs_2==4|database$att_shift_rs_3==4,'att_shift_rs'] = 2

database$qfamily = 1
database[database$qfamily_2!=1|database$qfamily_3!=1|database$qfamily_5!=1,'qfamily'] = 2


database$private_car = 1
database[database$private_car_1_1==1&database$private_car_1_2==1,'private_car'] = 2
database[database$private_car_1_1==2&database$private_car_1_2==2,'private_car'] = 3
database[database$private_car_1_1==1&database$private_car_1_2==2,'private_car'] = 4

database$att_pt_5 = 1
database[database$att_pt_1==3|database$att_pt_2==3,'att_pt_5'] = 2

database$att_pt_6 = 1
database[database$att_pt_3==3|database$att_pt_4==3,'att_pt_6'] = 2



database[database$edu>2,'edu'] = 2
database[database$cs_rs_1_1<=2,'cs_rs_1_1'] = 1
database[database$cs_rs_1_1>=3,'cs_rs_1_1'] = 2

length((database$qfamily == 1)[(database$qfamily == 1)==TRUE])


length((database$att_pt == 1)[(database$att_pt == 1)==TRUE])



res = function(lcc){
  entropy<-function (p) sum(-p*log2(p))
  error_prior <- entropy(lcc$P) # Class proportions
  error_post <- mean(apply(lcc$posterior, 1, entropy),na.rm=TRUE)
  R2_entropy <- (error_prior - error_post) / error_prior
  acc = mean(apply(poLCA.posterior(lcc,lcc$y),1,max))
  t = c(BIC.m=lcc$bic,accuracy.8=acc,entropy.6=R2_entropy,LL.m=lcc$llik,AIC.m=lcc$aic)
  return(t)
}

lca = function(f){
  lc <- poLCA(f,database,nclass=2,nrep=10,maxiter=5000)
  t = res(lc)
  lc <- poLCA(f,database,nclass=3,nrep=10,maxiter=5000)
  t = rbind(t,res(lc))
  lc <- poLCA(f,database,nclass=4,nrep=10,maxiter=10000)
  t = rbind(t,res(lc))
  return(t)
}


####  2  ####

f <- cbind(mode_2023,qoffice_3,qhousehold_3,att_pt_5,att_pt_6,cs_rs_1_1,private_car)~1
#lca(f)

lc <- poLCA(f,database,nclass=2,nrep=10,maxiter=5000,graphs=TRUE)


library(ggplot2)
lcmodel <- reshape2::melt(lc$probs, level=2)
zp1 <- ggplot(lcmodel,aes(x = L2, y = value, fill = Var2))
zp1 <- zp1 + geom_bar(stat = "identity", position = "stack")
zp1 <- zp1 + facet_grid(Var1 ~ .) 
zp1 <- zp1 + scale_fill_brewer(type="seq", palette="Greys") +theme_bw()
zp1 <- zp1 + labs(x = "Fragebogenitems",y="Anteil der Item-\nAntwortkategorien", fill ="Antwortkategorien")
zp1 <- zp1 + theme( axis.text.y=element_blank(),
                    axis.ticks.y=element_blank(),                    
                    panel.grid.major.y=element_blank())
zp1 <- zp1 + guides(fill = guide_legend(reverse=TRUE))
print(zp1)

res(lc)

####  3  ####
f <- cbind(mode_2023,license,qfamily,qoffice_3,qhousehold_3,att_pt)~1


lc <- poLCA(f,database,nclass=2)
res(lc)

lc <- poLCA(f,database,nclass=3)
lc <- poLCA(f,database,nclass=4)



data(gss82)
f <- cbind(PURPOSE,ACCURACY,UNDERSTA,COOPERAT)~1
gss.lc2 <- poLCA(f,gss82,nclass=2)

a=data(gss82)





data(carcinoma)
f <- cbind(A,B,C,D,E,F,G)~1
lca2 <- poLCA(f,carcinoma,nclass=2) # log-likelihood: -317.2568
lca3 <- poLCA(f,carcinoma,nclass=3) # log-likelihood: -293.705
lca4 <- poLCA(f,carcinoma,nclass=4,nrep=10,maxiter=5000) # log-likelihood: -289.2858 

# Maximum entropy (if all cases equally dispersed)
log(prod(sapply(lca2$probs,ncol)))

# Sample entropy ("plug-in" estimator, or MLE)
p.hat <- lca2$predcell$observed/lca2$N
H.hat <- -sum(p.hat * log(p.hat))
H.hat   # 2.42

# Entropy of fitted latent class models
poLCA.entropy(lca2)/log(prod(sapply(lca2$probs,ncol)))
poLCA.entropy(lca3)/log(prod(sapply(lca3$probs,ncol)))
poLCA.entropy(lca4)/log(prod(sapply(lca4$probs,ncol)))
