# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 14:17:03 2022

@author: Alex
Model Fitting to be run on cluster
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.optimize as opt

import modelFit as mf


#get data
data = pd.read_csv("./Data/e2_data.csv") 
fits = pd.concat([data['agent'],data['group']],axis=1)
fits = fits.drop_duplicates()
fits[["AS_fit","DB_fit","VS_fit","SG_fit"]] = np.nan
#get storage ready
AS_pars = []
DB_pars = []
VS_pars = []
SG_pars = []
nLL_AS = []
nLL_DB = []
nLL_VS = []
nLL_SG = []
#for cluster: what group is being recovered
g = np.unique(data.group)[int(sys.argv[1])]
np.random.seed(12345+g)
#subset to group of interest + task type
subdata = data.loc[(data['group']==g)]
subdata = subdata.loc[(subdata["taskType"]=="individual")]
#mean needs to be 0 for GP, is .5 in data
subdata.loc[:,"reward"] = subdata["reward"]-0.5 

nmodel = 4
shor = len(np.unique(subdata["trial"]))
rounds = len(np.unique(subdata["round"]))
mod2num = {"AS":0,"DB":1,"VS":2,"SG":3}

path = "./data/fitting_data/e2_ind/"
if not os.path.exists(path):
    os.makedirs(path,exist_ok=True)

#goes through agents of the given group
for ag in np.unique(subdata['agent']):
    fit = np.zeros((rounds,nmodel))
    r=0
    #separates one round out as test for crossvalidation, and splits into individual (target) data and social data
    for test in np.unique(subdata["round"]):
        tardata = subdata.loc[(subdata['agent']==ag) &
                              (subdata['round']!=test),
                              ['round','trial','choice','reward','isRandom']]
        tardata = tardata.to_numpy()
        socdata = subdata.loc[(subdata['agent']!=ag) &
                              (subdata['round']!=test),
                              ['round','trial','choice','reward','isRandom','agent']]
        socdata = socdata.to_numpy()
        testtar = subdata.loc[(subdata['agent']==ag) &
                              (subdata['round']==test),
                              ['round','trial','choice','reward','isRandom']]
        testtar = testtar.to_numpy()
        testsoc = subdata.loc[(subdata['agent']!=ag) &
                              (subdata['round']==test),
                              ['round','trial','choice','reward','isRandom','agent']]
        testsoc = testsoc.to_numpy()
        #iterates over models to fit to; parameters get exponentiated in the fitting function
        for m in range(nmodel):
            if m == 0:
                print(g,ag,m,r)
                pars = opt.differential_evolution(mf.model_fit,[(-5,3),(-5,3),(-7.5,3)],(m,tardata,socdata,shor),maxiter=100)['x']
                #random choices don't affect fit, but will affect assumed chance level if not taken care of
                fit[r,m] = mf.model_fit(pars,m,testtar,testsoc,shor)
                nLL_AS.append(fit[r,m])
                pars = np.append([ag,g],pars)
                AS_pars.append([pars])
            elif m==1: #gamma can't be >1
                print(g,ag,m,r)
                pars = opt.differential_evolution(mf.model_fit,[(-5,3),(-5,3),(-7.5,3),(np.log(0.2142857),0)],(m,tardata,socdata,shor),maxiter=100)['x']
                fit[r,m] = mf.model_fit(pars,m,testtar,testsoc,shor)
                nLL_DB.append(fit[r,m])
                pars = np.append([ag,g],pars)
                DB_pars.append([pars])
            elif m==2:
                print(g,ag,m,r)
                pars = opt.differential_evolution(mf.model_fit,[(-5,3),(-5,3),(-7.5,3),(np.log(0.116),0)],(m,tardata,socdata,shor),maxiter=100)['x'] #,(np.log(1/14),0)
                fit[r,m] = mf.model_fit(pars,m,testtar,testsoc,shor)
                nLL_VS.append(fit[r,m])
                pars = np.append([ag,g],pars)
                VS_pars.append([pars])
            else:
                print(g,ag,m,r)
                pars = opt.differential_evolution(mf.model_fit,[(-5,3),(-5,3),(-7.5,3),(-5,np.log(19))],(m,tardata,socdata,shor),maxiter=100)['x']
                fit[r,m] = mf.model_fit(pars,m,testtar,testsoc,shor)
                nLL_SG.append(fit[r,m])
                pars = np.append([ag,g],pars)
                SG_pars.append([pars])
        r+=1
        print(r)
    fits.loc[(fits['group']==g) & (fits["agent"]==ag),"AS_fit":"SG_fit"] = np.sum(fit,axis=0)
    fits.to_csv(path+"model_recov_full_{}.csv".format(g))

np.save(path+"AS_pars_full_{}.npy".format(g),AS_pars)
np.save(path+"DB_pars_full_{}.npy".format(g),DB_pars)
np.save(path+"VS_pars_full_{}.npy".format(g),VS_pars)
np.save(path+"SG_pars_full_{}.npy".format(g),SG_pars)
np.save(path+"nLL_AS_full_{}.npy".format(g),nLL_AS)
np.save(path+"nLL_DB_full_{}.npy".format(g),nLL_DB)
np.save(path+"nLL_VS_full_{}.npy".format(g),nLL_VS)
np.save(path+"nLL_SG_full_{}.npy".format(g),nLL_SG)