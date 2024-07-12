# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 14:17:03 2022

@author: Alex
Model Recovery to be run on cluster without full crossvalidation
"""
import os
import sys


import numpy as np
import pandas as pd
import scipy.optimize as opt

import modelFit as mf


#load in data
data = pd.read_csv("./Data/e3_data.csv") 
fits = pd.concat([data['agent'],data['group']],axis=1)
fits = fits.drop_duplicates()
fits[["AS_fit","DB_fit","VS_fit","SG_fit"]] = np.nan

#collectors
AS_pars = []
DB_pars = []
VS_pars = []
SG_pars = []
nLL_AS = []
nLL_DB = []
nLL_VS = []
nLL_SG = []

g = np.unique(data.group)[int(sys.argv[1])]
np.random.seed(2023+g)
nmodel = 4
shor = len(np.unique(data["trial"]))
rounds = len(np.unique(data["round"]))
mod2num = {"AS":0,"DB":1,"VS":2,"SG":3}

path = "./Data/fitting_data/e3/"
if not os.path.exists(path):
    os.makedirs(path,exist_ok=True)

#go to selected group
subdata = data.loc[(data['group']==g)]
#GPR assumes mean=0
subdata.loc[:,"reward"] = subdata["reward"]-0.5
#iterate over agents in group
for ag in np.unique(subdata['agent']):
    fit = np.zeros((rounds,nmodel))
    r=0
    #iterate over rounds to leave one out as test
    for test in range(rounds):
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
        #iterate over models, fit on train, test on test, save outputs
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
    #save summed nLLs over rounds by model
    fits.loc[(fits['group']==g) & (fits["agent"]==ag),"AS_fit":"SG_fit"] = np.sum(fit,axis=0)
    fits.to_csv(path+"model_recov_{}.csv".format(g))

np.save(path+"AS_pars_{}.npy".format(g),AS_pars)
np.save(path+"DB_pars_{}.npy".format(g),DB_pars)
np.save(path+"VS_pars_{}.npy".format(g),VS_pars)
np.save(path+"SG_pars_{}.npy".format(g),SG_pars)
np.save(path+"nLL_AS_{}.npy".format(g),nLL_AS)
np.save(path+"nLL_DB_{}.npy".format(g),nLL_DB)
np.save(path+"nLL_VS_{}.npy".format(g),nLL_VS)
np.save(path+"nLL_SG_{}.npy".format(g),nLL_SG)