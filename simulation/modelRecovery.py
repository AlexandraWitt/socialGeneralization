# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 14:17:03 2022

@author: Alex
Model Recovery to be run on cluster with crossvalidation
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.optimize as opt

import modelFit as mf



data = pd.read_csv("./Data/GP_het_parfits.csv") 
fits = pd.concat([data['agent'],data['group'],data['model']],axis=1)
fits = fits.drop_duplicates()
fits[["AS_fit","DB_fit","VS_fit","SG_fit"]] = np.nan

g = int(sys.argv[1])
np.random.seed(2023+g)
nmodel = len(np.unique(data['model']))
shor = len(np.unique(data["trial"]))
rounds = len(np.unique(data["round"]))


path = "./Data/recovery_data/mrecov"
if not os.path.exists(path):
    os.makedirs(path)
    
AS_pars = []
DB_pars = []
VS_pars = []
SG_pars = []

for ag in np.unique(data['agent']).astype(int):
    subdata = data.loc[(data['group']==g)]
    #data needs to be 0 mean
    if np.max(subdata['reward'])>0.6:
        subdata['reward'] = subdata['reward'] - 0.5 
    fit = np.zeros((rounds, nmodel))
    r = 0
    #iterate over rounds, leaving one out for testing
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
        #fit every model
        for m in range(nmodel):
            if m == 0:
                pars = opt.differential_evolution(mf.model_fit,[(-5,3),(-5,3),(-7.5,3)],(m,tardata,socdata,shor),maxiter=100)['x']
                fit[r,m] = mf.model_fit(pars,m,testtar,testsoc,shor)
                pars = np.append([ag,g],pars)
                AS_pars.append([pars])
            
            elif m==1:
                pars = opt.differential_evolution(mf.model_fit,[(-5,3),(-5,3),(-7.5,3),(np.log(0.2142857),0)],(m,tardata,socdata,shor),maxiter=100)['x']
                fit[r,m] = mf.model_fit(pars,m,testtar,testsoc,shor)
                pars = np.append([ag,g],pars)
                DB_pars.append([pars])
            
            elif m==2:
                pars = opt.differential_evolution(mf.model_fit,[(-5,3),(-5,3),(-7.5,3),(np.log(0.116),0)],(m,tardata,socdata,shor),maxiter=100)['x'] #,(np.log(1/14),0)
                fit[r,m] = mf.model_fit(pars,m,testtar,testsoc,shor)
                pars = np.append([ag,g],pars)
                VS_pars.append([pars])
                
            else:
                pars = opt.differential_evolution(mf.model_fit,[(-5,3),(-5,3),(-7.5,3),(-5,np.log(19))],(m,tardata,socdata,shor),maxiter=100)['x']
                fit[r,m] = mf.model_fit(pars,m,testtar,testsoc,shor)
                pars = np.append([ag,g],pars)
                SG_pars.append([pars])
                
        r+=1
    #save mean of test rounds per model
    fits.iloc[g*4+ag,3:3+nmodel] = np.mean(fit,axis=0)
    fits.to_csv(path+"/model_recov_poifit_{}.csv".format(g))
        
np.save(path+"/AS_pars_{}.npy".format(g),AS_pars)
np.save(path+"/DB_pars_{}.npy".format(g),DB_pars)
np.save(path+"/VS_pars_{}.npy".format(g),VS_pars)
np.save(path+"/SG_pars_{}.npy".format(g),SG_pars)
