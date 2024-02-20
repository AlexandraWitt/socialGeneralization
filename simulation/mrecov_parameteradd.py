# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:08:31 2022

@author: Alex
"""

#getting model and parameter recovery datsets from model recovery

import os
import sys
import re

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

import numpy as np
import pandas as pd
from fnmatch import fnmatch

data = pd.read_csv("./Data/GP_het_parfits.csv") 
nRounds = np.max(data["round"]) + 1
nRounds = int(nRounds)
data = pd.concat([data['agent'],data['group'],data['model'],data['lambda'],data['beta'],data['tau'],
                  data['gamma'],data['alpha'],data['eps_soc']],axis=1)
data = data.drop_duplicates()


path = "./Data/recovery_data/mrecov"

#get all files per model
AS_files = [file for file in os.listdir(path) if fnmatch(file, 'AS*.npy')]
AS_files = sorted_alphanumeric(AS_files)
DB_files = [file for file in os.listdir(path) if fnmatch(file, 'DB*.npy')]
DB_files = sorted_alphanumeric(DB_files)
VS_files = [file for file in os.listdir(path) if fnmatch(file, 'VS*.npy')]
VS_files = sorted_alphanumeric(VS_files)
SG_files = [file for file in os.listdir(path) if fnmatch(file, 'SG*.npy')]
SG_files = sorted_alphanumeric(SG_files)

#load in parameters model by model, exponentiate the parameters, average over
#load + concatenate
AS_pars = np.squeeze(np.array([np.load(path + "/" + file,allow_pickle=True) for file in AS_files]))
AS_pars = np.concatenate(AS_pars,axis=0)
AS_pars = np.squeeze(AS_pars)
#exponentiate
AS_pars[:,-3:AS_pars.shape[1]] = np.exp(AS_pars[:,-3:AS_pars.shape[1]])
#average by round
AS_mean = np.mean(np.split(AS_pars,len(AS_pars)/nRounds),axis=1)

#load + concatenate
DB_pars = np.squeeze(np.array([np.load(path + "/" + file,allow_pickle=True) for file in DB_files]))
DB_pars = np.concatenate(DB_pars,axis=0)
DB_pars = np.squeeze(DB_pars)
#exponentiate
DB_pars[:,-4:DB_pars.shape[1]] = np.exp(DB_pars[:,-4:DB_pars.shape[1]])
#average by round
DB_mean = np.mean(np.split(DB_pars,len(DB_pars)/nRounds),axis=1)

#load + concatenate
VS_pars = np.squeeze(np.array([np.load(path + "/" + file,allow_pickle=True) for file in VS_files]))
VS_pars = np.concatenate(VS_pars,axis=0)
VS_pars = np.squeeze(VS_pars)
#exponentiate
VS_pars[:,-4:VS_pars.shape[1]] = np.exp(VS_pars[:,-4:VS_pars.shape[1]])
#average by round
VS_mean = np.mean(np.split(VS_pars,len(VS_pars)/nRounds),axis=1)

#load + concatenate
SG_pars = np.squeeze(np.array([np.load(path + "/" + file,allow_pickle=True) for file in SG_files]))
SG_pars = np.concatenate(SG_pars,axis=0)
SG_pars = np.squeeze(SG_pars)
#exponentiate
SG_pars[:,-4:SG_pars.shape[1]] = np.exp(SG_pars[:,-4:SG_pars.shape[1]])
#average by round
SG_mean = np.mean(np.split(SG_pars,len(SG_pars)/nRounds),axis=1)


#get nLLs
files = [path+"/"+file for file in os.listdir(path) if fnmatch(file, 'model*.csv')]
files = sorted_alphanumeric(files)
fit = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
fit = fit.dropna()
fit = fit.reset_index(drop=True)

#combine with dataset
fit = pd.merge(fit,data)
fit['lambda_fit']=fit['beta_fit']=fit['tau_fit']=fit['par_fit']=np.nan

#for model recovery, we care about best fitting parameters
for i in range(len(fit)):
    if np.argmin(fit.iloc[i,4:8])==0:
        fit.loc[i,'lambda_fit':'tau_fit'] = AS_mean[i,2:]
    elif np.argmin(fit.iloc[i,4:8])==1:
        fit.loc[i,'lambda_fit':'par_fit'] = DB_mean[i,2:]
    elif np.argmin(fit.iloc[i,4:8])==2:
        fit.loc[i,'lambda_fit':'par_fit'] = VS_mean[i,2:]
    elif np.argmin(fit.iloc[i,4:8])==3:
        fit.loc[i,'lambda_fit':'par_fit'] = SG_mean[i,2:]
        
fit.to_csv(path+"/mrecov.csv",index=False)

#for parameter recovery, we care about recovering correct parameters, even if not best fit
fit['lambda_fit']=fit['beta_fit']=fit['tau_fit']=fit['par_fit']=np.nan
for i in range(len(fit)):
    if fit.iloc[i,3]=="AS":
        fit.loc[i,'lambda_fit':'tau_fit'] = AS_mean[i,2:]
    elif fit.iloc[i,3]=="DB":
        fit.loc[i,'lambda_fit':'par_fit'] = DB_mean[i,2:]
    elif fit.iloc[i,3]=="VS":
        fit.loc[i,'lambda_fit':'par_fit'] = VS_mean[i,2:]
    elif fit.iloc[i,3]=="SG":
        fit.loc[i,'lambda_fit':'par_fit'] = SG_mean[i,2:]
        
fit.to_csv(path+"/precov.csv",index=False)
