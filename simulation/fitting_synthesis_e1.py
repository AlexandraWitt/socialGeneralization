# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:51:10 2023
Get cluster fitting results into one dataframe
@author: Alex
"""

# Data restructuring - match up parameter fits of best performing model and agent (fit)
# and get all model fits per agent and model (parfits)
import os

import numpy as np
import pandas as pd
import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
from fnmatch import fnmatch

data = pd.read_csv("./Data/e1_data.csv") 
fits = pd.concat([data['agent'],data['group']],axis=1)
fits = fits.drop_duplicates()
groupSize=len(fits)
path = "./Data/fitting_data/e1/"

#get overall fits
files = [path+"/"+file for file in os.listdir(path) if fnmatch(file, 'model*.csv')]
files = sorted_alphanumeric(files)
fit = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
fit = fit.dropna()
fit = fit.reset_index(drop=True)

#get roundwise nLLs per model
AS_nLLs = [file for file in os.listdir(path) if fnmatch(file, 'nLL_AS*.npy')]
AS_nLLs = sorted_alphanumeric(AS_nLLs)
AS_nLL = np.squeeze(np.array([np.load(path + "/" + file,allow_pickle=True) for file in AS_nLLs]))
AS_nLL = AS_nLL.flatten()
AS_nLL = np.split(AS_nLL,groupSize)
AS_nLL = np.squeeze(AS_nLL)

DB_nLLs = [file for file in os.listdir(path) if fnmatch(file, 'nLL_DB*.npy')]
DB_nLLs = sorted_alphanumeric(DB_nLLs)
DB_nLL = np.squeeze(np.array([np.load(path + "/" + file,allow_pickle=True) for file in DB_nLLs]))
DB_nLL = DB_nLL.flatten()
DB_nLL = np.split(DB_nLL,groupSize)
DB_nLL = np.squeeze(DB_nLL)

VS_nLLs = [file for file in os.listdir(path) if fnmatch(file, 'nLL_VS*.npy')]
VS_nLLs = sorted_alphanumeric(VS_nLLs)
VS_nLL = np.squeeze(np.array([np.load(path + "/" + file,allow_pickle=True) for file in VS_nLLs]))
VS_nLL = VS_nLL.flatten()
VS_nLL = np.split(VS_nLL,groupSize)
VS_nLL = np.squeeze(VS_nLL)

SG_nLLs = [file for file in os.listdir(path) if fnmatch(file, 'nLL_SG*.npy')]
SG_nLLs = sorted_alphanumeric(SG_nLLs)
SG_nLL = np.squeeze(np.array([np.load(path + "/" + file,allow_pickle=True) for file in SG_nLLs]))
SG_nLL = SG_nLL.flatten()
SG_nLL = np.split(SG_nLL,groupSize)
SG_nLL = np.squeeze(SG_nLL)

#for R² we only consider choices participant choices, so random nLL depends on number of random choices
randomCounts = data.groupby(['agent',"group","round"]).sum("isRandom")
randomCounts = randomCounts["isRandom"].to_numpy().reshape(8,groupSize)

randnLL = -np.log(1/121)*(15-randomCounts)
r2AS = 1-(AS_nLL/randnLL.T)
r2AS = np.mean(r2AS,axis=1)
fit["r2_AS"] = r2AS

r2DB = 1-(DB_nLL/randnLL.T)
r2DB = np.mean(r2DB,axis=1)
fit["r2_DB"] = r2DB

r2VS = 1-(VS_nLL/randnLL.T)
r2VS = np.mean(r2VS,axis=1)
fit["r2_VS"] = r2VS

r2SG = 1-(SG_nLL/randnLL.T)
r2SG = np.mean(r2SG,axis=1)
fit["r2_SG"] = r2SG


fit['id'] = range(len(fit))
fit = fit.rename(columns={"AS_fit":"fit_AS","DB_fit":"fit_DB","VS_fit":"fit_VS","SG_fit":"fit_SG"})
#uncomment for nLL version - non nLL have r² for all models, but nLL for only one, nLL is vice versa
#parfits = pd.wide_to_long(fit,"r2",["agent","group"],"model","_",'\\w+')
parfits = pd.wide_to_long(fit,"fit",["agent","group"],"model","_",'\\w+')
parfits = parfits.reset_index()

AS_files = [file for file in os.listdir(path) if fnmatch(file, 'AS*.npy')]
AS_files = sorted_alphanumeric(AS_files)
DB_files = [file for file in os.listdir(path) if fnmatch(file, 'DB*.npy')]
DB_files = sorted_alphanumeric(DB_files)
VS_files = [file for file in os.listdir(path) if fnmatch(file, 'VS*.npy')]
VS_files = sorted_alphanumeric(VS_files)
SG_files = [file for file in os.listdir(path) if fnmatch(file, 'SG*.npy')]
SG_files = sorted_alphanumeric(SG_files)

AS = np.concatenate(np.squeeze(np.array([np.load(path + "/" + file,allow_pickle=True) for file in AS_files])))
DB = np.concatenate(np.squeeze(np.array([np.load(path + "/" + file,allow_pickle=True) for file in DB_files])))
VS = np.concatenate(np.squeeze(np.array([np.load(path + "/" + file,allow_pickle=True) for file in VS_files])))
SG = np.concatenate(np.squeeze(np.array([np.load(path + "/" + file,allow_pickle=True) for file in SG_files])))

#parameter values are saved as logged
AS[:,2:] = np.exp(AS[:,2:])
DB[:,2:] = np.exp(DB[:,2:])
VS[:,2:] = np.exp(VS[:,2:])
SG[:,2:] = np.exp(SG[:,2:])

meanAS = np.mean(np.split(AS,len(fit)),axis=1)
meanDB = np.mean(np.split(DB,len(fit)),axis=1)
meanVS = np.mean(np.split(VS,len(fit)),axis=1)
meanSG = np.mean(np.split(SG,len(fit)),axis=1)

meanAS=pd.DataFrame(meanAS,columns=["agent","group","lambda","beta","tau"])
meanAS["model"]="AS"
meanDB=pd.DataFrame(meanDB,columns=["agent","group","lambda","beta","tau","par"])
meanDB["model"]="DB"
meanVS=pd.DataFrame(meanVS,columns=["agent","group","lambda","beta","tau","par"]) 
meanVS["model"]="VS"
meanSG=pd.DataFrame(meanSG,columns=["agent","group","lambda","beta","tau","par"])
meanSG["model"]="SG"

pars = pd.concat((meanAS,meanDB,meanVS,meanSG))
#get all fits incl. parameters
parfits = pd.merge(parfits,pars)
parfits.to_csv("./Data/fit+pars_e1.csv",index=False)

#only get best fit parameters
fit['model'] = fit['lambda']=fit['beta']=fit['tau']=fit['par']=np.nan
for i in range(len(fit)):
    highest = np.argmin(fit.loc[i,"fit_AS":"fit_SG"])
    if highest==0:
        fit.loc[i,"model"]="AS"
        fit.loc[i,"lambda":"tau"]=pars.loc[(pars["agent"]==fit.loc[i,"agent"]) & (pars["group"]==fit.loc[i,"group"]) &
                  (pars["model"]==fit.loc[i,"model"]),"lambda":"tau"].T.to_numpy()[:,0]
    elif highest==1:
        fit.loc[i,"model"]="DB"
        fit.loc[i,"lambda":"par"]=pars.loc[(pars["agent"]==fit.loc[i,"agent"]) & (pars["group"]==fit.loc[i,"group"]) &
                  (pars["model"]==fit.loc[i,"model"]),["lambda","beta","tau","par"]].T.to_numpy()[:,0]
    elif highest == 2:
        fit.loc[i,"model"]="VS"
        fit.loc[i,"lambda":"par"]=pars.loc[(pars["agent"]==fit.loc[i,"agent"]) & (pars["group"]==fit.loc[i,"group"]) &
                  (pars["model"]==fit.loc[i,"model"]),["lambda","beta","tau","par"]].T.to_numpy()[:,0]
    elif highest==3:
        fit.loc[i,"model"]="SG"
        fit.loc[i,"lambda":"par"]=pars.loc[(pars["agent"]==fit.loc[i,"agent"]) & (pars["group"]==fit.loc[i,"group"]) &
                  (pars["model"]==fit.loc[i,"model"]),["lambda","beta","tau","par"]].T.to_numpy()[:,0]
fit.to_csv("./Data/model_fits_e1.csv",index=False)


# compute and save protected exceedance probability
from groupBMC.groupBMC import GroupBMC

fit = pd.read_csv("./Data/model_fits_e1.csv")

LL = -fit.loc[:,['fit_AS','fit_DB','fit_VS','fit_SG']].to_numpy().T
result = GroupBMC(LL).get_result().protected_exceedance_probability
model = np.array(["AS","DB","VS","SG"])
data = {'model':model,'exceedance':result}
result = pd.DataFrame(data)
result.to_csv("./Data/pxp_e1.csv",index=False)
