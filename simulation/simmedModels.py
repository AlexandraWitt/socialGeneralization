# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 09:26:42 2023

@author: Alex
"""
import numpy as np
import json
import os

import modelSim as ms
##########################
#Testing grounds for agent-based simulations
# ##########################
path = 'C:/PhD/Code/socialGeneralization/environments/'
json_files = [file for file in os.listdir(path) if file.endswith('_canon.json')]
envList = []
for file in json_files:
    f=open(os.path.join(path, file))
    envList.append(json.load(f))


#Test ranges of lambda on beta on parameter ranges

groupedPars = np.load("./Data/fitted_pars_e1+2_soc.npy",allow_pickle=True)
parPool = groupedPars.flatten()
lamList = []
betList = []
tauList = []
epsList = []
for part in parPool:
    lamList.append(part["lambda"])
    betList.append(part["beta"])
    tauList.append(part["tau"])
    if part["eps_soc"]>0:
        epsList.append(part["eps_soc"])
        

def param_gen_resample_lam(nAgents,nGroups,models=None):
    """
    Generates parameter sets for simulations

    Parameters
    ----------
    nAgents : int
        Number of agents per group. Since we have 4 environments per set, the default is 4.
    nGroups : int
        Number of groups.
    hom : boolean, optional
        Should the groups be homogenous. The default is True.
    models : int in range(5), optional
        If specified, homogenous groups will only have the input model. 
        0=AS,1=DB,2=VS,3=SG,4=dummy model. The default is None.
        
    Returns
    -------
    List of list of dictionaries as used in model_sim.

    """
    par_names = ["lambda","beta","tau","gamma","alpha","eps_soc","dummy"] #needed for dicts later #
    all_pars = []
    #randomly select model unless given as an argument
    if models is None:
        models = np.random.randint(0,2,nGroups)
    else:
        assert models in range(0,5), "Model has to be between 0 and 4"
        models = np.ones(nGroups)*models
    for g in range(nGroups):
        model = models[g]
        all_pars.append([]) #set up list for parameter dictionary
        par = np.zeros((nAgents,len(par_names)))
        par[:,0] = np.random.uniform(0.5,2.5,(nAgents)) #lambda
        par[:,1] = np.random.choice(betList,nAgents)
        #par[:,1] = np.random.lognormal(-0.75,0.5,(nAgents))
        par[:,2] = np.random.choice(tauList,nAgents)  #tau
        if model==1:
            #par[:,5] = np.random.uniform(0,19,(nAgents)) #eps_soc
            par[:,5] = np.random.choice(epsList,nAgents)
        for ag in range(nAgents):
            pars = dict(zip(par_names,par[ag,:]))
            all_pars[g].append(pars)
    return(all_pars)

def param_gen_resample_bet(nAgents,nGroups,models=None):
    """
    Generates parameter sets for simulations

    Parameters
    ----------
    nAgents : int
        Number of agents per group. Since we have 4 environments per set, the default is 4.
    nGroups : int
        Number of groups.
    hom : boolean, optional
        Should the groups be homogenous. The default is True.
    models : int in range(5), optional
        If specified, homogenous groups will only have the input model. 
        0=AS,1=DB,2=VS,3=SG,4=dummy model. The default is None.
        
    Returns
    -------
    List of list of dictionaries as used in model_sim.

    """
    par_names = ["lambda","beta","tau","gamma","alpha","eps_soc","dummy"] #needed for dicts later #
    all_pars = []
    #randomly select model unless given as an argument
    if models is None:
        models = np.random.randint(0,2,nGroups)
    else:
        assert models in range(0,5), "Model has to be between 0 and 4"
        models = np.ones(nGroups)*models
    for g in range(nGroups):
        model = models[g]
        all_pars.append([]) #set up list for parameter dictionary
        par = np.zeros((nAgents,len(par_names)))
        par[:,0] = np.random.choice(lamList,nAgents) #lambda
        par[:,1] = np.random.uniform(0,2,nAgents)
        #par[:,1] = np.random.lognormal(-0.75,0.5,(nAgents))
        par[:,2] = np.random.choice(tauList,nAgents)  #tau
        if model==1:
            #par[:,5] = np.random.uniform(0,19,(nAgents)) #eps_soc
            par[:,5] = np.random.choice(epsList,nAgents)
        for ag in range(nAgents):
            pars = dict(zip(par_names,par[ag,:]))
            all_pars[g].append(pars)
    return(all_pars)


np.random.seed(2023)
pars_lam = param_gen_resample_lam(4, 2000)
c = ms.model_sim(pars_lam,envList,8,15,debug=False,payoff=True,memory=False)

c['model'] = np.select([ (c['alpha']==0) & (c['gamma']==0) & (c['eps_soc']==0)&(c['dummy']==0),
                        (c['gamma']!=0),
                        (c['alpha']!=0),
                        (c['eps_soc']!=0),
                        (c['dummy']!=0)],
                        ['AS','DB','VS','SG',"dummy"])

c["reward"] = c["reward"]+0.5
c["isRandom"] = 0

c.to_csv("./Data/GP_fitpars_lambdasim.csv")


np.random.seed(2023)
pars_bet = param_gen_resample_bet(4, 2000)
c = ms.model_sim(pars_bet,envList,8,15,debug=False,payoff=True,memory=False)

c['model'] = np.select([ (c['alpha']==0) & (c['gamma']==0) & (c['eps_soc']==0)&(c['dummy']==0),
                        (c['gamma']!=0),
                        (c['alpha']!=0),
                        (c['eps_soc']!=0),
                        (c['dummy']!=0)],
                        ['AS','DB','VS','SG',"dummy"])

c["reward"] = c["reward"]+0.5
c["isRandom"] = 0

c.to_csv("./Data/GP_fitpars_betasim.csv")



#Test model variants against each other
import modelSim_allVariants as ms
np.random.seed(2023)
pars = ms.param_gen_DB(4,1000)
c = ms.model_sim(pars,envList,8,15,debug=False)

c['model'] = np.select([ (c['alpha']==0) & (c['gamma']==0) & (c['eps_soc']==0) & (c['dummy']==0) & (c['alpha_prior']==0)& (c['alpha_mem']==0)& (c['alpha_agnostic']==0)& (c['gamma_val']==0) & (c['alpha_minmax']==0),
                        (c['gamma']!=0),
                        (c['alpha']!=0),
                        (c['eps_soc']!=0),
                        (c["dummy"]!=0),
                        (c["alpha_prior"]!=0),
                        (c["alpha_mem"]!=0),
                        (c["alpha_agnostic"]!=0),
                        (c["gamma_val"]!=0),
                        (c["alpha_minmax"]!=0)],
                        ['AS','DB','VS','SG',"dummy","VS_prior","VS_mem","VS_agnostic","DB_val","VS_minmax"])

c["reward"] = c["reward"]+0.5
c.to_csv("./Data/DB_payoff.csv")


np.random.seed(2023)
pars = ms.param_gen_VS(4,1000)
c = ms.model_sim(pars,envList,8,15,debug=False)

c['model'] = np.select([ (c['alpha']==0) & (c['gamma']==0) & (c['eps_soc']==0) & (c['dummy']==0) & (c['alpha_prior']==0)& (c['alpha_mem']==0)& (c['alpha_agnostic']==0)& (c['gamma_val']==0) & (c['alpha_minmax']==0),
                        (c['gamma']!=0),
                        (c['alpha']!=0),
                        (c['eps_soc']!=0),
                        (c["dummy"]!=0),
                        (c["alpha_prior"]!=0),
                        (c["alpha_mem"]!=0),
                        (c["alpha_agnostic"]!=0),
                        (c["gamma_val"]!=0),
                        (c["alpha_minmax"]!=0)],
                        ['AS','DB','VS','SG',"dummy","VS_prior","VS_mem","VS_agnostic","DB_val","VS_minmax"])

c["reward"] = c["reward"]+0.5
c.to_csv("./Data/VS_payoff.csv")

np.random.seed(2023)
pars = ms.param_gen_SG(4,1000)
c = ms.model_sim(pars,envList,8,15,debug=False)

c['model'] = np.select([ (c['alpha']==0) & (c['gamma']==0) & (c['eps_soc']==0) & (c['dummy']==0) & (c['alpha_prior']==0)& (c['alpha_mem']==0)& (c['alpha_agnostic']==0)& (c['gamma_val']==0) & (c['alpha_minmax']==0),
                        (c['gamma']!=0),
                        (c['alpha']!=0),
                        (c['eps_soc']!=0),
                        (c["dummy"]!=0),
                        (c["alpha_prior"]!=0),
                        (c["alpha_mem"]!=0),
                        (c["alpha_agnostic"]!=0),
                        (c["gamma_val"]!=0),
                        (c["alpha_minmax"]!=0)],
                        ['AS','DB','VS','SG',"dummy","VS_prior","VS_mem","VS_agnostic","DB_val","VS_minmax"])

c["reward"] = c["reward"]+0.5
c.to_csv("./Data/SG_vs_dummy.csv")