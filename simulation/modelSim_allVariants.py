# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 13:27:47 2022

@author: Alex
This has all the basic functions to simulate data.
"""

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import json
import os
from scipy.spatial.distance import cdist

def ucb(pred,beta=0.5): 
    """    
    Parameters
    ----------
    pred : tuple of arrays in shape (2,n,1) 
        mean and variance values for n options (GPR prediction output).
    beta : numerical, optional
        governs exploration tendency. The default is 0.5.

    Returns
    -------
    Upper confidence bound of the data.
    """
    out = pred[0] + beta * np.sqrt(pred[1]) 
    return(out)

def rbf(x,y,l):
    D = cdist(x,y)**2
    return(np.exp(-D/(2*l**2)))

def GPR(Xstar,obs,rewards,socIndex, pars, baseEpsilon, k = rbf,vs_boost={}):
    """
    Parameters
    ----------
    Xstar : list
        The space we project to. Maps observation indices to coordinates as well.
    obs : list
        Indices of agent's observation history.
    rewards : list
        Agent's reward history.
    socIndex : list
        Relevant for SG model; 1 if observation is social, 0 if individual.
    pars : dict
        Model parameters.
    k : function, optional
        Kernel function. The default is rbf.
    vs_boost : dictionary, optional
        Priors for VS with memory. The default is {}.

    Returns
    -------
    List of posterior mean and variance.

    """
    choices = np.array([Xstar[x] for x in obs])
    K = k(choices,choices,pars['lambda']) #covariance of observations
    #add noise (social noise gets added to social observations)
    noise = np.ones_like(socIndex)*baseEpsilon + socIndex*pars['eps_soc']
    KK = K+np.identity(len(noise))*noise
    KK_inv = np.linalg.inv(KK)
    #if VS with memory, create prior to subtract from data and add back on later
    if pars['alpha']>0:
        prior = np.zeros(len(Xstar))
        for i in range(len(Xstar)):
            try:
                prior[i] = vs_boost[i]
            except KeyError:
                continue          
        rewards = rewards-prior[obs]
    Ky = np.dot(KK_inv,rewards)
    Kstarstar = k(Xstar,Xstar,pars['lambda']) #covariance of Xstar with itself
    Kstar = k(Xstar,choices,pars['lambda'])
    mu = np.dot(Kstar,Ky) #get mean
    cov = Kstarstar - np.dot(np.dot(Kstar,KK_inv),Kstar.T)#covariance
    var = np.diagonal(cov).copy() #and variance; if I don't copy, var isn't writeable, which breaks for SG
    if pars['alpha']>0: #in VS with memory, prior needs to be added to mean vector
        mu = mu+prior
    return([mu,var])

def model_sim(allParams, envList, rounds, shor, baseEpsilon=.0001, debug = False,prior_mean=0.5,prior_scale=1):
    """
    Simulates experimental data based on input specifications
    
    Parameters
    ----------
    allParams : list of list of dictionaries
        List of agent's parameter values. Number of groups is specified based on this.
    envList : list
        Reward environments.
    rounds : int
        Number of rounds (and thus environments used).
    shor : int
        Search horizon (number of datapoints per round).
    baseEpsilon : numeric, optional
        Assumed observation noise for individual observations. The default is .0001.
    debug : Boolean, optional
        Should GP posterior, value function and policy be plotted when running. The default is False.
    memory : Boolean, optional
        Should VS use info from all previous trials rather than t-1. The default is False.
    payoff : Boolean, optional
        Should VS use outcome information. The default is True.

    Returns
    -------
    A pandas dataframe with the simulated data (specific agents, environments, scores and parameters).
    """
    #set up lists to collect output info
    agent = []
    group = []
    r0und = [] 
    env = []
    trial = []
    choice = []
    reward = []
    lambda_ind = []
    dummy = []
    beta = []
    tau = []
    gamma = []
    alpha = []
    eps_soc = []
    alpha_prior = []
    alpha_mem = []
    alpha_agnostic = []
    gamma_val = []
    alpha_minmax = []
    
    nAgents = len(allParams[0])
    gridSize = len(envList[0][0])
    Xstar = np.array([(x, y) for x in range(np.sqrt(gridSize).astype(int)) for y in range(np.sqrt(gridSize).astype(int))])
    for g in range(len(allParams)): #iterate over groups
        pars = allParams[g]
        vals = np.zeros((gridSize,nAgents))
        policy = vals = np.zeros((gridSize,nAgents))
        envs = np.random.randint(0,len(envList[0]),(rounds)) #change here to use set envs
        #envs = list(range(rounds)) #same environments for debug purposes
        for r in range(rounds): 
            X = (np.zeros((shor,nAgents))*np.nan).astype(int)
            Y = np.zeros((shor,nAgents))*np.nan
            for t in range(shor):
                prediction = []
                if t==0:
                    #First choice is random
                    X[0,:] = np.random.randint(0,gridSize,(nAgents))
                    Y[0,:] = [(envList[ag][envs[r]][X[0,ag]]['payoff']-prior_mean)/prior_scale + np.random.normal(0,.01) for ag in range(nAgents)] 
                else:
                    for ag in range(nAgents):
                        obs = X[:t,ag] #self observations
                        rewards = Y[:t,ag]
                        socIndex = np.zeros_like(obs)
                        if (pars[ag]['eps_soc']>0) or (pars[ag]["dummy"]!=0):#social generalization uses social obs
                            obs = np.append(obs, X[:t,np.arange(nAgents)!=ag])
                            rewards = np.append(rewards, Y[:t,np.arange(nAgents)!=ag])
                            if pars[ag]["dummy"]!=0:
                                socIndex = np.zeros_like(obs)
                            else:
                                socIndex = np.append(socIndex, np.ones_like(X[:t,np.arange(nAgents)!=ag]))
                       
                        if (pars[ag]['alpha_prior']>0): #collect previous observations for prior in VS memory
                            rew_exp=np.mean(rewards) #value based on deviation from overall reward expectation
                            vs_boost = {i:np.mean(Y[X==i]-rew_exp) for i in np.unique(X[:t,np.arange(nAgents)!=ag])} #every observed tile is assigned mean observed value - rew_exp
                            prediction.append(GPR(Xstar,obs,rewards,socIndex,pars[ag],baseEpsilon,vs_boost = vs_boost))
                        else:
                            prediction.append(GPR(Xstar,obs,rewards,socIndex,pars[ag],baseEpsilon)) 
                        if debug and ag==0:
                            plt.imshow((prediction[ag][0]).reshape(int(np.sqrt(gridSize)),int(np.sqrt(gridSize))))
                            plt.title('Prediction - agent: ' + str(ag) + 'trial: '+ str(t))
                            plt.colorbar()
                            plt.gca().invert_yaxis()
                            plt.axis('off')
                            plt.show()
                            
                            # plt.imshow(prediction[ag][1].reshape(int(np.sqrt(gridSize)),int(np.sqrt(gridSize))))
                            # plt.title('Prediction uncertainty - agent: ' + str(ag) + 'trial: '+ str(t))
                            # plt.colorbar()
                            # plt.gca().invert_yaxis()
                            # plt.axis('off')
                            # plt.show()
                            

                        vals[:,ag] = np.squeeze(ucb(prediction[ag],pars[ag]['beta']))
                        #count occurrences of social choices in the previous round
                        bias = [(i,np.sum(X[t-1,np.arange(nAgents)!=ag]==i),
                                 np.mean(Y[t-1,np.arange(nAgents)!=ag][np.nonzero(X[t-1,np.arange(nAgents)!=ag]==i)])) #mean of Y at t-1 for each unique X at t-1
                                for i in np.unique(X[t-1,np.arange(nAgents)!=ag])]  
                        
                        if pars[ag]['alpha_mem']>0: #VS memory implementation: prediction error learning from posterior vs. soc info
                            vs_boost = [(i,np.sum(X[:t,np.arange(nAgents)!=ag]==i),np.mean(Y[X==i])) for i in np.unique(X[:t,np.arange(nAgents)!=ag])] 
                            #vs_boost = [(X[np.where(Y==i)],i) for i in np.max(Y[:t,np.arange(nAgents)!=ag],axis=0)]
                            for b in vs_boost:
                                #vals[b[0],ag] += pars[ag]['alpha']*b[1]*(b[2]-rew_exp)
                                vals[b[0],ag] = vals[b[0],ag] + pars[ag]['alpha_mem']*(b[1]*b[2]-vals[b[0],ag]) # np.mean(rewards)
                        #Value shaping
                        if pars[ag]['alpha']>0:
                            rew_exp=np.mean(rewards)
                            for b in bias:
                                vals[b[0],ag] += pars[ag]['alpha']*b[1]*(b[2]-rew_exp) #baby heuristic method

                        elif pars[ag]['alpha_agnostic']>0:
                            for b in bias:
                                    vals[b[0],ag] += pars[ag]['alpha']*b[1]
                        elif pars[ag]['alpha_minmax']>0:
                            rew_exp=np.mean(rewards)
                            minmax = [(i, np.sum(X[:t,np.arange(nAgents)!=ag]==i),np.mean(Y[:t,np.arange(nAgents)!=ag][np.nonzero(X[:t,np.arange(nAgents)!=ag]==i)]))
                                    for i in np.unique(np.concatenate((X[np.argmin(Y[:t,np.arange(nAgents)!=ag],axis=0),np.arange(nAgents)!=ag],X[np.argmax(Y[:t,np.arange(nAgents)!=ag],axis=0),np.arange(nAgents)!=ag])))]
                            for b in minmax:
                                vals[b[0],ag] += pars[ag]['alpha_minmax']*b[1]*(b[2]-rew_exp) #baby heuristic with memory
                        if debug and ag==0:
                            plt.imshow(vals[:,ag].reshape(int(np.sqrt(gridSize)),int(np.sqrt(gridSize))))
                            plt.title('Value - agent: ' + str(ag) + 'trial: '+ str(t))
                            plt.colorbar()
                            plt.gca().invert_yaxis()
                            plt.axis('off')
                            plt.show()
                        vals[:,ag] = vals[:,ag]-np.max(vals[:,ag]) #avoid overflow
                        #Policy
                        policy[:,ag] = np.exp(vals[:,ag]/pars[ag]['tau'])
                        #Decision biasing
                        if pars[ag]['gamma']>0:
                            socChoices = np.ones(gridSize)*.000000000001  
                            for b in bias:
                                socChoices[b[0]] += b[1]
                            socpolicy = socChoices/sum(socChoices)
                            #mixture policy
                            policy[:,ag] = ((1 - pars[ag]['gamma'])*policy[:,ag]) + (pars[ag]['gamma'] * socpolicy)
                        elif pars[ag]['gamma_val']>0:
                            rew_exp=np.mean(rewards) #individual reward experience
                            socChoices = np.ones(gridSize)*.000000000001  
                            for b in bias:#subtracting from policy overcomplicates things; only boost helpful info#TODO: matrix notation
                                if b[2]-rew_exp<0:
                                    continue
                                socChoices[b[0]] += b[1]*(b[2]-rew_exp)
                        if debug and ag==0:
                            plt.imshow((policy[:,ag]/np.sum(policy[:,ag])).reshape(int(np.sqrt(gridSize)),int(np.sqrt(gridSize))))
                            plt.colorbar()
                            plt.title('Policy - agent: ' + str(ag) + 'trial: '+ str(t))
                            plt.gca().invert_yaxis()
                            plt.axis('off')
                            plt.show()
                        #policy[:,ag] = [0.001 if x/np.sum(policy[:,ag])==0 else policy[i,ag] for i,x in enumerate(policy[:,ag])] #obsolete(?) underflow prevention
                    #Choice
                    policy = policy/np.sum(policy,axis=0) 
                    X[t,:] = [np.random.choice(gridSize, p = policy[:,ag]) for ag in range(nAgents)]
                    Y[t,:] = [(envList[ag][envs[r]][X[t,ag]]['payoff']-prior_mean)/prior_scale +np.random.normal(0,.01) for ag in range(nAgents)]
            for ag in range(nAgents):
                agent.append(np.ones((shor,1))*ag)
                group.append(np.ones((shor,1))*g)
                r0und.append(np.ones((shor,1))*r)
                env.append(np.ones((shor,1))*envs[r])
                trial.append(np.arange(shor))
                choice.append(X[:,ag])
                reward.append(Y[:,ag])
                lambda_ind.append(np.ones((shor,1))*pars[ag]['lambda'])
                beta.append(np.ones((shor,1))*pars[ag]['beta'])
                tau.append(np.ones((shor,1))*pars[ag]['tau'])
                gamma.append(np.ones((shor,1))*pars[ag]['gamma'])
                alpha.append(np.ones((shor,1))*pars[ag]['alpha'])
                eps_soc.append(np.ones((shor,1))*pars[ag]['eps_soc'])
                dummy.append(np.ones((shor,1))*pars[ag]["dummy"])
                alpha_prior.append(np.ones((shor,1))*pars[ag]["alpha_prior"])
                alpha_mem.append(np.ones((shor,1))*pars[ag]["alpha_mem"])
                alpha_agnostic.append(np.ones((shor,1))*pars[ag]["alpha_agnostic"])
                gamma_val.append(np.ones((shor,1))*pars[ag]["gamma_val"])
                alpha_minmax.append(np.ones((shor,1))*pars[ag]["alpha_minmax"])
    
    data = np.column_stack((np.concatenate(agent),np.concatenate(group),np.concatenate(r0und),np.concatenate(env),np.concatenate(trial),
            np.concatenate(choice),np.concatenate(reward),np.concatenate(lambda_ind),
            np.concatenate(beta),np.concatenate(tau),np.concatenate(gamma),np.concatenate(alpha),np.concatenate(eps_soc),np.concatenate(dummy),
            np.concatenate(alpha_prior),np.concatenate(alpha_mem),np.concatenate(alpha_agnostic),np.concatenate(gamma_val),np.concatenate(alpha_minmax)))
    agentData = pd.DataFrame(data,columns=('agent','group','round','env','trial','choice','reward','lambda','beta',
                                           'tau','gamma','alpha','eps_soc',"dummy","alpha_prior","alpha_mem",
                                           "alpha_agnostic","gamma_val","alpha_minmax"))      
    return(agentData)
    
def param_gen(nAgents,nGroups,hom=True,models=None):
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
    par_names = ["lambda","beta","tau","gamma","alpha","eps_soc","dummy","alpha_prior",
                 "alpha_mem","alpha_agnostic","gamma_val","alpha_minmax"] #needed for dicts later
    all_pars = []
    if hom:
        #randomly select model unless given as an argument
        if models is None:
            models = np.random.randint(0,10,nGroups)
        else:
            assert models in range(0,10), "Model has to be between 0 and 9"
            models = np.ones(nGroups)*models
        for g in range(nGroups):
            model = models[g]
            all_pars.append([]) #set up list for parameter dictionary
            par = np.zeros((nAgents,len(par_names)))
            par[:,0] = np.random.lognormal(-0.75,0.5,(nAgents)) #lambda
            par[:,1] = np.random.lognormal(-0.75,0.5,(nAgents)) #beta
            par[:,2] = np.random.lognormal(-4.5,0.9,(nAgents))  #tau
            if model==1:
                par[:,3] = np.random.uniform(0,1,(nAgents))        #gamma
            elif model==2:
                par[:,4] = np.random.exponential(6.5,(nAgents))   #alpha 
                #par[:,4] = np.random.lognormal(1,0.5,(nAgents)) +4 #tighter value range
            elif model==3:
                par[:,5] = np.random.exponential(2,(nAgents)) #eps_soc
            elif model == 4:
                par[:,6] = np.ones(nAgents) #dummy variable for a model test
            elif model == 5:
                par[:,7] = np.ones(nAgents) #alpha prior
            elif model == 6:
                par[:,8] = np.random.uniform(0,1,(nAgents))   #alpha_mem
            elif model == 7:
                par[:,9] = np.random.exponential(1,(nAgents))   #alpha_agnostic
            elif model == 8:
                par[:,10] = np.random.uniform(0,1,(nAgents)) #gamma_val
            elif model == 9:
                par[:,11] =  np.random.exponential(0.2,(nAgents)) #alpha_minmax
            for ag in range(nAgents):
                pars = dict(zip(par_names,par[ag,:]))
                all_pars[g].append(pars)
    else:
        for g in range(nGroups):
            all_pars.append([])
            model = np.random.randint(0,10,(nAgents))
            par = np.zeros((nAgents,len(par_names)))
            par[:,0] = np.random.lognormal(-0.75,0.5,(nAgents)) #lambda
            #par[:,0] = np.ones((nAgents))*2
            par[:,1] = np.random.lognormal(-0.75,0.5,(nAgents)) #beta
            par[:,2] = np.random.lognormal(-4.5,0.9,(nAgents))  #tau
            for ag in range(nAgents):
                if model[ag]==1:
                    par[ag,3] = np.random.uniform(0,1,(1))        #gamma
                elif model[ag]==2:
                    par[ag,4] = np.random.exponential(6.5,(1))     #alpha
                elif model[ag]==3:
                      par[ag,5] = np.random.exponential(2,(1)) #eps_soc
                elif model[ag]==4:
                    par[ag,6] = 1 #dummy variable for a model test
                elif model[ag]==5:
                    par[ag,7] = 1 #alpha_prior
                elif model[ag]==6:
                    par[ag,8] = np.random.uniform(0,1,(1)) #alpha_mem
                elif model[ag]==7:
                    par[ag,9] = np.random.exponential(1,(1)) #alpha_agnostic
                elif model[ag]==8:
                    par[ag,10] = np.random.uniform(0,1,(1))  #gamma_val
                elif model[ag]==9:
                    par[ag,11] = np.random.exponential(0.2,(1)) #alpha_minmax
                pars = dict(zip(par_names,par[ag,:]))
                all_pars[g].append(pars)
    return(all_pars)


def param_gen_DB(nAgents,nGroups):
    #generates groups of 2 AS + one no payoff and one payoff DB agent
    par_names = ["lambda","beta","tau","gamma","alpha","eps_soc","dummy","alpha_prior",
                 "alpha_mem","alpha_agnostic","gamma_val","alpha_minmax"]
    all_pars = []
    for g in range(nGroups):
        all_pars.append([])
        model = np.array([0,0,1,2])
        par = np.zeros((nAgents,len(par_names)))
        par[:,0] = np.random.lognormal(-0.75,0.5,(nAgents)) #lambda
        #par[:,0] = np.ones((nAgents))*2
        par[:,1] = np.random.lognormal(-0.75,0.5,(nAgents)) #beta
        par[:,2] = np.random.lognormal(-2.5,0.5,(nAgents))  #tau
        for ag in range(nAgents):
            if model[ag]==1:
                par[ag,3] = np.random.uniform(0,1,(1)) #no value DB
            elif model[ag]==2:
                par[ag,10] = np.random.uniform(0,1,(1)) #gamma_val
            pars = dict(zip(par_names,par[ag,:]))
            all_pars[g].append(pars)
    return(all_pars)


def param_gen_VS(nAgents,nGroups):
    #generates groups of 2 AS + one no payoff and one payoff VS agent
    par_names = ["lambda","beta","tau","gamma","alpha","eps_soc","dummy","alpha_prior",
                 "alpha_mem","alpha_agnostic","gamma_val","alpha_minmax"]
    all_pars = []
    for g in range(nGroups):
        all_pars.append([])
        model = np.array([0,0,1,2])
        par = np.zeros((nAgents,len(par_names)))
        par[:,0] = np.random.lognormal(-0.75,0.5,(nAgents)) #lambda
        #par[:,0] = np.ones((nAgents))*2
        par[:,1] = np.random.lognormal(-0.75,0.5,(nAgents)) #beta
        par[:,2] = np.random.lognormal(-2.5,0.5,(nAgents))  #tau
        for ag in range(nAgents):
            if model[ag]==1:
                par[ag,8] = np.random.uniform(0,1,(1)) #VS value sensitive
            elif model[ag]==2:
                par[ag,9] = np.random.exponential(1,(1)) #VS agnostic
            pars = dict(zip(par_names,par[ag,:]))
            all_pars[g].append(pars)
    return(all_pars)


def param_gen_SG(nAgents,nGroups):
    #generates groups of 2 AS + one socially discrimintate and one non discriminate SG agent
    par_names = ["lambda","beta","tau","gamma","alpha","eps_soc","dummy","alpha_prior",
                 "alpha_mem","alpha_agnostic","gamma_val","alpha_minmax"]
    all_pars = []
    for g in range(nGroups):
        all_pars.append([])
        model = np.array([0,0,3,4])
        par = np.zeros((nAgents,len(par_names)))
        par[:,0] = np.random.lognormal(-0.75,0.5,(nAgents)) #lambda
        #par[:,0] = np.ones((nAgents))*2
        par[:,1] = np.random.lognormal(-0.75,0.5,(nAgents)) #beta
        par[:,2] = np.random.lognormal(-2.5,0.5,(nAgents))  #tau
        for ag in range(nAgents):
            if model[ag]==3:
                par[ag,5] = np.random.exponential(2) #eps_soc
            elif model[ag] == 4:
                par[ag,6] = 1 #dummy variable for a model test
            pars = dict(zip(par_names,par[ag,:]))
            all_pars[g].append(pars)
    return(all_pars)

def pop_gen(popSize, models):
    """
    Generates populations of size popSize and including
    strategies in models for evolutionary simulations.

    Parameters
    ----------
    popSize : int
        Population size.
    models : list of int range(4)
        Specifies models desired in the population.
        0=AS,1=DB,2=VS,3=SG

    Returns
    -------
    List of list of dict.

    """
    pop = []
    subpop = popSize//len(models) #get even proportion of desired models
    for i in range(4):
        if i in models:
            pop.extend(param_gen(1,subpop,models=i)) 
    while len(pop)<popSize: #if even proportion can't result in correct population size, populate with random model from list of desired
        pop.extend(param_gen(1,1,models = np.random.choice(models)))
    return(pop)