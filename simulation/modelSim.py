# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 13:27:47 2022

@author: Alex
This has all the basic functions to simulate data.
"""

import numpy as np
import pandas as pd
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
    Ky = np.dot(KK_inv,rewards)
    Kstarstar = k(Xstar,Xstar,pars['lambda']) #covariance of Xstar with itself
    Kstar = k(Xstar,choices,pars['lambda'])
    mu = np.dot(Kstar,Ky) #get mean
    cov = Kstarstar - np.dot(np.dot(Kstar,KK_inv),Kstar.T)#covariance
    var = np.diagonal(cov).copy() #and variance; if I don't copy, var isn't writeable, which breaks for SG
    return([mu,var])

def model_sim(allParams, envList, rounds, shor, baseEpsilon=.0001, debug = False, memory=False,payoff=True,prior_mean=0.5,prior_scale=1):
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
    #simulation parameters
    nAgents = len(allParams[0])
    gridSize = len(envList[0][0])
    Xstar = np.array([(x, y) for x in range(np.sqrt(gridSize).astype(int)) for y in range(np.sqrt(gridSize).astype(int))])
    for g in range(len(allParams)): #iterate over groups
        #get parameters and set up collectors for value and policy
        pars = allParams[g]
        vals = np.zeros((gridSize,nAgents))
        policy = vals = np.zeros((gridSize,nAgents))
        #sample random set of environments
        envs = np.random.randint(0,len(envList[0]),(rounds)) #change here to use set envs
        for r in range(rounds): 
            #in every round, reset observations and rewards
            X = np.zeros((shor,nAgents)).astype(int)
            Y = np.zeros((shor,nAgents))
            for t in range(shor):
                #on each trial, reset prediction
                prediction = []
                if t==0:
                    #First choice is random
                    X[0,:] = np.random.randint(0,gridSize,(nAgents))
                    Y[0,:] = [(envList[ag][envs[r]][X[0,ag]]['payoff']-prior_mean)/prior_scale + np.random.normal(0,.01) for ag in range(nAgents)] 
                else:
                    for ag in range(nAgents):
                        obs = X[:t,ag] #self observations
                        rewards = Y[:t,ag] #self rewards
                        socIndex = np.zeros_like(obs) #individual info
                        #social generalization uses social obs in GP
                        if (pars[ag]['eps_soc']>0) or (pars[ag]["dummy"]!=0):
                            obs = np.append(obs, X[:t,np.arange(nAgents)!=ag]) #social observations
                            rewards = np.append(rewards, Y[:t,np.arange(nAgents)!=ag]) #social rewards
                            if pars[ag]["dummy"]!=0:
                                socIndex = np.zeros_like(obs) #dummy model treats everything the same
                            else:
                                socIndex = np.append(socIndex, np.ones_like(X[:t,np.arange(nAgents)!=ag])) #otherwise flag as social observations
                       
                        prediction.append(GPR(Xstar,obs,rewards,socIndex,pars[ag],baseEpsilon)) 
                        #values from UCB
                        vals[:,ag] = np.squeeze(ucb(prediction[ag],pars[ag]['beta']))
                        #count occurrences of social choices in the previous round
                        bias = [(i,np.sum(X[t-1,np.arange(nAgents)!=ag]==i),
                                 np.mean(Y[t-1,np.arange(nAgents)!=ag][np.nonzero(X[t-1,np.arange(nAgents)!=ag]==i)])) #mean of Y at t-1 for each unique X at t-1
                                for i in np.unique(X[t-1,np.arange(nAgents)!=ag])]  
                        
                        #Value shaping
                        if pars[ag]['alpha']>0: #prediction error learning from individual value vs. soc info
                            vs_boost = [(i,np.sum(X[:t,np.arange(nAgents)!=ag]==i),np.mean(Y[X==i])) for i in np.unique(X[:t,np.arange(nAgents)!=ag])] 
                            #vs_boost = [(X[np.where(Y==i)],i) for i in np.max(Y[:t,np.arange(nAgents)!=ag],axis=0)]
                            for b in vs_boost:
                                #vals[b[0],ag] += pars[ag]['alpha']*b[1]*(b[2]-rew_exp)
                                vals[b[0],ag] = vals[b[0],ag] + pars[ag]['alpha']*(b[1]*b[2]-vals[b[0],ag]) # np.mean(rewards)

                        #avoid overflow
                        vals[:,ag] = vals[:,ag]-np.max(vals[:,ag]) 
                        #Policy
                        policy[:,ag] = np.exp(vals[:,ag]/pars[ag]['tau'])
                        #Decision biasing
                        if pars[ag]['gamma']>0:
                                #payoff bias (generally unused)
                                rew_exp=np.mean(rewards) #individual reward experience
                                socChoices = np.ones(gridSize)*.000000000001  #just zeros breaks policy sometimes
                                if payoff:
                                    for b in bias:
                                        #subtracting from policy overcomplicates things; only boost helpful info
                                        if b[2]-rew_exp<0:
                                            continue
                                        #social policy proportional to how much better than experience social option is
                                        socChoices[b[0]] += b[1]*(b[2]-rew_exp)
                                else:
                                    for b in bias:
                                        socChoices[b[0]] += b[1]
                                socpolicy = socChoices/sum(socChoices)
                                #mixture policy
                                policy[:,ag] = ((1 - pars[ag]['gamma'])*policy[:,ag]) + (pars[ag]['gamma'] * socpolicy)
                    #Choice
                    policy = policy/np.sum(policy,axis=0) 
                    X[t,:] = [np.random.choice(gridSize, p = policy[:,ag]) for ag in range(nAgents)]
                    Y[t,:] = [(envList[ag][envs[r]][X[t,ag]]['payoff']-prior_mean)/prior_scale +np.random.normal(0,.01) for ag in range(nAgents)]
            for ag in range(nAgents):
                #collect information
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
    #format dataset
    data = np.column_stack((np.concatenate(agent),np.concatenate(group),np.concatenate(r0und),np.concatenate(env),np.concatenate(trial),
            np.concatenate(choice),np.concatenate(reward),np.concatenate(lambda_ind),
            np.concatenate(beta),np.concatenate(tau),np.concatenate(gamma),np.concatenate(alpha),np.concatenate(eps_soc),np.concatenate(dummy)))
    agentData = pd.DataFrame(data,columns=('agent','group','round','env','trial','choice','reward','lambda','beta',
                                           'tau','gamma','alpha','eps_soc',"dummy"))      
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
    par_names = ["lambda","beta","tau","gamma","alpha","eps_soc","dummy"] #needed for dicts later #
    all_pars = []
    #homogenous groups
    if hom:
        #randomly select model unless given as an argument
        if models is None:
            models = np.random.randint(0,4,nGroups)
        else:
            assert models in range(0,5), "Model has to be between 0 and 4"
            models = np.ones(nGroups)*models
        for g in range(nGroups):
            model = models[g]
            all_pars.append([]) #set up list for parameter dictionary
            par = np.zeros((nAgents,len(par_names)))
            par[:,0] = np.random.lognormal(-0.75,0.5,(nAgents)) #lambda
            par[:,1] = np.random.lognormal(-0.75,0.5,(nAgents))
            par[:,2] = np.random.lognormal(-4.5,0.9,(nAgents))  #tau
            if model==1:
                par[:,3] = np.random.uniform(0,1,(nAgents)) #gamma previously 1/14
            elif model==2:
                par[:,4] = np.random.uniform(0,1,(nAgents))
            elif model==3:
                par[:,5] = np.random.exponential(2,(nAgents)) #eps_soc
            elif model == 4:
                par[:,6] = np.ones(nAgents) #dummy variable for a model test
            for ag in range(nAgents):
                pars = dict(zip(par_names,par[ag,:]))
                all_pars[g].append(pars)
    else: #heterogeneous groups
        for g in range(nGroups):
            all_pars.append([])
            model = np.random.randint(0,4,(nAgents))
            par = np.zeros((nAgents,len(par_names)))
            par[:,0] = np.random.lognormal(-0.75,0.5,(nAgents)) #lambda
            par[:,1] = np.random.lognormal(-0.75,0.5,(nAgents)) #beta
            par[:,2] = np.random.lognormal(-4.5,0.9,(nAgents))  #tau
            for ag in range(nAgents):
                if model[ag]==1:
                    par[ag,3] = np.random.uniform(0.2142857,1,(1))        #gamma
                elif model[ag]==2:
                    par[ag,4] = np.random.uniform(0.116,1,(1))
                elif model[ag]==3:
                    while par[ag,5]==0 or par[ag,5]>19:
                      par[ag,5] = np.random.exponential(2,(1)) #eps_soc
                elif model[ag]==4:
                    par[ag,6] = 1 #dummy variable for a model test
                pars = dict(zip(par_names,par[ag,:]))
                all_pars[g].append(pars)
    return(all_pars)

def param_gen_pilot(nAgents,nGroups,hom=True,models=None):
    """
    Generates parameter sets for simulations based on parameter distribution from pilot

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
    if hom:
        #randomly select model unless given as an argument
        if models is None:
            models = np.random.randint(0,4,nGroups)
        else:
            assert models in range(0,5), "Model has to be between 0 and 4"
            models = np.ones(nGroups)*models
        for g in range(nGroups):
            model = models[g]
            all_pars.append([]) #set up list for parameter dictionary
            par = np.zeros((nAgents,len(par_names)))
            par[:,0] = np.random.lognormal(0.1,0.1,(nAgents)) #lambda
            par[:,1] = np.random.lognormal(-1.25,0.2,(nAgents))
            #par[:,1] = np.zeros(nAgents)*0.1#beta
            par[:,2] = np.random.lognormal(-4.5,0.1,(nAgents))  #tau
            if model==1:
                par[:,3] = np.random.lognormal(-3.5,1,(nAgents)) #gamma previously 1/14
            elif model==2:
                par[:,4] = np.random.lognormal(-3.5,1,(nAgents))
            elif model==3:
                par[:,5] = np.random.lognormal(3.6,0.3,(nAgents)) #eps_soc #formerly 3,0.5 for dist_grids
            elif model == 4:
                par[:,6] = np.ones(nAgents) #dummy variable for a model test
            for ag in range(nAgents):
                pars = dict(zip(par_names,par[ag,:]))
                all_pars[g].append(pars)
    else:
        for g in range(nGroups):
            all_pars.append([])
            model = np.random.randint(0,4,(nAgents))
            par = np.zeros((nAgents,len(par_names)))
            par[:,0] = np.random.lognormal(0.1,0.1,(nAgents)) #lambda
            #par[:,0] = np.ones((nAgents))*2
            par[:,1] = np.random.lognormal(-1.25,0.2,(nAgents)) #beta
            par[:,2] = np.random.lognormal(-4.5,0.1,(nAgents))  #tau
            for ag in range(nAgents):
                if model[ag]==1:
                    while par[ag,3]<0.143 or par[ag,3]>1:
                        #par[ag,3] = np.random.lognormal(-3.5,1,(1))     #gamma
                        par[ag,3] = np.random.uniform(0,1,(1))
                elif model[ag]==2:
                    while par[ag,4]<0.116:
                    #par[ag,4] = np.random.exponential(6.5,(1))     #alpha minmax 0.2; VS+ has 6.5 mean
                        #par[ag,4] = np.random.lognormal(-3.5,1,(1))
                        par[ag,4] = np.random.uniform(0,1,(1))
                elif model[ag]==3:
                    while par[ag,5]==0 or par[ag,5]>19:
                      par[ag,5] = np.random.lognormal(3.6,0.3,(1)) #eps_soc
                elif model[ag]==4:
                    par[ag,6] = 1 #dummy variable for a model test
                pars = dict(zip(par_names,par[ag,:]))
                all_pars[g].append(pars)
    return(all_pars)


#for evolutionary simulations
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
        pop.extend(param_gen_pilot(1,1,models = np.random.choice(models)))
    return(pop)


#for evolutionary simulations
def pop_gen_pilot(popSize, models):
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
            pop.extend(param_gen_pilot(1,subpop,models=i)) 
    while len(pop)<popSize: #if even proportion can't result in correct population size, populate with random model from list of desired
        pop.extend(param_gen_pilot(1,1,models = np.random.choice(models)))
    return(pop)
    
#simulate roger's paradox
def roger(nAgents,nGroups,socModel,nSoc):
    par_names = ["lambda","beta","tau","gamma","alpha","eps_soc","dummy"] #needed for dicts later
    all_pars = []
    model = np.zeros(nAgents)
    model[:nSoc] = socModel
    for g in range(nGroups):
        all_pars.append([])
        par = np.zeros((nAgents,len(par_names)))
        for ag in range(nAgents):
            if model[ag]==0:
                par[ag,0] = np.random.lognormal(0.55,0.2) #lambda
                par[ag,1] = np.random.lognormal(-1.45,0.2) #beta
                par[ag,2] = np.random.lognormal(-5.5,0.9)  #tau
            elif model[ag]==1:
                par[ag,0] = np.random.lognormal(0.55,0.2) #lambda
                par[ag,1] = np.random.lognormal(-1.6,0.2) #beta
                par[ag,2] = np.random.lognormal(-5.5,0.9) #tau
                par[ag,3] = np.random.lognormal(-2.25,0.4)    #gamma
            elif model[ag]==2:
                par[ag,0] = np.random.lognormal(0.4,0.15) #lambda
                par[ag,1] = np.random.lognormal(-1.9,0.4) #beta
                par[ag,2] = np.random.lognormal(-5.3,0.9) #tau
                par[ag,4] = np.random.lognormal(-2.2,0.4) #alpha
            else:
                par[ag,0] = np.random.lognormal(0.7,0.2) #lambda
                par[ag,1] = np.random.lognormal(-1.7,0.2) #beta
                par[ag,2] = np.random.lognormal(-5.5,0.9)  #tau
                par[ag,5] = np.random.lognormal(1.09,0.35) #eps_soc
            pars = dict(zip(par_names,par[ag,:]))
            all_pars[g].append(pars)
    return(all_pars)