# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 20:01:41 2022

@author: Alex
The functionto get a fit's negative log likelihood
"""
import numpy as np

try:
    import modelSim as ms
except ModuleNotFoundError:
    from simulation import modelSim as ms

#input only for one group
def model_fit(pars,m,targetdata,socdata,shor,gridSize=121,baseEpsilon=.0001,payoff=True):
    """
    Takes parameters, a candidate model, data and task specs
    and returns a their negative log likelihood.

    Parameters
    ----------
    pars : list of numeric
        A list of parameters required for the model in the order 
        lambda, beta, tau and, optionally, additional free parameters needed for model m.
    m : int range(4)
        The model the fit of which is to be determined.
        0=AS,1=DB,2=VS,3=SG
    targetdata : pandas dataframe
        Data of the agent whose model fit is to be determined.
    socdata : pandas dataframe
        Data of the other agents in the target's group.
    shor : int
        Search horizon of the input dataset.
    gridSize : int, optional
        Length of the flattened environment. The default is 121.

    Returns
    -------
    Input models negative log likelihood of having generated the input data.

    """
    
    rounds = np.unique(targetdata[:,0]) #get number of rounds from data
    Xstar = np.array([(x, y) for x in range(np.sqrt(gridSize).astype(int)) for y in range(np.sqrt(gridSize).astype(int))]) #generate MAB grid
    pars = np.exp(pars) #exponentiate input pars (can no longer be 0)
    arr = np.zeros(7) #set up parameters
    arr[:len(pars)] = pars #and add the provided parameters, all unused are 0
    #create dict for the specified model
    if m == 0: 
        par_names = ["lambda","beta","tau","gamma","alpha","eps_soc","dummy"]
        pars = dict(zip(par_names,arr))
    elif m ==1:
        par_names = ["lambda","beta","tau","gamma","alpha","eps_soc","dummy"]
        pars = dict(zip(par_names,arr))
    elif m == 2:
        par_names = ["lambda","beta","tau","alpha","gamma","eps_soc","dummy"]
        pars = dict(zip(par_names,arr))
    elif m == 3:
        par_names = ["lambda","beta","tau","eps_soc","alpha","gamma","dummy"]
        pars = dict(zip(par_names,arr))

    nLL = np.zeros(len(rounds)) #set up overall negative log likelihood
    for r in range(len(rounds)): #iterate over rounds
        nll=0 #negative log likelihood of the round
        #get data for the current round; predictions start with the first chosen option
        indD = targetdata[np.nonzero(targetdata[:,0]==rounds[r]),:][0]# TODO: try to reduce pandas ops
        socD = socdata[np.nonzero(socdata[:,0]==rounds[r]),:][0]
        observed = indD[:,2]
        chosen = observed[1:] #we don't want to predict the first random choice
        reward = indD[:,3]
        random = indD[:,4]
        #just making extra sure our trials are base 0, otherwise the code breaks
        if np.min(indD[:,1])!=0:
            indD[:,1] = indD[:,1]-1
            socD[:,1] = socD[:,1]-1
        #iterate over search horizon
        for i in range(shor-1):
            t=i+1
            if random[t]==1:
                continue
            else:
                obs = observed[:t] #self observations
                rewards = reward[:t]
                socIndex = np.zeros_like(obs)
                if pars['eps_soc']>0 or pars["dummy"]!=0:#if social generalization, add social obs
                    obs = np.append(obs, socD[np.nonzero(socD[:,1]<t),2]) 
                    rewards = np.append(rewards, socD[np.nonzero(socD[:,1]<t),3])
                    if pars['eps_soc']>0:
                        socIndex = np.append(socIndex, np.ones_like(socD[np.nonzero(socD[:,1]<t),2]))
                    else:
                        socIndex = np.zeros_like(obs)
                
                prediction = ms.GPR(Xstar,obs.astype(int),rewards,socIndex,pars,baseEpsilon)
                
                vals = np.squeeze(ms.ucb(prediction,pars['beta']))
                if pars['alpha']>0:
                    vs_boost=[(i.astype(int),np.sum(socD[np.nonzero(socD[:,1]<t),2]==i), 
                              np.mean(socD[np.nonzero(socD[:,1]<t),3][socD[np.nonzero(socD[:,1]<t),2]==i])) for i in np.unique(socD[np.nonzero(socD[:,1]<t),2][0])]
                    for b in vs_boost:
                        vals[b[0]] = vals[b[0]] + pars['alpha']*(b[1]*b[2]-vals[b[0]])

                #count occurrences of social choices in the previous round
                if pars['gamma']>0:
                    bias = [(i,np.sum(socD[np.nonzero(socD[:,1]==t-1),2]==i), 
                             np.mean(socD[np.nonzero(socD[:,1]==t-1),3][socD[np.nonzero(socD[:,1]==t-1),2]==i])) for i in socD[np.nonzero(socD[:,1]==t-1),2][0]] #TODO: edit to match VS if we keep it as :t
                
                vals = vals-np.max(vals) #avoid overflow
                #Policy
                policy = np.exp(vals/pars['tau'])
            
                #Decision biasing
                if pars['gamma']>0:
                    socChoices = np.ones(gridSize)*.000000000001  
                    for b in bias:
                        socChoices[b[0].astype(int)] += b[1]
                    socpolicy = socChoices/sum(socChoices)
                    #mixture policy
                    policy = ((1 - pars['gamma'])*policy) + (pars['gamma'] * socpolicy)
                policy[policy/np.sum(policy)==0]=0.001*np.sum(policy) #quick fix, consider adjusting gamma priors instead
                policy[policy<0]=0.001*np.sum(policy)
                
                policy = policy/np.sum(policy,axis=0)
    
                nll = nll-np.log(policy[chosen[i].astype(int)]) #negative log likelihood for this trial
        nLL[r]=nll #append negative log likelihood for this round
    return(np.sum(nLL))
