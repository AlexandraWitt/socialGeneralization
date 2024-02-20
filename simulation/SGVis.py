# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:08:32 2022

@author: Alex
"""
import json
import os

import GPy
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from simulation import modelSim as ms

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

#What does noisy generalization actually do
distsize=30
ind_size=4
soc_size=6
#1. generate data to fit to
k = GPy.kern.RBF(input_dim=1,lengthscale=4,variance=.0001) #CW: a lengthscale of 4 for an input width of 30 is similar to lambda= 1.5 for a width of 11
#X = np.linspace(0.,1.,500) # 500 points evenly spaced over [0,1]
X = np.arange(distsize)
X = X[:,None] # reshape X to make it n*D
mu = np.zeros((distsize)) # vector of the means
C = k.K(X,X) # covariance matrix
# Generate 20 sample path with mean mu and covariance C
np.random.seed(12345)
Z = np.random.multivariate_normal(mu,C,20)
for i in range(20):
    plt.plot(X[:],Z[i,:])
    
#Select the i=0th sampled function
m_0 = Z[0,:]
ind_x = [4,11, 18, 22, 28]
obs = X[ind_x] 
rew = m_0[ind_x] + np.random.normal(0,.0001,len(ind_x))


plt.plot(X[:],m_0)
plt.scatter(obs,rew)
plt.show()

#2. Models with different amounts of noise

pars = ms.param_gen(4, 1,models=0)
pars = pars[0][0]
#pars['lambda']=0.9
pars['lambda']=4 #Lengthscale of 4 is equivalent for this larger input size
pars['beta']=0.5
pars['tau']=0.01 #

n_00001 = ms.GPR(X,ind_x,rew,np.ones_like(obs),pars,0.0001)
n_0001 = ms.GPR(X,ind_x,rew,np.ones_like(obs),pars,0.001)
n_001 = ms.GPR(X,ind_x,rew,np.ones_like(obs),pars,0.01)
n_01 = ms.GPR(X,ind_x,rew,np.ones_like(obs),pars,0.1)
n_1 = ms.GPR(X,ind_x,rew,np.ones_like(obs),pars,1)


#effect of social noise
#plt.style.use('dark_background') #default
cmap = plt.get_cmap('viridis')
cols = cmap(np.linspace(0, 0.8, 5))
plt.scatter(obs,rew)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.plot(X,n_00001[0],label="ɛ=0.0001",c=cols[0])
plt.plot(X,n_0001[0],label="ɛ=0.001",c=cols[0])
plt.plot(X,n_001[0],label="ɛ=0.01",c=cols[1])
plt.plot(X,n_01[0],label="ɛ=0.1",c=cols[2])
plt.plot(X,n_1[0],label="ɛ=1",c=cols[3])
plt.legend(prop={'size':13})
plt.xlabel("Input")
plt.ylabel("Reward")
#plt.savefig("./plots/noise_var_example_disc.png",bbox_inches="tight",dpi=300)


#prediction by model (main text figure)

pars = ms.param_gen(4, 1,models=0)
pars = pars[0][0]
pars['lambda']=4
pars['beta']=0.5
pars['tau']=0.01
pars['eps_soc']=0.5

distsize=30
ind_size=4
soc_size=6
#1. generate data to fit to
k = GPy.kern.RBF(input_dim=1,lengthscale=4,variance=1)
#X = np.linspace(0.,1.,500) # 500 points evenly spaced over [0,1]
X = np.arange(distsize)
X = X[:,None] # reshape X to make it n*D
mu = np.zeros((distsize)) # vector of the means
C = k.K(X,X) # covariance matrix
# Generate 20 sample path with mean mu and covariance C
np.random.seed(12345)
m_0 = np.random.multivariate_normal(mu,C)
m_soc = np.random.multivariate_normal(m_0,C)
np.corrcoef(m_0,m_soc)
plt.plot(X[:],m_0)
plt.plot(X[:],m_soc)

#ind_x = np.random.randint(0,distsize,ind_size)
ind_x= [4,11, 18,  28]
#soc_x = np.random.randint(0,distsize,soc_size)
soc_x = [2,7, 16, 22,  24, 29]
socIndex = np.concatenate((np.zeros_like(ind_x),np.ones_like(soc_x)))

# np.random.seed(2023)
# m_0=m_soc=np.random.normal(0,1,distsize)

obs = X[np.concatenate((ind_x,soc_x))]
rew = np.concatenate((m_0[ind_x],m_soc[soc_x]))+np.random.normal(0,0.0001,len(socIndex))

predAS = ms.GPR(X,ind_x,rew[:ind_size],socIndex[:ind_size],pars,0.0001)
predDB = ms.GPR(X,ind_x,rew[:ind_size],socIndex[:ind_size],pars,0.0001)
predVS = ms.GPR(X,ind_x,rew[:ind_size],socIndex[:ind_size],pars,0.0001)
predSG = ms.GPR(X,np.concatenate((ind_x,soc_x)),rew,socIndex,pars,0.0001)

valAS = ms.ucb(predAS)
valDB = ms.ucb(predDB)
valVS = ms.ucb(predVS)
valSG = ms.ucb(predSG)

soc_rew = rew[ind_size:]
valVS[soc_x] = valVS[soc_x] + 0.75*(soc_rew-valVS[soc_x])

#from matplotlib.pyplot import figure
#figure(figsize=(10,5),dpi=300)
policyAS = valAS-np.max(valAS)
policyDB = valDB-np.max(valDB)
policyVS = valVS-np.max(valVS)
policySG = valSG-np.max(valSG)

policyAS = np.exp(policyAS/0.2)
policyDB = np.exp(policyDB/0.2)
policyVS = np.exp(policyVS/0.2)
policySG = np.exp(policySG/0.2)

socpolicy = np.zeros_like(policyDB)
socpolicy[soc_x]+=1
socpolicy = socpolicy/np.sum(socpolicy)

policyDB=0.5*policyDB+0.5*socpolicy

policyAS = policyAS/np.sum(policyAS)
policyDB = policyDB/np.sum(policyDB)
policyVS = policyVS/np.sum(policyVS)
policySG = policySG/np.sum(policySG)

maxAS = np.argmax(policyAS)
maxDB = np.argmax(policyDB)
maxVS = np.argmax(policyVS)
maxSG = np.argmax(policySG)


#Main text plot begins

plt.rcParams['font.size'] = '10'
plt.rcParams['figure.figsize'] = 2.5, 3.75
fig,(pred,val,pol) = plt.subplots(3,constrained_layout=True)
#fig.figure(figsize=(10,5),dpi=300)
pred.plot(X[:],m_0,linestyle="dashed",color="red",alpha=0.75, zorder=1)
pred.fill_between(np.squeeze(X),predAS[0]-predAS[1],predAS[0]+predAS[1],color="#999999",alpha=0.5, zorder=2)
#pred.fill_between(np.squeeze(X),predDB[0]-predDB[1],predDB[0]+predDB[1],color="none",edgecolor="#E69F00",hatch="/",  zorder=3)
#pred.fill_between(np.squeeze(X),predVS[0]-predVS[1],predVS[0]+predAS[1],color="none",edgecolor="#009E73",hatch="\\",  zorder=4)
pred.fill_between(np.squeeze(X),predSG[0]-predSG[1],predSG[0]+predSG[1],color="#56B4E9",alpha=0.6, zorder = 5)
pred.plot(X,predAS[0],color="#999999",label="AS", zorder=6)
#pred.plot(X,predDB[0],color="#E69F00",label="DB", zorder=7)
#pred.plot(X,predAS[0],color="#009E73",label="VS", zorder=8)
pred.plot(X,predSG[0],color="#56B4E9",label="SG", zorder=9)
pred.scatter(obs[ind_size:],rew[ind_size:],color="#0072B2",label="Social", zorder=10)
pred.scatter(obs[:ind_size],rew[:ind_size],color="#999999",label="Private", zorder=11)             
pred.spines['top'].set_visible(False)
pred.spines['right'].set_visible(False)
pred.set(ylabel="Reward")
#pred.set_aspect("equal")

#plt.savefig("./plots/all_models_pred_CogSci.png",bbox_inches="tight",dpi=300)
dist = val.plot(X[:],m_0,linestyle="dashed",color="red",label="True", zorder=1)
val.plot(X,valDB,color="#E69F00",zorder=2) 
val.plot(X,valVS,color="#009E73", zorder=3)
val.plot(X,valAS,color="#999999", zorder=4)
val.plot(X,valSG,color="#56B4E9", zorder=5)
sc2 = val.scatter(obs[ind_size:],rew[ind_size:],color="#0072B2",label="Social", zorder=6)
sc1 = val.scatter(obs[:ind_size],rew[:ind_size],color="#999999",label="Private", zorder=7)
val.spines['top'].set_visible(False)
val.spines['right'].set_visible(False)
val.set(ylabel="Value")
#val.set_aspect("equal")
#plt.savefig("./plots/all_models_vals_CogSci_noleg.png",bbox_inches="tight",dpi=300)


#scatter = plt.scatter(obs,rew,c=socIndex,cmap=test)
#pol.plot(X[:],m_0,linestyle="dashed",color="red",alpha=0.75, zorder=1)
pol.plot(X,policyAS,color="#999999",label="AS" , zorder = 1)#,alpha=0.75)
pol.plot(X,policyDB,color="#E69F00",label="DB", zorder = 2)#,alpha=0.75)
pol.plot(X,policyVS,color="#009E73",label="VS", zorder = 3)#,alpha=0.75)
pol.plot(X,policySG,color="#56B4E9",label="SG", zorder = 4)#,alpha=0.75)
pol.scatter(maxAS,policyAS[maxAS],c="#999999",marker="x",label="AS", zorder=5 )
pol.scatter(maxDB,policyDB[maxDB],c="#E69F00",label="DB",marker = 'x',zorder=6)
pol.scatter(maxVS,policyVS[maxVS],c="#009E73",label="VS",marker = 'x', zorder=7)
pol.scatter(maxSG,policySG[maxSG],c="#56B4E9",label="SG",marker = 'x', zorder=8)
pol.spines['top'].set_visible(False)
pol.spines['right'].set_visible(False)
#plt.legend(loc=(0.45,0.75),handles=scatter.legend_elements()[0], labels=["Private","Social"],prop={'size':18})
pol.set(xlabel="Choice",ylabel="Policy")
#pol.set_yscale("log")
handles, labels = plt.gca().get_legend_handles_labels()
order = [3,0,1,2]
leg1 = plt.legend(loc=(1.1,1.5),handles=[handles[idx] for idx in order],labels=[labels[idx] for idx in order],prop={'size':8},title="Model",frameon=False)
plt.gca().add_artist(leg1)
leg2 = plt.legend(loc=(1.05,3),handles=[sc1,sc2], labels=["Private","Social"],prop={'size':8},title="Info Source",frameon=False)
plt.gca().add_artist(leg2)
leg3 = plt.legend(loc=(1.05,0.75),handles=dist, labels=["True \nDistribution"],prop={'size':8},frameon=False)
plt.gca().add_artist(leg3)
#pol.set_aspect("equal")
#fig.tight_layout()
#plt.savefig("./plots/model_fig_CogSci_disc.pdf",bbox_inches="tight")


#########
#VS == SG for alpha 1 and no gen

distsize=30
ind_size=4
soc_size=6
#1. generate data to fit to
    
#Select the i=0th sampled function
ind_x= [4,11, 18,  28]
#soc_x = np.random.randint(0,distsize,soc_size)
soc_x = [2,7, 16, 22,  24, 29]
socIndex = np.concatenate((np.ones_like(soc_x),np.zeros_like(ind_x)))

pars = ms.param_gen(4, 1,models=0)
pars = pars[0][0]

pars['lambda']=0.01
pars['beta']=0.5
pars['tau']=0.01
pars['eps_soc']=0

np.random.seed(2023)
m_0=m_soc=np.random.normal(0,1,distsize)

obs = X[np.concatenate((ind_x,soc_x))]
rew = np.concatenate((m_0[ind_x],m_soc[soc_x]))+np.random.normal(0,0.0001,len(socIndex))

predVS = ms.GPR(X,ind_x,rew[:ind_size:],socIndex[:ind_size],pars,0.0001)
predSG = ms.GPR(X,np.concatenate((ind_x,soc_x)),rew,socIndex,pars,0.0001)

valVS = ms.ucb(predVS)
valSG = ms.ucb(predSG)

soc_rew = rew[ind_size:]
valVS[soc_x] = valVS[soc_x] + (soc_rew-valVS[soc_x])

#from matplotlib.pyplot import figure
#figure(figsize=(10,5),dpi=300)
policyVS = valVS-np.max(valVS)
policySG = valSG-np.max(valSG)

policyVS = np.exp(policyVS/0.2)
policySG = np.exp(policySG/0.2)

policyVS = policyVS/np.sum(policyVS)
policySG = policySG/np.sum(policySG)

maxVS = np.argmax(policyVS)
maxSG = np.argmax(policySG)


plt.rcParams['font.size'] = '10'
plt.rcParams['figure.figsize'] = 2.5, 3.75
fig,(pred,val,pol) = plt.subplots(3,constrained_layout=True)
#fig.figure(figsize=(10,5),dpi=300)
pred.plot(X[:],m_0,linestyle="dashed",color="red",alpha=0.75, zorder=1)
pred.fill_between(np.squeeze(X),predVS[0]-predVS[1],predVS[0]+predVS[1],color="#009E73",alpha=0.5, zorder=2)
pred.fill_between(np.squeeze(X),predSG[0]-predSG[1],predSG[0]+predSG[1],color="#56B4E9",alpha=0.6, zorder = 5)
pred.plot(X,predVS[0],color="#009E73",label="VS", zorder=6)
pred.plot(X,predSG[0],color="#56B4E9",label="SG", zorder=9)
pred.scatter(obs[ind_size:],rew[ind_size:],color="#0072B2",label="Social", zorder=10)
pred.scatter(obs[:ind_size],rew[:ind_size],color="#999999",label="Private", zorder=11)             
pred.spines['top'].set_visible(False)
pred.spines['right'].set_visible(False)
pred.set(ylabel="Reward")
#pred.set_aspect("equal")

#plt.savefig("./plots/all_models_pred_CogSci.png",bbox_inches="tight",dpi=300)
dist = val.plot(X[:],m_0,linestyle="dashed",color="red",label="True", zorder=1)
val.plot(X,valVS,color="#009E73", zorder=3)
val.plot(X,valSG,color="#56B4E9", zorder=5)
sc2 = val.scatter(obs[ind_size:],rew[ind_size:],color="#0072B2",label="Social", zorder=6)
sc1 = val.scatter(obs[:ind_size],rew[:ind_size],color="#999999",label="Private", zorder=7)
val.spines['top'].set_visible(False)
val.spines['right'].set_visible(False)
val.set(ylabel="Value")
#val.set_aspect("equal")
#plt.savefig("./plots/all_models_vals_CogSci_noleg.png",bbox_inches="tight",dpi=300)


#scatter = plt.scatter(obs,rew,c=socIndex,cmap=test)

pol.plot(X,policyVS,color="#009E73",label="VS", zorder = 3)#,alpha=0.75)
pol.plot(X,policySG,color="#56B4E9",label="SG", zorder = 4)#,alpha=0.75)
pol.scatter(maxVS,policyVS[maxVS],c="#009E73",label="VS",marker = 'x', zorder=7)
pol.scatter(maxSG,policySG[maxSG],c="#56B4E9",label="SG",marker = 'x', zorder=8)
pol.spines['top'].set_visible(False)
pol.spines['right'].set_visible(False)
#plt.legend(loc=(0.45,0.75),handles=scatter.legend_elements()[0], labels=["Private","Social"],prop={'size':18})
pol.set(xlabel="Choice",ylabel="Policy")
#pol.set_yscale("log")
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1]
leg1 = plt.legend(loc=(1.1,1.5),handles=[handles[idx] for idx in order],labels=[labels[idx] for idx in order],prop={'size':8},title="Model",frameon=False)
plt.gca().add_artist(leg1)
leg2 = plt.legend(loc=(1.05,3),handles=[sc1,sc2], labels=["Private","Social"],prop={'size':8},title="Info Source",frameon=False)
plt.gca().add_artist(leg2)
leg3 = plt.legend(loc=(1.05,0.75),handles=dist, labels=["True \nDistribution"],prop={'size':8},frameon=False)
plt.gca().add_artist(leg3)
#pol.set_aspect("equal")
#fig.tight_layout()
#plt.savefig("./plots/model_fig_VSSG.pdf",bbox_inches="tight")