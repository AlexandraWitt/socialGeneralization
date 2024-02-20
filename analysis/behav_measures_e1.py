# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 15:11:24 2022
calculates additional behavioural measures because for loops in R suck
could this be a function that goes behind the simulations? Potentially
@author: Alex
"""
import numpy as np
import pandas as pd
#import 
data = pd.read_csv("./Data/e1_data.csv")
#task settings
nAgent = 4
shor=15
nround=8
gridSize=121
Xstar = np.array([(x, y) for x in range(np.sqrt(gridSize).astype(int)) for y in range(np.sqrt(gridSize).astype(int))])

data.loc[data["reward"]<0.05,"reward"]=0.05 #participants never saw rewards lower than .05
data["search_dist"] = np.nan
data["uncovered_ind"] = np.nan
data["prev_rew"] = np.nan                      
data["max_rew"] = np.nan
data["coherence"] = np.nan

for g in np.unique(data["group"]):
    for r in np.unique(data["round"]):
        for t in range(1,len(np.unique(data["trial"]))):
            for ag in np.unique(data.loc[(data['group']==g)]['agent']).astype(int):
                # #search distance                
                data.loc[(data["agent"]==ag) & (data["group"]==g) & (data["round"]==r) & (data["trial"]==t),"search_dist"] = np.linalg.norm(
                              Xstar[data.loc[(data["agent"]==ag) & (data["group"]==g) & (data["round"]==r) & (data["trial"]==t),"choice"].astype(int)] -
                              Xstar[data.loc[(data["agent"]==ag) & (data["group"]==g) & (data["round"]==r) & (data["trial"]==t-1),"choice"].astype(int)])
                #previous reward
                data.loc[(data["agent"]==ag) & (data["group"]==g) & (data["round"]==r) & (data["trial"]==t),"prev_rew"] = data.loc[(data["agent"]==ag) &
                                                                                                                                    (data["group"]==g) & 
                                                                                                                                    (data["round"]==r) & 
                                                                                                                                    (data["trial"]==t-1),"reward"].values
                #uncoverage
                data.loc[(data["agent"]==ag) & (data["group"]==g) & 
                          (data["round"]==r) & (data["trial"]==t),"uncoverage"] = np.sum(len(np.unique(data.loc[(data["agent"]==ag) & (data["group"]==g) & 
                                                                                                            (data["round"]==r) & (data["trial"]<t),"choice"])))
                #max reward
                data.loc[(data["agent"]==ag) & (data["group"]==g) & 
                          (data["round"]==r) & (data["trial"]==t),"max_rew"] = np.max(data.loc[(data["agent"]==ag) & (data["group"]==g) & 
                                                                                                            (data["round"]==r) & (data["trial"]<t),"reward"])
                #coherence (distance from group average position)
                data.loc[(data["agent"]==ag) & (data["group"]==g) & (data["round"]==r) & (data["trial"]==t),"coherence"] = np.linalg.norm(
                              Xstar[data.loc[(data["agent"]==ag) & (data["group"]==g) & (data["round"]==r) & (data["trial"]==t),"choice"].astype(int)] -
                              np.mean(Xstar[data.loc[(data["agent"]!=ag) & (data["group"]==g) & (data["round"]==r) & (data["trial"]==t-1),"choice"].astype(int)]))
                

data["expl_eff"] = data["reward"]-data["prev_rew"]
                             
data["variance"] = np.nan
data["uncoverage_grp"] = np.nan
data["max_rew_grp"] = np.nan
data["prev_rew_grp"] = np.nan
for g in np.unique(data["group"]):
    for r in range(len(np.unique(data["round"]))):
        for t in (np.unique(data["trial"])):
            #variance of choices
            data.loc[(data["group"]==g) & (data["round"]==r) & (data["trial"]==t),"variance"] = np.var(
                Xstar[data.loc[(data["group"]==g) & (data["round"]==r) & (data["trial"]==t),"choice"].astype(int)])
            #uncoverage grp (how many unique tiles were uncovered across participants up until this trial)
            data.loc[(data["group"]==g) & (data["round"]==r) & (data["trial"]==t),"uncoverage_grp"] = np.sum(
                len(np.unique(data.loc[(data["group"]==g) & (data["round"]==r) & (data["trial"]<t),"choice"])))
            #average max rews per group
            data.loc[(data["group"]==g) & (data["round"]==r) & (data["trial"]==t),"max_rew_grp"] = np.mean(
                np.unique(data.loc[(data["group"]==g) & (data["round"]==r) & (data["trial"]==t-1),"max_rew"]))
            #average prev rews per group
            data.loc[(data["group"]==g) & (data["round"]==r) & (data["trial"]==t),"prev_rew_grp"] = np.mean(
                np.unique(data.loc[(data["group"]==g) & (data["round"]==r) & (data["trial"]==t-1),"reward"]))
            
data["max_rew_prop"] = data["max_rew"]/data["max_rew_grp"]
data["prev_rew_prop"] = data["prev_rew"]/data["prev_rew_grp"]
data["uncoverage_prop"] = data["uncoverage"]/data["uncoverage_grp"]

                                                                            
data["soc_sd1"] = np.nan
data["soc_sd2"] = np.nan
data["soc_sd3"] = np.nan
data["soc_sd"] = np.nan   

data["soc_rew1"] = np.nan
data["soc_rew2"] = np.nan
data["soc_rew3"] = np.nan  

data["soc_choice1"] = np.nan
data["soc_choice2"] = np.nan
data["soc_choice3"] = np.nan                                                                          
                                                                            
for g in np.unique(data["group"]):
    for r in range(len(np.unique(data["round"]))):
        for t in range(1,len(np.unique(data["trial"]))):
            for ag in np.unique(data.loc[(data['group']==g)]['agent']).astype(int):
            #social search distance
                data.loc[(data["agent"]==ag) & (data["group"]==g) &
                         (data["round"]==r) & (data["trial"]==t),"soc_sd1"] = np.linalg.norm(
                             Xstar[data.loc[(data["agent"]==ag) & (data["group"]==g) &
                         (data["round"]==r) & (data["trial"]==t),"choice"].astype(int)]-
                             Xstar[data.loc[(data["agent"]!=ag) & (data["group"]==g) &
                         (data["round"]==r) & (data["trial"]==t-1),"choice"].astype(int)][0])
                #previous social reward
                data.loc[(data["agent"]==ag) & (data["group"]==g) &
                          (data["round"]==r) & (data["trial"]==t),"soc_rew1"] = np.array(data.loc[(data["agent"]!=ag) & (data["group"]==g) &
                (data["round"]==r) & (data["trial"]==t-1),"reward"])[0]
                #social choice (option index)
                data.loc[(data["agent"]==ag) & (data["group"]==g) &
                          (data["round"]==r) & (data["trial"]==t),"soc_choice1"] = np.array(data.loc[(data["agent"]!=ag) & (data["group"]==g) &
                (data["round"]==r) & (data["trial"]==t-1),"choice"])[0]                                                                                          
                                                                                                  
                try:     #depending on the remaining number of players, these measures might not be available        
                    #social search distance
                    data.loc[(data["agent"]==ag) & (data["group"]==g) &
                             (data["round"]==r) & (data["trial"]==t),"soc_sd2"] = np.linalg.norm(
                                 Xstar[data.loc[(data["agent"]==ag) & (data["group"]==g) &
                             (data["round"]==r) & (data["trial"]==t),"choice"].astype(int)],
                                 Xstar[data.loc[(data["agent"]!=ag) & (data["group"]==g) &
                             (data["round"]==r) & (data["trial"]==t-1),"choice"].astype(int)][1])
                    #previous social reward
                    data.loc[(data["agent"]==ag) & (data["group"]==g) &
                              (data["round"]==r) & (data["trial"]==t),"soc_rew2"] = np.array(data.loc[(data["agent"]!=ag) & (data["group"]==g) &
                    (data["round"]==r) & (data["trial"]==t-1),"reward"])[1]
                    #previous social choice
                    data.loc[(data["agent"]==ag) & (data["group"]==g) &
                              (data["round"]==r) & (data["trial"]==t),"soc_choice2"] = np.array(data.loc[(data["agent"]!=ag) & (data["group"]==g) &
                    (data["round"]==r) & (data["trial"]==t-1),"choice"])[1]                                                                                          
                    #social search distance          
                    data.loc[(data["agent"]==ag) & (data["group"]==g) &
                             (data["round"]==r) & (data["trial"]==t),"soc_sd3"] = np.linalg.norm(
                                 Xstar[data.loc[(data["agent"]==ag) & (data["group"]==g) &
                             (data["round"]==r) & (data["trial"]==t),"choice"].astype(int)],
                                 Xstar[data.loc[(data["agent"]!=ag) & (data["group"]==g) &
                             (data["round"]==r) & (data["trial"]==t-1),"choice"].astype(int)][2])
                    #previous social reward
                    data.loc[(data["agent"]==ag) & (data["group"]==g) &
                              (data["round"]==r) & (data["trial"]==t),"soc_rew3"] = np.array(data.loc[(data["agent"]!=ag) & (data["group"]==g) &
                    (data["round"]==r) & (data["trial"]==t-1),"reward"])[2]
                    #previous social choice
                    data.loc[(data["agent"]==ag) & (data["group"]==g) &
                              (data["round"]==r) & (data["trial"]==t),"soc_choice3"] = np.array(data.loc[(data["agent"]!=ag) & (data["group"]==g) &
                    (data["round"]==r) & (data["trial"]==t-1),"choice"])[2]                                                                                  
                except IndexError:
                    continue
                             
                             
data["soc_sd"]=data[["soc_sd1","soc_sd2","soc_sd3"]].mean(axis=1)
data["soc_rew"]=data[["soc_rew1","soc_rew2","soc_rew3"]].mean(axis=1)

                             
data.to_csv("./Data/e1_data.csv",index=False)



#################################################################################
# adjust for search distance regression, which I don't want to do any other analyses on
#################################################################################

data = data.drop(columns=["soc_sd","soc_rew"])
data = data.rename(columns={'prev_rew':'soc_rew0','search_dist':'soc_sd0',"choice":"soc_choice0"})
data["prev_rew"] = data["soc_rew0"]
data = pd.wide_to_long(data,["soc_sd","soc_rew","soc_choice",],["agent","group","round","trial"],"ref_agent")

data = data.reset_index()
data["social"] = np.select([(data["ref_agent"]==0),(data["ref_agent"]!=0)],[False,True])

data.to_csv("./Data/e1_data_regressable.csv",index=False)