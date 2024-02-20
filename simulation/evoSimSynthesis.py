# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:14:55 2022
Merges the separate run's evosim outputs into one nice big dataframe

@author: Alex
"""
#combine evoSim output into one dataset

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb


#Get data

path = "./Data/evoSims/evoSim"
files = [file for file in os.listdir(path) if 
         file.endswith(('0.npy','1.npy','2.npy','3.npy','4.npy','5.npy','6.npy','7.npy','8.npy','9.npy'))]
#get populations
sims = np.array([np.load(path + "/" + file,allow_pickle=True) for file in files])
#get scores and concatenate
score_files = [file for file in os.listdir(path) if 
         file.endswith("scores.npy")]
scores = np.array([np.load(path + "/" + file,allow_pickle=True) for file in score_files])

evo = []

par_names = ["lambda","beta","tau","gamma","alpha","eps_soc","dummy"]
#iterate over mixes and generations, add scores
for mix in range(len(sims)):
    for gen in range(len(sims[0])):
        for ag in range(len(sims[0][0])):
            if gen==499:
                score = np.nan
            else:
                score = scores[mix][gen][ag]
            id_data = ["_".join(files[mix].split("_")[:-2]),gen,ag,score] #-2 for only 1 _ in the file name, 3 for 2 etc.
            pars = [sims[mix][gen][ag][0][key] for key in par_names]
            for i,par in enumerate(pars):
                if type(par) == np.ndarray:
                    pars[i]=pars[i][0]
            id_data.extend(pars)
            evo.append(id_data)
            
            

evo_data = pd.DataFrame(evo,columns=["mix","gen","agent","score","lambda","beta","tau","gamma","alpha","eps_soc","dummy"])#


evo_data['model'] = np.select([ (evo_data['alpha']==0) & (evo_data['gamma']==0) & (evo_data['eps_soc']==0) & (evo_data['dummy']==0), #
                        (evo_data['gamma']!=0),
                        (evo_data['alpha']!=0),
                        (evo_data['eps_soc']!=0)],
                        ['AS','DB','VS','SG'])#
