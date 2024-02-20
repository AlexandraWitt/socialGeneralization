# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 18:28:19 2022

@author: Alex
"""
import json
import os
import sys

import numpy as np

import modelSim_Najar as ms

path = ('./environments' ) 
json_files = [file for file in os.listdir(path) if file.endswith('_lambda0_soc06.json')]
envList = []
for file in json_files:
    f=open(os.path.join(path, file))
    envList.append(json.load(f))
#identical environments across agents
envList = [envList[0],envList[0],envList[0],envList[0]]
#Agent parameters
popSize = iterations = 100 #population size is equal to the number of iterations in each generation to ensure equal numbers in each generation
generations  = 500
groupSize = 4

parNames = np.array(["lambda","beta","tau","gamma","alpha","eps_soc"])
#Mutation Rates
paramMutation = .02
paramMutationVariance = .2 
#model could be randomly selected to stay the same in 1/4 of the cases, so type Mutation is 0.002
typeMutation = 0.002*(4/3)

#get initial population
mix = int(sys.argv[1])
outputList = [] #Store the outputs
scoreboard  =[]

if mix in range(10):
    name = "AS"
    outputList.append(ms.pop_gen(popSize,[0]))
if mix in range(10,20):
    name="DB"
    outputList.append(ms.pop_gen(popSize,[1]))
elif mix in range(20,30):
    name="VS"
    outputList.append(ms.pop_gen(popSize,[2]))
elif mix in range(30,40):
    name = "SG"
    outputList.append(ms.pop_gen(popSize,[3]))
elif mix in range(40,50):
    name = "AS.DB"
    outputList.append(ms.pop_gen(popSize,[0,1]))
elif mix in range(50,60):
    name = "AS.VS"
    outputList.append(ms.pop_gen(popSize,[0,2]))
elif mix in range(60,70):
    name = "AS.SG"
    outputList.append(ms.pop_gen(popSize,[0,3]))
elif mix in range(70,80):
    name = "DB.VS"
    outputList.append(ms.pop_gen(popSize,[1,2]))
elif mix in range(80,90):
    name = "DB.SG"
    outputList.append(ms.pop_gen(popSize,[1,3]))
elif mix in range(90,100):
    name = "VS.SG"
    outputList.append(ms.pop_gen(popSize,[2,3]))
elif mix in range(100,110):
    name = "AS.DB.VS"
    outputList.append(ms.pop_gen(popSize,[0,1,2]))
elif mix in range(110,120):
    name = "AS.DB.SG"
    outputList.append(ms.pop_gen(popSize,[0,1,3]))
elif mix in range(120,130):
    name = "AS.VS.SG"
    outputList.append(ms.pop_gen(popSize,[0,2,3]))
elif mix in range(130,140):
    name = "DB.VS.SG"
    outputList.append(ms.pop_gen(popSize,[1,2,3]))
elif mix in range(140,150):
    name = "AS.DB.VS.SG"
    outputList.append(ms.pop_gen(popSize,[0,1,2,3]))

    
#iterate over generations
for gen in range(1,generations):
    print(gen)
    #data collectors
    pop = outputList[gen-1]
    scores = np.zeros(popSize)
    agentAssignment = np.zeros((iterations,groupSize)).astype(int)
    probs = np.ones(popSize)*(1/popSize)
    #for every member of the next generation, draw a sample of four
    for i in range(iterations):
        ids = np.random.choice(range(popSize),groupSize,replace=False,p=probs)
        #reduce selection probability for previously chosen agents
        probs[ids] = probs[ids]/groupSize
        probs = probs/sum(probs)
        agentAssignment[i,:] = ids    
    
    #for every member, determine who does best in one round of the task
    for i in range(iterations):
        for ag in range(len(agentAssignment[i,:])):
            agent = pop[agentAssignment[i,ag]] #pop has double nested lists I can't work around otherwise
            result = ms.model_sim([agent],envList,1,15,memory=False, payoff=True)
            scores[agentAssignment[i,ag]] = scores[agentAssignment[i,ag]] + np.array(result.groupby("agent").mean()['reward'])
    
    #scores get summed, so if they win multiple times, we average
    for ag in range(popSize):
        scores[ag] = scores[ag] / sum(sum(agentAssignment==ag))
    scoreboard.append(scores)
    winners = [agentAssignment[i,np.argmax(scores[agentAssignment[i,:]])] for i in range(popSize)]
    newPop = []
    #populate next generation with the winners
    for i in range(popSize):
        newPop.append([dict(pop[winners[i]][0])])
    #Mutations
    mutateType = np.random.uniform(size=popSize)<typeMutation
    mutateParams = np.random.uniform(size=(popSize,6))<paramMutation
    #params
    for i in range(popSize): #dictionary has come to bite me in the ass - go through pop, check if any params should be mutated, get their names and mutate them if they were part of the model before (otherwise we change type)
        if True in mutateParams[i,:]:
            mut = parNames[mutateParams[i,:]]
            for par in mut:
                if newPop[i][0][par] !=0:
                    newPop[i][0][par] = newPop[i][0][par] + np.random.normal(0,np.sqrt(paramMutationVariance)) 
    #keep pars within bounds after mutation
    for i in range(popSize):
        if newPop[i][0]['lambda']<0:
            newPop[i][0]['lambda'] = np.random.lognormal(-0.75,0.5)
        if newPop[i][0]['beta']<0:
            newPop[i][0]['beta'] = np.random.lognormal(-0.75,0.5)
        if newPop[i][0]['tau'] <0:
            newPop[i][0]['tau'] = np.random.lognormal(-4.5,0.9)
        if newPop[i][0]['gamma']!=0 and newPop[i][0]['gamma']<1/14 or newPop[i][0]['gamma'] > 1:
            newPop[i][0]['gamma'] = np.random.uniform(0.5,1)
        if newPop[i][0]['alpha']!=0 and newPop[i][0]['alpha']<1/14 or newPop[i][0]['alpha'] > 1:
            newPop[i][0]['alpha'] = np.random.uniform(0.5,1)
        if newPop[i][0]['eps_soc']<0:
            newPop[i][0]['eps_soc'] = np.random.exponential(2)

    for i in range(popSize):
        if mutateType[i]:
            #this keeps the evolved parameters stable for invasions in the .6 envs
            newPop[i][0]["gamma"]=0
            newPop[i][0]["alpha"]=0
            newPop[i][0]["eps_soc"]=0
            # #this resets 
            # newPop[i][0]['lambda'] = np.random.lognormal(-0.75,0.5)
            # newPop[i][0]['beta'] = np.random.lognormal(-0.75,0.5)
            # newPop[i][0]['tau'] = np.random.lognormal(-4.5,0.9)
            model = np.random.randint(0,4)
            if model == 1:
                newPop[i][0]["gamma"] = np.random.uniform(1/14,1)
            elif model==2:
                newPop[i][0]["alpha"]=np.random.uniform(1/14,1)
            elif model==3:
                newPop[i][0]['eps_soc'] = np.random.exponential(2)
    outputList.append(newPop)
    
    
path = "./Data/evoSims/evoSim_Najar/"
if not os.path.exists(path):
    os.makedirs(path)
filename = path+name+"_Najar_"+str(mix) 
np.save(filename,outputList)

filename = path+name+"_Najar_"+str(mix)+"_scores"
np.save(filename,scoreboard)