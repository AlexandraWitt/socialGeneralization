# socialGeneralization

This repository contains all data and code necessary to reproduce the results of **Flexible integration of social information despite interindividual differences in reward** (Witt, Toyokawa, Lala, Gaissmaier & Wu, 2024), in which we investigated social learning in correlated reward environments.

## ./simulation

has all code used for simulating

- `environmentGenerator.R` generates correlated environments

- `modelSim.py` has simulation, parameter generation and necessary helper (GP,UCB) functions
	- `modelSim_allVariants.py` has the same, but includes legacy models not used for fitting analysis
	- `modelSim_Najar.py` has the same for a version of the task that has one agent and one expert making optimal choices
	- `simmedModels.py` uses `modelSim.py` to generate datasets for analysis

- `evoSim*` generates evolutionary simulations
	- `evoSimSynthesis.py` converts this script's output for further analysis

- `SGVis.py` generates explanatory plots for the models (e.g. Fig. 2)

and everything related to model fitting and recovery

- `modelFit.py` has the function to calculate negative log likelihoods of parameter sets given data

- `modelRecovery.py` is the model recovery script
	- `mrecov_parameteradd.py` converts this script's output for further analysis

- `modelFitting*` script for model fitting
	- `fitting_synthesis*` converts this script's output for further analysis

## ./analysis
analysis scripts

- `behav_measures*` generates additional behavioural measures for data (search distance, previous reward etc.); already run on the data

- `evoSimAnalysis.py` converts

- `analysis_evo_e1.R` has analysis for evolutionary simulations and Exp. 1

- `analysis_e2.R` has analysis for Exp. 2

- `recovery_analyses.R` has model and parameter recovery, as well as bounding logic

- `suppAnalysis.R` has the supplementary analyses

## ./environments

home to the environments used in the experiment

## ./Data
data from experiments, fitting data, evolutionary simulations

## ./plots
all the plots used in the paper + SI