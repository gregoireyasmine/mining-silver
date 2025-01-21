# mining-silver
Code for our group project at 2024 Workshop on Simulation-based Inference. 
We show empirically than when it is possible to access latent variables in a model it is more efficient to infer latent variable posteriors and use them to infer parameter posterior, rather than directly consider parameters.

# setup :
adding the path to the project might be necessary  
export PYTHONPATH=/path/to/mining-silver:$PYTHONPATH

## References: 
##### SNPE : 
 Papamakarios, George and Murray Iain
 Fast epsilon-free Inference of Simulation Models with Bayesian Conditional Density Estimation
 NeurIPS 2015

##### Neural Spline Flows : 
  Durkan, Conor and Artur, Bekasov and Murray, Iain and Papamakarios, George
  Neural Spline Flows
  NeurIPS 2016

##### SBI module: 
  https://sbi-dev.github.io/sbi/
