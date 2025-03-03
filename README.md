# mining-silver
Code for our group project at 2024 Workshop on Simulation-based Inference. 
We show empirically than when it is possible to access latent variables in a model it is more efficient to infer latent variable posteriors and use them to infer parameter posterior, rather than directly consider parameters.

For an observation $x$, given the posterior distribution over latent variables $p(z|x)$, and the posterior distribution over parameters given the latent variables $p(\theta|z, x)$, we can compute the posterior over latent variables as follows : 
\begin{equation}
p(\theta|x) = \int p(\theta|z, x)p(z|x) dz
\end{equation}

On the two-moons toy distribution, we use Sequential Posterior Neural Estimation (SNPE) to first infer the latent variable posterior distribution (moon centers), then we infer the posterior distribution over parameters, outperforming the direct SNPE method. 

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
