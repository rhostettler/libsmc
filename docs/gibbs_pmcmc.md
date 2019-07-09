# Particle Gibbs Markov chain Monte Carlo sampler
## Usage
* `x = gibbs_pmcmc(model, y)`
* `[x, theta, sys] = gibbs_pmcmc(model, y, theta0, lambda, K, par)`
 
## Description
Particle Markov chain Monte Carlo Gibbs sampler for generating samples
from the posterior distribution p(theta, x[0:N] | y[1:N]). The sampler
uses the Gibbs approach, which samples each parameter condtionally on
the others in turn. In particular, the method first samples from
 
    x[0:N] ~ p(x[0:N] | theta, y[1:N])
 
and then from
 
    theta ~ p(theta | x[0:N], y[1:N]).
 
Sampling is achieved by calling two user-defined functions (supplied as
fields in the 'par' argument):
 
1. `sample_states()`, and
2. `sample_parameters()`.
 
If the former is not specified, cpfas is used by default and if the
latter is not specified, no parameters are sampled.
  
## Input
* `model`: Function handle of the form @(theta) to construct the state-
  space model struct. **N.B.**: This is likely to change in the future as
  parameter handling should not be directed to a "constructor".
* `y`: dy-times-N measurement matrix.
* `theta0`: Initial guess of the model parameters (if any; default: 
  `NaN`).
* `lambda`: Set of static (known) parameters (default: `[]`).
* `K`: No. of MCMC samples to generate (default: `10`).
* `par`: Additional parameters:
    - `Kburnin`: No. of burn-in samples (removed after sampling; default:
      `0`).
    - `Kmixing`: No. of samples for improving the mixing (removed after
      sampling; default: `1`).
    - `x = sample_states(model, y, x, lambda)`: Function to sample the 
      states (default: `@cpfas`).
    - `[theta, state] = sample_parameters(y, t, x, theta, model, state)`:
      Function to sample the parameters (default: `[]`). In addition to 
      the newly sampled parameters, the function may also return a state 
      variable which stores the sampler's state (useful for adaptive 
      sampling).
    - `show_progress(p, x, theta)`: Function to display or otherwise 
      illustrate the progress (default: `[]`). The parameters are the
      progress of the sampling in [0,1], and the so-far sampled 
      trajectories and parameters.

## Output
* `x`: dx-times-N-times-K array of trajectory samples.
* `theta`: dtheta-times-K matrix of parameter samples.
* `sys`: Cell array of particle systems. *Warning: This should only be
  used if absolutely necessary since storing the particle systems takes a
  lot of memory and may cause Matlab to crash.*
 
## References
1. C. Andrieu, A. Doucet, and R. Holenstein, "Particle Markov chain
   Monte Carlo methods," Journal of the Royal Statistical Society: 
   Series B (Statistical Methodology), vol. 72, no. 3, pp. 269-342, 2010
 
2. F. Lindsten, M. I. Jordan, and T. B. Schon, "Particle Gibbs with
   ancestor sampling," Journal of Machine Learning Research, vol. 15, 
   pp. 2145-2184, 2014.

## Authors
2017-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
