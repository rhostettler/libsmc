# Conditional particle filter with ancestor sampling
## Usage
* `x = cpfas(model, y)`
* `[x, sys] = cpfas(model, y, xtilde, theta, J, par)`

## Description
Conditional particle filter with ancestor sampling (CPF-AS), a
fast-mixing, Markov kernel invariant over state trajectories x[0:N]. See
[1,2] for details.
 
## Input
* `model`: State-space model struct.
* `y`: dy-times-N measurement matrix y[1:N].
* `xtilde`: dx-times-N seed trajectory xtilde[0:N] (optional, a bootstrap
  particle filter is used to generate a seed trajectory if omitted).
* `theta`: Additional parameters (optional).
* `J`: Number of particles (default: 100).
* `par`: Additional algorithm parameters:
    - `xp = sample(model, y, x, theta)`: Function to sample new particles
      (used for the J-1 particles; default: `@sample_bootstrap`).
    - `lw = calculate_incremental_weights(model, y, xp, x, theta)`: 
      Function to calculate the incremental particle weights (must match 
      the sampling function defined above; default: 
      `@calculate_incremental_weights_bootstrap`).
    - `[alpha, state] = sample_ancestor_index(model, y, xtilde, x, lw, theta)`:
      Function to sample the ancestor indices (default:
      `@sample_ancestor_index`).
 
## Output
* `x`: The newly sampled trajectory (dx-times-N).
* `sys`: Struct of the particle system containing:
    - `x`: Particles of the marginal filtering density (not complete
      trajectories).
    - `w`: Particle weights of the marginal filtering density
      corresponding to x.
    - `alpha`: Ancestor indices.
    - `r`: Resampling indicator (always true for CPF-AS).
    - `state`: Internal state of the ancestor index sampling algorithm,
      see the corresponding algorithm for details.
 
## References
1. C. Andrieu, A. Doucet, and R. Holenstein, "Particle Markov chain
   Monte Carlo methods," Journal of the Royal Statistical Society: 
   Series B (Statistical Methodology), vol. 72, no. 3, pp. 269-342, 2010
 
2. F. Lindsten, M. I. Jordan, and T. B. Schon, "Particle Gibbs with
   ancestor sampling," Journal of Machine Learning Research, vol. 15, 
   pp. 2145-2184, 2014.

## Authors
2017-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
