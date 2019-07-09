# Sequential importance sampling with resampling particle filter
## Usage
* `xhat = sisr_pf(model, y)`
* `[xhat, sys] = sisr_pf(model, y, theta, J, par)`
 
## Description
`pf` is a generic sequential importance sampling with resampling particle
filter (PF). It can be used as anything ranging from the bootstrap PF to
the auxiliary particle filter.
 
In its minimal form, a bootstrap particle filter with conditional
resampling based on the effective sample size with `J = 100` particles is
used.
 
## Input
* `model`: State-space model struct.
* `y`: dy-times-N matrix of measurements.
* `theta`: dtheta-times-1 vector or dtheta-times-N matrix of additional
  parameters (default: `[]`).
* `J`: Number of particles (default: 100).
* `par`: Struct of additional (optional) parameters:
    - `[alpha, lw, r] = resample(lw)`: Function handle to the resampling 
      function. The input `lw` is the log-weights and the function must 
      return the indices of the resampled particles (`alpha`), the log-
      weights of the resampled (`lw`) particles, as well as a boolean
      indicating whether resampling was performed or not (`r`). Default:
      `@resample_ess`.
    - `xp = sample(model, y, x, theta)`: Function handle to the sampling
      function to draw new state vectors. Default: `@sample_bootstrap`.
    - `lv = calculate_incremental_weights(model, y, xp, x, theta)`:
      Function to calculate the weight increment `lv`. This function must
      match the `sample` function. Default:
      `@calculate_incremental_weights_bootstrap`.
 
## Output
* `xhat`: Minimum mean squared error filter state estimate (calculated 
  using the marginal filtering density).
* `sys`: Particle system array of structs with the following fields:
    - `x`: dx-times-J matrix of particles for the marginal filtering 
      density.
    - `w`: 1-times-J vector of the particle weights for the marginal
      filtering density.
    - `alpha`: 1-times-J vector of ancestor indices.
    - `r`: Boolean resampling indicator.
 
## Authors
2018-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
