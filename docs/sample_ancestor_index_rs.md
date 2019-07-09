# Rejection-sampling-based ancestor index sampling for CPF-AS
## Usage
* `[alpha, state] = SAMPLE_ANCESTOR_INDEX_RS(model, y, xt, x, lw, theta, L)`

## Description
Samples the ancestor index for the seed trajectory in the conditional
particle filter with ancestor sampling (CPF-AS) using rejection sampling
(for Markovian state-space models).

## Input
* `model`: State-space model struct.
* `y`: dy-times-1 measurement vector.
* `xtilde`: dx-times-1 sample of the seed trajectory, xtilde[n].
* `x`: dx-times-J matrix of particles x[n-1]^j.
* `lw`: 1-times-J row vector of particle log-weights log(w[n-1]^j).
* `theta`: Additional parameters.
* `L`: Maximum number of rejection sampling trials before falling back
  to sampling from the categorical distribution (default: `10`).
 
## Output
* `alpha`: Sampled ancestor index.
* `state`: Internal state of the sampler. Struct that contains the
  following fields:
    - `l`: Number of rejection sampling trials performed.
    - `accepted`: `true` if the ancestor index was sampled using
      rejection sampling, `false` otherwise.
    - `dgamma`: Difference in true acceptance probability and the lower
      bound used in rejection sampling.
 
## Authors
2017-2019 -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
