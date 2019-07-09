# Calculate ancestor weigths for CPF-AS and Markovian state-space models
## Usage
* `lvtilde = calculate_ancestor_weights(model, y, xtilde, x, lw, theta)`
 
## Description
Calculates the non-normalized ancestor weights for the conditional
particle filter with ancestor sampling and Markovian state-space models.

## Input
* `model`: State-space model struct.
* `y`: dy-times-1 measurement vector.
* `xtilde`: dx-times-1 sample of the seed trajectory, xtilde[n].
* `x`: dx-times-J matrix of particles x[n-1]^j.
* `lw`: 1-times-J row vector of particle log-weights log(w[n-1]^j).
* `theta`: Additional parameters.
 
## Output
* `lvtilde`: Non-normalized ancestor weights.
 
## Authors
2017-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
