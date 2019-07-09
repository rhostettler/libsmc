# Incremental particle weights for the bootstrap particle filter
## Usage
* `lv = calculate_incremental_weights_bootstrap(model, y, xp, ~, theta)`
 
## Description
Calculates the incremental particle weights for the bootstrap particle
filter. In this case, the incremental weight is given by
 
    v[n] ~= p(y[n] | x[n]).
 
Note that the function actually computes the non-normalized log weights
for numerical stability.
 
## Input
* `model`: State-space model struct.
* `y`: Measurement y[n].
* `xp`: Particles at time t[n] (i.e. x[n]).
* `x`: Particles at time t[n-1] (i.e. x[n-1]).
* `theta`: Model parameters.
 
## Output
* `lv`: The non-normalized log-weights.
 
## Author
2017-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
