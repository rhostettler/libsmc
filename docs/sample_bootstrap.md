# Sample from the bootstrap importance density
## Usage
* `xp = sample_bootstrap(model, y, x, theta)`
 
## Description
Samples a set of new samples x[n] from the bootstrap importance density,
that is, samples
 
    x[n] ~ p(x[n] | x[n-1]).
 
## Input
* `model`: State-space model struct.
* `y`: Measurement vector y[n].
* `x`: Samples at x[n-1].
* `theta`: Model parameters.
 
## Output
* `xp`: The new samples x[n].
 
## Author
2018-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
