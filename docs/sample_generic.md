# Sample from an arbitrary importance density
## Usage
* `xp = sample_generic(model, y, x, theta, q)`
 
## Description
Samples a set of new particles x[n] from the importance distribution 
q(x[n]).
 
## Input
* `model`: State-space model struct.
* `y`: dy-times-1 measurement vector y[n].
* `x`: dx-times-J particle matrix at x[n-1].
* `theta`: Additional parameter vector.
* `q`: Importance density such that x[n] ~ q(x[n] | x[n-1], y[n]).
 
## Output
* `xp`: The new samples x[n].
 
## See also
calculate_incremental_weights_generic
 
## Authors
2017-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
