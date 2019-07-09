# General incremental weights for sequential importance sampling
## Usage
* `lv = calculate_incremental_weights_generic(model, y, xp, x, theta, q)`
 
## Description
Calculates the incremental importance weight for sequential importance
sampling for an arbitrary proposal density q.
 
## Input
* `model`: State-space model struct.
* `y`: dy-times-1 measurement vector y[n].
* `xp`: dx-times-J matrix of newly drawn particles for the state x[n].
* `x`: dx-times-J matrix of previous state particles x[n-1].
* `theta`: Additional parameters.
* `q`: Importance density struct.
 
## Output
* `lv`: Logarithm ov incremental weights.
 
## See also
sample_generic
 
## Authors
2017-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
