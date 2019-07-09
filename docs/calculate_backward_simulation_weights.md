# Calculate backward simulation weights
## Usage
* `lv = calculate_backward_simulation_weights(model, xs, x, theta)`
 
## Description
Calculates the backward simulation weights needed in the Monte Carlo
approximation of the smoothing kernel in forward-filtering
backward-simulation smoothing [1].
 
## Input
* `model`: State-space model structure.
* `xs`: dx-times-1 smoothed particle at n+1, that is, x[n+1|N].
* `x`: dx-times-Jf matrix of filtered particles at n, that is, x[n|n].
* `theta`: Additional parameters.

## Output
* `lv`: Backward simulation weights.

## References
1. W. Fong, S. J. Godsill, A. Doucet, and M. West, "Monte Carlo 
   smoothing with application to audio signal enhancement," IEEE 
   Transactions on Signal Processing, vol. 50, pp. 438? 449, February 
   2002.
 
## Author
2017-present -- Roland Hostettler
