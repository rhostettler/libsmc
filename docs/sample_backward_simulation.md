# Backward trajectory simulation sampling
## Usage
* `[beta, state] = sample_backward_simulation(model, xs, x, lw, theta)`
 
## Description
Samples the indices of the smoothed particle trajectories for
backward-simulation particle smoothing. Calculates all the kernel weights
prior to sampling, which corresponds to the original (slow) approach in
[1].
 
## Input
* `model`: State-space model struct.
* `xs`: dx-times-Js matrix of smoothed particles at n+1, that is, 
   x[n+1|N].
* `x`: dx-times-Jf matrix of filtered particles at n, that is, x[n|n].
* `lw`: dx-times-Jf matrix of log-weights of the filtered particles at n,
   that is, lw[n|n].
* `theta`: Additional parameters.
 
## Output
* `beta`: Sampled indices.
* `state`: Sampler state (empty).
 
## References
1. W. Fong, S. J. Godsill, A. Doucet, and M. West, "Monte Carlo 
   smoothing with application to audio signal enhancement," IEEE 
   Transactions on Signal Processing, vol. 50, pp. 438? 449, February 
   2002.
 
## Authors
2017-present -- Roland Hostettler
