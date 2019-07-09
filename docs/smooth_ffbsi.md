# Forward-filtering backward-simulation particle smoothing
## Usage
* `xhat = smooth_ffbsi(model, y, theta, Js, sys)`
* `[xhat, sys] = smooth_ffbsi(model, y, theta, Js, sys, par)`
 
## Description
Forward-filtering backward-simulation (FFBSi) particle smoother as
described in [1].
 
## Input
* `model`: State-space model struct.
* `y`: dy-times-N matrix of measurements.
* `theta`: Additional parameters.
* `Js`: No. of smoothing particles.
* `sys`: Particle system array of structs.
* `par`: Algorithm parameter struct, may contain the following fields:
    - `[beta, state] = sample_backward_simulation(model, xs, x, lw, theta)`:
      Function to sample from the backward smoothing kernel (default:
      `@sample_backward_simulation`).
 
## Output
* `xhat`: dx-times-N matrix of smoothed state estimates (MMSE).
* `sys`: Particle system array of structs for the smoothed particle
  system. The following fields are added by `smooth_ffbsi`:
    - `xs`: Smoothed particles.
    - `ws`: Smoothed particle weights (`1/Js` for FFBSi).
    - `state`: State of the backward simulation sampler.

## References
1. W. Fong, S. J. Godsill, A. Doucet, and M. West, "Monte Carlo 
   smoothing with application to audio signal enhancement," IEEE 
   Transactions on Signal Processing, vol. 50, pp. 438? 449, February 
   2002.
 
## Authors
2017-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
