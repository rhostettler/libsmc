# Particle smoother
## Usage
* `xhat, = ps(model, y)`
* `[xhat, sys] = ps(model, y, theta, Jf, Js, par, sys)`
 
## Description
Forward filtering backward simulation particle smoother as described in
[1].
 
## Input
* `model`: State-space model struct.
* `y`: dy-times-N matrix of measurements.
* `theta`: Additional parameters.
* `Jf`: Number of particles to be used in the forward filter (if no `sys`
  is provided, see below; default: 250).
* `Js`: Number of particles for the smoother (default: 100).
* `par`: Struct of additional parameters. The following parameters are
  supported:
    - `[xhat, sys] = par.smooth(model, y, theta, Js, sys)`: The actual
      smoothing function used for the backward recursion (default:
      `@smooth_ffbsi`).
* `sys`: Particle system as obtained from a forward filter. If no system
  is provided, a bootstrap particle filter is run to generate it. `sys`
  must contain the following fields:
    - `x`: Matrix of particles for the marginal filtering density.
    - `w`: Vector of particle weights for the marginal filtering density.
 
## Output
* `xhat`: dx-times-N matrix of smoothed state estimates (MMSE).
* `sys`: Particle system array of structs for the smoothed particle
  system. At least the following fields are added (additional fields may
  be added by the specific backward recursions):
    - `xs`: Smoothed particles.
    - `ws`: Smoothed particle weights.
 
## References
1. W. Fong, S. J. Godsill, A. Doucet, and M. West, "Monte Carlo 
   smoothing with application to audio signal enhancement," IEEE 
   Transactions on Signal Processing, vol. 50, pp. 438? 449, February 
   2002.
 
## Authors
2017-present -- Roland Hostettler
