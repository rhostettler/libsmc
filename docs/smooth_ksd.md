# Kronander-Schon-Dahlin marginal particle smoother
## Usage
* `xhat = smooth_ksd(model, y, theta, Js, sys)`
* `[xhat, sys] = smooth_ksd(model, y, theta, Js, sys)`

## Description
Backward sampling particle smoother targeting the marginal smoothing
density according to [1].

Note that it is well known that this smoother is biased, see [1].
 
## Input
* `model`: State-space model struct.
* `y`: dy-times-N matrix of measurements.
* `theta`: Additional parameters.
* `Js`: No. of smoothing particles.
* `sys`: Particle system array of structs.
 
## Output
* `xhat`: dx-times-N matrix of smoothed state estimates (MMSE).
* `sys`: Particle system array of structs for the smoothed particle
  system. The following fields are added by `smooth_ffbsi`:
    - `xs`: Smoothed particles.
    - `ws`: Smoothed particle weights.
 
## References
1. J. Kronander, T. B. Schon, and J. Dahlin, "Backward sequential Monte 
   Carlo for marginal smoothing," in IEEE Workshop on Statistical Signal 
   Processing (SSP), June 2014, pp. 368-371.
 
## Authors
2017-present -- Roland Hostettler
