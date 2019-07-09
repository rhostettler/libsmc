# Nonlinear Gaussian state-space Model
## Usage
* `model = MODEL_NONLINEAR_GAUSSIAN(f, Q, g, R, m0, P0, fast)`
 
## Description
Defines the model structure for nonlinear state-space models with
Gaussian process- and measurement noise of the form
 
    x[0] ~ N(m0, P0),
    x[n] = f(x[n-1], n) + q[n],
    y[n] = g(x[n], n) + r[n],
 
where q[n] ~ N(0, Q[n]) and r[n] ~ N(0, R[n]).
 
## Input
* `f`: Mean function of the dynamic model. Function handle of the form 
  @(x, theta).
* `Q`: Process noise covariance, either a dx-times-dx matrix or a
  function handle of the form @(x, theta).
* `g`: Mean function of the observation model. Function handle of the 
  form @(x, theta).
* `R`: Measurement noise covariance, either a dy-times-dy matrix or a
  function handle of the form @(x, theta).
* `fast`: Boolean flag set to 'true' if the transition density and
  likelihood (i.e., `f`, `Q`, `g`, and `R`) can evaluate whole dx-times-J
  particle matrices at once (default: false).
 
## Output
* `model`: The state-space model struct that contains the necessary 
  fields (i.e., the probabilistic representation of the state-space model
  px0, px, py).
 
## Authors
2018-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
