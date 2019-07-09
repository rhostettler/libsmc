# Linear Gaussian state-space model
## Usage
* `model = MODEL_LGSSM(F, Q, G, R, m0, P0)`
 
## Description
Creates a model structure with the appropriate probabilistic description
for linear, Gaussian state-space models. Given the model of the form
 
    x[n] = F*x[n-1] + q
    y[n] = G*x[n] + r
 
where q[n] ~ N(0, Q), r[n] ~ N(0, R), and p(x[0]) = N(m0, P0) (F, G, Q, 
and R may all depend on t[n] or any other parameter(s)), the function 
initializes the corresponding transition density and likelihood given by
 
    p(x[n] | x[n-1]) = N(x[n]; F*x[n-1], Q), and
    p(y[n] | x[n]) = N(y[n]; G*x[n], R),
 
respectively.
 
## Input
* `F`, `Q`, `G`, `R`, `m0`, `P0`: Model parameters as described above. If
  any of the parameters is time-varying or depends on any other
  parameters, it must be a function handle of the form @(~, theta).
 
## Output
* `model`: Model struct containing px0, px, and py as described above.
 
## Authors
2017-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
