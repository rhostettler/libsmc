# Multivariate normal pdf structure
## Usage
* `py = pdf_mvn(dx, m)`
* `py = pdf_mvn(dx, m, P, fast)`
 
## Description
Initializes the pdf struct for a multivariate normal distribution with
mean m (mean function m(x, theta)) and covariance P (covariance function
P(x, theta).
 
## Input
* `dx`: Dimension of the state.
* `m`: Mean vector (dx-times-1). May be static or a function handle of
  the form @(x, theta).
* `P`: Covariance matrix (dx-times-dx). May be static or a function
  handle of the form @(x, theta). Default: `eye(dx)`.
* `fast`: `true` if `m(x, theta)` and `P(x, theta)` can evaluated for a
  complete dx-times-J particle matrix at once. Default: `false`.
 
## Output
* `py`: pdf struct with fields:
  - `rand(x, theta)`: Random sample generator.
  - `logpdf(y, x, theta)`: Log-pdf.
  - `fast`: Flag for particle matrix evaluation.
  - `kappa(x, theta)`: Bounding constant of the pdf such that `p(y) <= 
     kappa` for all `y`.
 
## Authors
  2019-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
