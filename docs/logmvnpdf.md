# Logarithm of multivariate normal pdf
## Usage
* `px = logmvnpdf(x)`
* `px = logmvnpdf(x, m, P)`
 
# Description
Returns the logarithm of N(x; m, P) (or N(x; 0, I) if m and P are 
omitted), that is, the log-likelihood. Everything is calculated in
log-domain such that numerical precision is retained to a high degree.
 
The arguments x and m are automatically expanded to match each other.
 
**Note**: For consistency with Matlab's `mvnpdf`, each *row* in `x` (and
consequently `m`) is assumed to be a vector.
 
## Input
* `x`: N-times-dx vector of values to evaluate.
* `m`: N-times-dx vector of means (default: 0).
* `P`: dx-times-dx covariance matrix (default: I).
 
## Output
* `px`: The logarithm of the pdf value.
 
## Authors
2016-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
