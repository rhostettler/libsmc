# Time-averaged root mean squared error
## Usage
* `mrmse = trmse(e)`
* `[mrmse, varrmse, ermse] = trmse(e)`
 
## Description
Calculates the mean (and variance) of the time-averaged root mean squared
error (RMSE) for a set of Monte Carlo simulations.

## Input
* `e`: dx-times-N-times-L 3D matrix of errors (dx: state dimension; N:
  number of time samples; L: number of Monte Carlo simulations).
 
## Output
* `mrmse`: Mean of the time-averaged RMSE.
* `varrmse`: Variance of the time-averaged RMSE.
* `ermse`: The time-averaged RMSEs.
 
## Authors
2019-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
