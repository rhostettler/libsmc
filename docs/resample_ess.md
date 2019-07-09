# Effective sample size-based conditional resampling
## Usage
* `[alpha, lw, r] = resample_ess(lw)`
* `[alpha, lw, r] = resample_ess(lw, par)`
 
## Description
Conditional resampling function using an estimate of the effecitve sample
size (ESS) given by
 
    J_ess = 1./sum(w.^2)
 
as the resampling criterion. By default, the resampling threshold is set
to M/3 and systematic resampling is used. The resampled ancestor indices
are returned in the 'alpha'-variable, together with the updated (or 
unaltered if no resampling was done) log-weights.
 
## Input
* `lw`: Normalized log-weights.
* `par`: Struct of optional parameters. Possible parameters are:
    - `Jt`: Resampling threshold (default: J/3).
    - `resample`: Resampling function handle (default: 
      `@resample_stratified`).
 
## Output
* `alpha`: Resampled indices.
* `lw`: Log-weights.
* `r`: Indicator whether resampling was performed or not.
 
## Authors
2017-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
