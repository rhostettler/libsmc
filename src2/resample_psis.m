function [alpha, lw, r] = resample_psis(lw, par)
% PSIS-based resampling
%
% SYNOPSIS
%   [alpha, lw, r] = RESAMPLE_PSIS(lw, par)
%
% DESCRIPTION
%   Conditional resampling based on fitting a pareto distribution to the
%   tail of the importance weights (~ pareto smoothed importance sampling).
%   Resampling is performed if the k parameter of the fitted pareto
%   distribution falls below the threshold kt, meaning that all the moments
%   larger than 1/k are undefined for the weights' distribution. In
%   practice, if kt = 0.5, then once the estimated k is smaller than 0.5,
%   the weights' variance does not exist.
%
%   This method requires (and uses) the PSIS implementation from [1], which
%   must be present on the Matlab path and can be obtained from [2].
%
% PARAMETERS
%   lw      1 times M vector of log-weights (normalized).
%   par     Addtional (optional) parameters:
%   
%               kt      Resampling threshold (default: 0.5).
%               smooth  Enable/disable smoothing of the particle weights 
%                       (default: false).
%               resample    Resampling function handle (default: 
%                           sysresample).
%   
% RETURNS
%   alpha   Resampled indices.
%   lw      Log-weights
%   r       Indicator whether resampling was performed or not.
%
% REFERENCES
%   [1] A. Vehtari, A. Gelman and J. Gabry, "Pareto smoothed importance 
%       sampling," arXiv preprint arXiv:1507.02646, 2016
%
%   [2] https://github.com/avehtari/PSIS
%
% AUTHORS
%   2017-11-02 -- Roland Hostettler <roland.hostettler@aalto.fi>

    %% Defaults
    narginchk(1, 2);
    if nargin < 2
        par = [];
    end
    def = struct( ...
        'kt', 0.5, ...
        'smooth', false, ...
        'resample', @sysresample ...
    );
    par = parchk(par, def);

    %% Resampling
    [lws, khat] = psislw(lw.');
    if par.smooth
        lw = lws.';
    end
    r = (khat > par.kt);
    M = length(lw);
    alpha = 1:M;
    if r
        w = exp(lw);
        alpha = par.resample(w);
        lw = log(1/M)*ones(1, M);
    end
end
