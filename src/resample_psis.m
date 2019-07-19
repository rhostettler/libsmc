function [alpha, lw, state] = resample_psis(lw, par)
% # Pareto smoothing importance sampling resampling
% ## Usage
% * `[alpha, lw, state] = resample_psis(lw, par)`
%
% ## Description
% Conditional resampling based on fitting a pareto distribution to the tail
% of the importance weights (~ pareto smoothed importance sampling).
% Resampling is performed if the k parameter of the fitted pareto
% distribution falls below the threshold kt, meaning that all the moments
% larger than 1/k are undefined for the weights' distribution. In practice,
% if kt = 0.5, then once the estimated k is smaller than 0.5, the weights'
% variance does not exist.
%
% This method requires (and uses) the PSIS implementation from [1,2], which
% is a submodule in the `external` directory.
%
% ## Input
% % * `lw`: Normalized log-weights.
% * `par`: Struct of optional parameters. Possible parameters are:
%     - `kt`: Resampling threshold (default: `0.5`).
%     - `smooth`: Use smoothed weights if not resampling (default:
%       `false`).
%     - `resample`: Resampling function handle (default: 
%       `@resample_stratified`).
%
% ## Output
% * `alpha`: Resampled indices.
% * `lw`: Log-weights.
% * `state`: Resampling state, contains the following fields:
%     - `ess`: Effective sample size.
%     - `r`: Resampling indicator (`true` if resampled, `false` otherwise).
%
% ## References
% 1. A. Vehtari, A. Gelman and J. Gabry, "Pareto smoothed importance 
%    sampling," arXiv preprint arXiv:1507.02646, 2016
% 2. https://github.com/avehtari/PSIS
% 
% ## Authors
% 2017-present -- Roland Hostettler

%{
% This file is part of the libsmc Matlab toolbox.
%
% libsmc is free software: you can redistribute it and/or modify it under 
% the terms of the GNU General Public License as published by the Free 
% Software Foundation, either version 3 of the License, or (at your option)
% any later version.
% 
% libsmc is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
% details.
% 
% You should have received a copy of the GNU General Public License along 
% with libsmc. If not, see <http://www.gnu.org/licenses/>.
%}

% TODO:
% * Make this fit for APF-like interfaces. Needs to be done, of course.

    %% Defaults
    narginchk(1, 2);
    if nargin < 2
        par = struct();
    end
    def = struct( ...
        'kt', 0.5, ...
        'smooth', false, ...
        'resample', @resample_stratified ...
    );
    par = parchk(par, def);

    %% Resampling
    [lws, khat] = psislw(lw.');
    if par.smooth
        lw = lws.';
    end
    r = (khat > par.kt);
    J = length(lw);
    alpha = 1:J;
    if r
        w = exp(lw);
        alpha = par.resample(w);
        lw = log(1/J)*ones(1, J);
    end
    state = struct('r', r, 'khat', khat);
end
