function [alpha, lalpha, state] = resample_ess(lw, par)
% # Effective sample size-based conditional resampling
% ## Usage
% * `[alpha, lalpha, state] = resample_ess(lw)`
% * `[alpha, lalpha, state] = resample_ess(lw, par)`
%
% ## Description
% Conditional resampling function using an estimate of the effecitve sample
% size (ESS) given by
%
%     J_ess = 1./sum(w.^2)
%
% as the resampling criterion. By default, the resampling threshold is set
% to M/3 and systematic resampling is used. The resampled ancestor indices
% are returned in the 'alpha'-variable, together with the updated (or 
% unaltered if no resampling was done) log-weights.
%
% ## Input
% * `lw`: Normalized log-weights.
% * `par`: Struct of optional parameters. Possible parameters are:
%     - `Jt`: Resampling threshold (default: J/3).
%     - `resample`: Resampling function handle (default: 
%       `@resample_stratified`).
%
% ## Output
% * `alpha`: Resampled indices.
% * `lalpha`: Log-weights.
% * `state`: Resampling state, contains the following fields:
%     - `ess`: Effective sample size.
%     - `r`: Resampling indicator (`true` if resampled, `false` otherwise).
%
% ## Authors
% 2017-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

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

    %% Defaults
    narginchk(1, 2);
    if nargin < 2
        par = struct();
    end
    J = length(lw);
    def = struct( ...
        'Jt', J/3, ...                          % Resampling threshold
        'resample', @resample_stratified ...    % Resampling function
    );
    par = parchk(par, def);

    %% Resampling
    w = exp(lw);
    Jess = 1/sum(w.^2);
    r = (Jess < par.Jt);
    alpha = 1:J;
    lalpha = log(1/J)*ones(1, J);
    if r
        % ESS is too low: Sample according to the categorical distribution
        % defined by the sample weights w, alpha ~ Cat{w}
        alpha = par.resample(w);
        lalpha = lw(alpha);
    end
    state = struct('r', r, 'ess', Jess);
end
