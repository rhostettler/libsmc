function [alpha, lw, r] = resample_ess(lw, par)
% Effective sample size-based resampling
%
% SYNOPSIS
%   [alpha, lw, r] = resample_ess(lw, par)
%
% DESCRIPTION
%   Conditional resampling function using an estimate of the effecitve
%   sample size (ESS) given by
%
%       M_ess = 1./sum(w.^2)
%
%   as the resampling criterion. The default resampling threshold is when
%   the ESS falls below M/3 and by default, systematic resampling is used.
%   The resampled ancestor indices are returned in the 'alpha'-variable,
%   together with the updated (or unaltered if no resampling was done) log-
%   weights.
%
% PARAMETERS
%   lw  Normalized log-weights.
%
%   par Struct of optional parameters. Possible parameters are:
%       
%       Mt          Resampling threshold (default: M/3).
%       resample    Resampling function handle (default: sysresample).
%
% RETURNS
%   alpha
%       Resampled indices.
%
%   lw  Log-weights
%
%   r   Indicator whether resampling was performed or not.
%
% VERSION
%   2017-04-29
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

    %% Default Parameters
    narginchk(1, 2);
    if nargin < 2
        par = [];
    end
    M = length(lw);
    def = struct( ...
        'Mt', M/3, ...                  % Resampling threshold
        'resample', @sysresample ...    % Resampling function
    );
    par = parchk(par, def);

    %% Resampling
    w = exp(lw);
    Meff = 1/sum(w.^2);
    r = (Meff < par.Mt);
    alpha = 1:M;
    if r
        alpha = sysresample(w); %par.resample(w); % TODO: Sever bug here: par.resample can point to itself
        lw = log(1/M)*ones(1, M);
    end
end
