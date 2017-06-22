function [xhat, Phat, sys] = bootstrap_pf(y, t, model, M, par)
% Bootstrap for AWG process & measurement noise, and Gau
%
% SYNOPSIS
%   [xhat, Phat] = bootstrap_pf(y, t, model, M)
%   [xhat, Phat, sys] = bootstrap_pf(y, t, model, M, par)
%
% DESCRIPTION
%      ------------------
%   A simple bootstrap particle filter that assumes zero-mean additive
%   Gaussian process and measurement nosie with covariance Q and R,
%   respectively, non-linear state transition function f and observation
%   function g, and initial distribution N(m0, P0).
%
%%%%%%%%%   Uses systematic resampling whenever the ESS < M/3.
%
% PARAMETERS
%   ------
%
% VERSION
%   2017-03-27
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

    %% Preliminary Checks
    % Check that we get the correct no. of parameters and a well-defined
    % model so that we can detect model problems already here.
    narginchk(3, 5);
    if nargin < 4 || isempty(M)
        M = 100;
    end
    if nargin < 5
        par = [];
    end
    
    % Default parameters
    def = struct(...
        'resample', @resample_ess, ... % Resampling function
        'Mt', M/3 ...       % Resampling threshold
    );
    par = parchk(par, def);
    par.bootstrap = 1;
    [px, ~, ~] = modelchk(model);
    
    %% Filtering
    % Bootstrap PF is nothing but a general SISR PF with the proposal being
    % the dynamcis
    q.fast = px.fast;
    q.logpdf = @(xp, y, x, t) px.logpdf(xp, x, t);
    % q.pdf = @(xp, y, x, t) px.pdf(xp, x, t);
    q.rand = @(y, x, t) px.rand(x, t);
    switch nargout
        case {1, 2}
            [xhat, Phat] = sisr_pf(y, t, model, q, M, par);
        case 3
            [xhat, Phat, sys] = sisr_pf(y, t, model, q, M, par);
    end
end
