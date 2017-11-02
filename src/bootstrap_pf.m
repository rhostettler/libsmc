function [xhat, sys] = bootstrap_pf(y, t, model, M, par)
% Bootstrap particle filter
%
% SYNOPSIS
%   xhat = BOOTSTRAP_PF(y, t, model, M)
%   [xhat, sys] = BOOTSTRAP_PF(y, t, model, M, par)
%
% DESCRIPTION
%   This is the very common bootstrap particle filtering algorithm which
%   uses the dynamic model as the proposal distribution and the incremental
%   weights reduce to the likelihood.
%
% PARAMETERS
%   y       Ny times N matrix of measurements.
%   t       1 times N vector of timestamps.
%   model   State space model structure.
%   M       Number of particles (optional, default: 100).
%   par     Structure of additional (optional) parameters:
%
%           [alpha, lw, r] = resample(lw)
%               Function handle to the resampling function. The argument lw
%               is the log-weights and the must return the indices of the
%               resampled (alpha) particles, the weights of the resampled 
%               (lw) particles, as well as a bool indicating whether
%               resampling was performed or not.
%
% RETURNS
%   xhat    Minimum mean squared error state estimate (calculated using the
%           marginal filtering density).
%   sys     Particle system array of structs with the following fields:
%           
%               xf  Nx times M matrix of particles for the marginal
%                   filtering density.
%               wf  1 times M vector of the particle weights for the
%                   marginal filtering density.
%               af  1 times M vector of ancestor indices.
%               r   Boolean resampling indicator.
%
% AUThORS
%   2017-03-27 -- Roland Hostettler <roland.hostettler@aalto.fi>

    %% Defaults
    narginchk(3, 5);
    if nargin < 4 || isempty(M)
        M = 100;
    end
    if nargin < 5
        % Initialize the parameters if none were given; by default, we use
        % the defaults from sisr_pf. This helps simplifying code
        % maintenance.
        par = struct();
    end
    modelchk(model);
    
    %% Filtering
    % The Bootstrap PF is a spcial case of the more general SISR PF where
    % the proposal is the dynamic model. Hence, we simply set the proposal
    % accordingly and let sisr_pf to the work. Furthermore, we also set the
    % function to calculate the incremental weights (we could use the
    % generic one, but the one for bootstrap is somewhat faster).
    q = struct();
    q.fast = model.px.fast;
    q.logpdf = @(xp, y, x, t) model.px.logpdf(xp, x, t);
    % q.pdf = @(xp, y, x, t) px.pdf(xp, x, t);
    q.rand = @(y, x, t) model.px.rand(x, t);
    par.calculate_incremental_weights = @calculate_incremental_weights_bootstrap;
    
    if nargout == 1
        xhat = sisr_pf(y, t, model, q, M, par);
    else
        [xhat, sys] = sisr_pf(y, t, model, q, M, par);
    end
end
