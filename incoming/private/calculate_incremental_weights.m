function lv = calculate_incremental_weights(y, xp, x, t, model, q)
% Calculate the incremental weights
% 
% SYNOPSIS
%   lv = CALCULATE_INCREMENTAL_WEIGHTS(y, xp, x, t, model, q)
%
% DESCRIPTION
%   Calculates the incremental weights in sequential importance sampling.
%   This function is only a wrapper function and delegates the actual
%   computation forward to an appropriate weight calculatation function
%   depending on the type of proposal q (e.g. bootstrap, etc.).
%
% PARAMETERS
%   y       Measurement y[n].
%   xp      Particles at time t[n] (i.e. x[n]).
%   x       Particles at time t[n-1] (i.e. x[n-1]).
%   t       Time t[n] or time difference t[n]-t[n-1].
%   model   Model structure.
%   q       Proposal density used.
%
% RETURNS
%   lv      Log of non-normalized incremental particle weights, that is,
%           log(v[n]).
%
% AUTHORS
%   2017-11-02 -- Roland Hostettler <roland.hostettler@aalto.fi>

    narginchk(6, 6);
    warning('This function is deprecated and should be called anymore. Instead, set the appropriate value for par.calculate_incremental_weights.');
    if isfield(q, 'bootstrap') && q.bootstrap
        lv = calculate_incremental_weights_bootstrap(y, xp, x, t, model, q);
    else
        lv = calculate_incremental_weights_generic(y, xp, x, t, model, q);
    end
end
