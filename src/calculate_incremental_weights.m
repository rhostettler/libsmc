function [v, lv] = calculate_incremental_weights(y, xp, x, t, model, q)
% Calculate the incremental weights in SMC
% 
% SYNOPSIS
%   [v, lv] = calculate_incremental_weights(y, xp, x, t, model, q)
%
% DESCRIPTION
%   
%
% PARAMETERS
%   y       Measurement y[n]
%
%   xp      Particles at time t[n] (i.e. x[n])
%
%   x       Particles at time t[n-1] (i.e. x[n-1])
%
%   t       Time t[n] or time difference t[n]-t[n-1]
%
%   model   Model structure
%
%   q       Proposal density used
%
% RETURNS
%   v       Non-normalized incremental particle weights v[n]
%
%   lv      Log of non-normalized incremental particle weights, that is,
%           log(v[n])
%
% VERSION
%   2017-06-18
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * Description in documentation

    narginchk(6, 6);
    if q.bootstrap
        [v, lv] = bootstrap_incremental_weights(y, xp, t, model);
    else
        [v, lv] = sis_incremental_weights(y, xp, x, t, model, q);
    end
end
