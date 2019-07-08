function [alpha, state] = sample_ancestor_index(model, y, xtilde, x, lw, theta)
% Ancestor index sampling for the conditional particle filter
%
% USAGE
%   [alpha, state] = SAMPLE_ANCESTOR_INDEX(model, y, xtilde, x, lw, theta)
%
% DESCRIPTION
%   Calculates the ancestor weights for Markovian state-space models given
%   by
%
%       vtilde = w[n-1]^j*p(xtilde[n] | x[n]^j)
%       wtilde = vtilde/sum(vtilde)
%
%   and samples an ancestor index from the categorical distribution defined
%   by the ancestor weights.
%
%   This is the default way of sampling the ancestor indices in the
%   conditional particle filter with ancestor sampling (CPF-AS).
%
% PARAMETERS
%   model   Model structure
%   y       Measurement vector
%   xtilde  Sample of the seed trajectory, xtilde[n]
%   x       Matrix of particles x[n-1]^j
%   lw      Row vector of particle log-weights log(w[n-1]^j)
%   theta   Additional parameters
%
% RETURNS
%   alpha   Sampled ancestor index
%   state   Sampler state (empty)
%
% AUTHOR(S)
%   2017-2019 -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

    %% Defaults
    narginchk(6, 6);
    state = [];
    J = size(x, 2);
    
    %% Sampling
    lvtilde = calculate_ancestor_weights(model, y, xtilde, x, lw, theta);
    wtilde = exp(lvtilde-max(lvtilde));
    wtilde = wtilde/sum(wtilde);
    tmp = sysresample(wtilde);
    alpha = tmp(randi(J, 1));
end
