function lvtilde = calculate_ancestor_weights(model, y, xtilde, x, lw, theta)
% Calculate ancestor weigths for CPF-AS and Markovian state-space models
%
% USAGE
%   lvtilde = calculate_ancestor_weights(model, y, xtilde, x, lw, theta)
%
% DESCRIPTION
%   Calculates the non-normalized ancestor weights for the conditional
%   particle filter with ancestor sampling and Markovian state-space
%   models.
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
%   lvtilde Non-normalized ancestor weight.
%
% AUTHOR(S)
%   2017-2019 -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

    %% Defaults
    narginchk(6, 6);
    J = size(x, 2);
    
    %% Calculate weights
    px = model.px;
    if px.fast
        lvtilde = lw + px.logpdf(xtilde*ones(1, J), x, theta);
    else
        lvtilde = zeros(1, J);
        for j = 1:J
            lvtilde(j) = lw(j) + px.logpdf(xtilde, x(:, j), theta);
        end
    end
end
