function [xp, px] = sample_bootstrap(model, y, x, theta)
% Bootstrap sampling
% 
% USAGE
%   xp = SAMPLE_BOOTSTRAP(model, y, x, theta)
%   [xp, q] = SAMPLE_BOOTSTRAP(model, y, x, theta)
%
% DESCRIPTION
%
% PARAMETERS
%
% RETURNS
%
%
%
% AUTHOR(S)
%   2017-2019 -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

    %% Defaults
    narginchk(4, 4);
    [Nx, J] = size(x);

    %% Sample
    px = model.px;
    if px.fast
        xp = px.rand(x, theta);
    else
        xp = zeros(Nx, J);
        for j = 1:J
            xp(:, j) = px.rand(x(:, j), theta);
        end
    end
end
