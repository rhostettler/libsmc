function xp = sample_bootstrap2(model, ~, x, theta)
% Sample from the bootstrap proposal for non-Markovian SSMs
%
% USAGE
%   FIXME
%
% DESCRIPTION
%   FIXME
%
% PARAMETERS
%   FIXME
%
% RETURNS
%   FIXME
%
% AUTHOR
%   2017-2019 -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

% TODO:
%   * Put into non-hacky state

    [Nx, J, ~] = size(x);
    px = model.px;
    if px.fast
        xp = px.rand(x, theta);
    else
        xp = zeros(Nx, J);
        for j = 1:J
            xp(:, j) = px.rand(x(:, j, :), theta);
        end
    end
end
