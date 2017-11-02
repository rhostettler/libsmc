function [x, lw, r] = psisresample(x, lw, kt, smooth)
% PSIS-based resampling
%
% SYNOPSIS
%   
%
% DESCRIPTION
%   
%
% PARAMETERS
%   
%
% VERSION
%   2016-
%
% AUTHORS
%   Roland Hostettler <roland.hostettler@aalto.fi>

    %% Defaults
    narginchk(2, 4);
    if nargin < 3 || isempty(kt)
        kt = 0.5;
    end
    if nargin < 4 || isempty(smooth)
        smooth = false;
    end

    %% Resampling
    [lws, khat] = psislw(lw.');
%     khat = fittail(lw.');
    if smooth
        lw = lws.';
    end
    r = (khat > kt);
    if r
        w = exp(lw);
        M = length(lw);
        ir = sysresample(w);
        x = x(:, ir);
        lw = log(1/M)*ones(1, M);
        r = true;
    end
end
