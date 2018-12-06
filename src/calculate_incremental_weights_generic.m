function lv = calculate_incremental_weights_generic(y, xp, x, t, model, q)
% General incremental weights for sequential importance sampling
%
% USAGE
%   lv = CALCULATE_INCREMENTAL_WEIGHTS_GENERIC(y, xp, x, t, model, q)
%
% DESCRIPTION
%   Calculates the incremental importance weight for sequential importanc
%   sampling for an arbitrary proposal density q.
%
%   Note that the function actually calculates the non-normalized
%   log-weight to improve numerical stability.
%
% PARAMETERS
%   y       Measurement y[n].
%   xp      New state x[n].
%   x       Previous state x[n-1].
%   t       Timestamp.
%   model   State-space model struct.
%   q       Proposal density struct.
%
% RETURNS
%   lv      Logarithm ov incremental weights.
%
% SEE ALSO
%   sample_generic
%
% AUTHORS
%   2017-04-07 -- Roland Hostettler <roland.hostettler@aalto.fi>

    narginchk(6, 6);
    M = size(xp, 2);
    px = model.px;
    py = model.py;    
    if px.fast && py.fast && q.fast
        lv = ( ...
            py.logpdf(y*ones(1, M), xp, t) ...
            + px.logpdf(xp, x, t) ...
            - q.logpdf(xp, y*ones(1, M), x, t) ...
        );
    else
        M = size(xp, 2);
        lv = zeros(1, M);
        for m = 1:M
            lv(m) = ( ...
                py.logpdf(y, xp(:, m), t) ...
                + px.logpdf(xp(:, m), x(:, m), t) ...
                - q.logpdf(xp(:, m), y, x(:, m), t) ...
            );
        end
    end
end
