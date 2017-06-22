function model = wiener_model(F, Q, g, R, m0, P0)
% Initializes a Wiener state-space model
%
% SYNOPSIS
%   model = wiener_model(F, Q, g, R, m0, P0)
%
% DESCRIPTION
%
%
% PARAMETERS
%
% RETURNS
%
% VERSION
%   2017-03-28
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>


    %% 
    
    % TODO: 'functionize'
    if ~isa(F, 'function_handle')
        F = @(t) F;
    end
    if ~isa(Q, 'function_handle')
        Q = @(t) Q;
    end
    
    Nx = size(m0, 1);
    C0 = chol(P0).';
    px0 = struct('rand', @(M) m0*ones(1, M)+C0*randn(Nx, M));
    
    % State transition densiy
    px = struct( ...
        'fast', 1, ...
        'rand', @(x, t) F(t)*x + chol(Q(t)).'*randn(Nx, size(x, 2)), ...
        'logpdf', @(xp, x, t) logmvnpdf(xp.', (F(t)*x).', Q(t).').', ...
        'pdf', @(xp, x, t) mvnpdf(xp.', (F(t)*x).', Q(t).').' ...
    );
    py = struct();
       
    model = struct('F', F, 'Q', Q, 'g', g, 'R', R, 'px', px, 'py', py, 'px0', px0, 'm0', m0, 'P0', P0);
end
