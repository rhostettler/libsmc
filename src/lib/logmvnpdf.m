function px = logmvnpdf(x, m, C)
% Logarithm of Multivariate Normal PDF
%
% SYNOPSIS
%   px = logmvnpdf(x)
%   px = logmvnpdf(x, m, C)
%
% DESCRIPTION
%   Returns the logarithm of N(x; m, C) or N(x; 0, I) if m and C are
%   omitted, i.e. the log-likelihood. Everything is calculated in
%   log-domain such that numerical precision is retained.
%
%   The arguments x and m are automatically expanded to match each other.
%
% PARAMETERS
%   x   MxN vector of values to evaluate.
%   m   MxN vector of means (optional, default: 0).
%   C   Covariance matrix (optional, default: I).
%
% VERSION
%   2016-12-02
%
% AUTHORS
%   Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * Include sanity checks for C and the M-sizes of the vectors.
%   * Allow for N-D covariance matrices
%   * If Nx1 is supplied instad of 1xN, the algorithm breaks; add sanity
%     check for that.

    %% Autocomplete
    switch nargin
        case 1
            m = zeros(size(x));
            C = eye(size(x, 2));
        case 2
            C = eye(size(x, 2));
    end

    % Some sanity checks
    [Nx, Mx] = size(x);
    [Nm, Mm] = size(m);
    Nv = size(C, 1);
    
    % Automagically expand the arguments
    if Nx ~= Nm
        if Nx == 1 && Nm > 1
            x = ones(Nm, 1)*x;
        elseif Nx > 1 && Nm == 1
            m = ones(Nx, 1)*m;
        else
            error('Input argument size mismatch');
        end
    end
    
    %% Calculation
    Cinv = C\eye(Nv);
    Linv = chol(Cinv).';
if 0
    a = zeros(Nx, 1);
    for i = 1:Nx
        epsilon = (x(i, :) - m(i, :))*Linv;
        a(i) = -1/2*(epsilon*epsilon');
    end
end
    
    epsilon = (x-m)*Linv;
    a = -1/2*sum(epsilon.^2, 2);
    b = -1/2*log(det(C));
    c = -Nv/2*log(2*pi);
    px = a + b + c;
end
