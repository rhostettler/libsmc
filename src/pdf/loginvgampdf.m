function lp = loginvgampdf(x, alpha, beta)
% Logarithm of inverse Gamma probability density function
%
% USAGE
%   lp = LOGINVGAMPDF(x)
%   lp = LOGINVGAMPDF(x, alpha, beta)
%
% DESCRIPTION
%   Calculates the log-value of the PDF for an inverse Gamma distributed
%   variable x with parameters alpha and beta. The inverse Gamma density in
%   this parametrization is given by
%
%               beta^alpha
%       p(x) = ------------*x^(-alpha-1)*exp(-beta/x),
%              Gamma(alpha)
%
%   where Gamma(.) is the Gamma-function.
%
% PARAMETERS
%   x           Point to evaluate the PDF in.
%   alpha, beta Density parameters (default: 1).
%
% RETURNS
%   lp          Log-likelihood.
%
% AUTHORS
%   2017 -- Roland Hostettler <roland.hostettler@aalto.fi>

    % Defaults
    narginchk(1, 3);
    if nargin < 2 || isempty(alpha)
        alpha = 1;
    end
    if nargin < 3 || isempty(beta)
        beta = 1;
    end
    if alpha <= 0
        error('''alpha'' must be strictly positive (%f).', alpha);
    end
    if beta <= 0
        error('''beta'' must be strictly positive (%f).', beta);
    end

    % Calculate density value
    lp = alpha*log(beta)-log(gamma(alpha))-(alpha+1)*log(x)-beta./x;
end
