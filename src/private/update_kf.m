function [mz, Pz] = update_kf(model, y, s, mzp, Pzp, theta)
% # Conditional Kalman filter update
% ## Usage
% * `[mz, Pz] = update_kf(model, y, s, mzp, Pzp, theta)`
% 
% ## Description
% Kalman filter update for conditionally linear measurement models of the
% form
%
%    y[n] = h(s[n], theta) + H(s[n], theta)*z[n] + r[n]
%
% with `r[n] ~ N(0, R(s[n], theta))`.
%
% ## Input
% * `model`: Model struct, must contain the following fields:
%    - `model.py.h`: Function handle of the nonlinear function.
%    - `model.py.H`: Function handle of the sensing matrix.
%    - `model.py.R`: Function handle of the measurement noise covariance
%      matrix.
% * `y`: Measurement sample.
% * `s`: ds-times-J matrix of nonlinear states.
% * `mzp`: dz-times-J matrix of predictions of the linear states.
% * `Pzp`: dz-times-dz-times-J array of predictive covariances of the
%   linear states.
% * `theta`: Additional parameters.
%
% ## Output
% * `mz`: dz-times-J matrix of posterior means.
% * `Pz`: dz-times-dz-times-J array of posterior covariances.
%
% ## References
% 1. T. Schön, F. Gustafsson, and P.-J. Nordlund, “Marginalized particle 
%    filters for mixed linear/nonlinear state-space models,” IEEE 
%    Transactions on Signal Processing, vol. 53, no. 7, pp. 2279–2289, 
%    July 2005.
% 2. F. Lindsten, P. Bunch, S. Särkkä, T. B. Schön, and S. J. Godsill, 
%    “Rao–Blackwellized particle smoothers for conditionally linear 
%    Gaussian models,” IEEE Journal of Selected Topics in Signal 
%    Processing, vol. 10, no. 2, pp. 353–365, March 2016.
%
% ## Authors
% 2019-present -- Roland Hostettler

%{
% This file is part of the libsmc Matlab toolbox.
%
% libsmc is free software: you can redistribute it and/or modify it under 
% the terms of the GNU General Public License as published by the Free 
% Software Foundation, either version 3 of the License, or (at your option)
% any later version.
% 
% libsmc is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
% details.
% 
% You should have received a copy of the GNU General Public License along 
% with libsmc. If not, see <http://www.gnu.org/licenses/>.
%}

    narginchk(6, 6);
    [dz, J] = size(mzp);
    mz = zeros(dz, J);
    Pz = zeros([dz, dz, J]);
    
    for j = 1:J
        hj = model.py.h(s(:, j), theta);
        Hj = model.py.H(s(:, j), theta);
        Rj = model.py.R(s(:, j), theta);

        Mn = Hj*Pzp(:, :, j)*Hj' + Rj;
        Kn = Pzp(:, :, j)*Hj'/Mn;
        mz(:, j) = mzp(:, j) + Kn*(y - hj - Hj*mzp(:, j));
        Pz(:, :, j) = Pzp(:, :, j) - Kn*Mn*Kn';
    end
end
