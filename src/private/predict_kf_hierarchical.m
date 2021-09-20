function [mzp, Pzp] = predict_kf_hierarchical(model, sp, alpha, ~, mz, Pz, theta)
% # Conditional Kalman filter prediction for hierarchical models
% ## Usage
% * `[mzp, Pzp] = predict_kf_hierarchical(model, sp, alpha, s, mz, Pz, theta)`
%
% ## Description
% Kalman filter prediction for hierarchical conditionally linear Gaussian
% state-space models with dynamic model
%
%    s[n] ~ p(s[n] | s[n-1]; theta)
%    z[n] = g(s[n], theta) + G(s[n], theta)*z[n] + q[n]
%
% with `q[n] ~ N(0, Q(s[n], theta))`.
%
% ## Input
% * `model`: Model struct, must contain the following fields:
%    - `model.pz.g`: Function handle for the nonlinear function.
%    - `model.pz.G`: Function handle for the linear state transition 
%      matrix.
%    - `model.pz.Q`: Function handle for the process noise covariance
%      matrix.
% * `sp`: ds-times-J matrix of nonlinear states `s[n]`.
% * `alpha`: 1-times-J vector of ancestor indices for the state `s[n]`.
% * `s`: ds-times-J matrix of nonlinear states `s[n-1]`.
% * `mz`: dz-times-J matrix of previous means of the linear states.
% * `Pz`: dz-times-dz-times-J array of previous covariances of the linear
%   states.
% * `theta`: Additional parameters.
% 
% ## Output
% * `mzp`: dz-times-J matrix of predicted means.
% * `Pzp`: dz-times-dz-times-J array of predicted covariances.
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

    narginchk(7, 7);
    mz = mz(:, alpha);
    Pz = Pz(:, :, alpha);
    [dz, J] = size(mz);
    mzp = zeros(dz, J);
    Pzp = zeros([dz, dz, J]);
    for j = 1:J
        spj = sp(:, j);
        gn = model.pz.g(spj, theta);
        Gn = model.pz.G(spj, theta);
        Qn = model.pz.Q(spj, theta);
        
        mzp(:, j) = gn + Gn*mz(:, j);
        Pzp(:, :, j) = Gn*Pz(:, :, j)*Gn' + Qn;
    end
end
