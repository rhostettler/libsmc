function [mz, Pz] = initialize_kf(model, s)
% # Conditional Kalman filter initialization
% ## Usage
% * [mz, Pz] = initialize_kf(model, s)`
% 
% ## Description
% Initialization of the mean and covariance for conditionally linear states
% `z` given the nonlinear states `s`.
%
% ## Input
% * `model`: Model struct, must contain the following fields:
%    - `model.pz0.m`: Function handle for the initial conditional mean.
%    - `model.pz0.P`: Function handle for the initial conditional
%      covariance.
%    - `model.pz0.fast`: Indicator whether the mean and covariance can be
%      evaluated using a sigle function call.
% * `s`: ds-times-J matrix of nonlinear states.
%
% ## Output
% * `mz`: dz-times-J matrix of initial means.
% * `Pz`: dz-times-dz-times-J array of initial covariances.
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

% TODO:
% * Some parts are still hacky and need proper revision.

    narginchk(2, 2);
    if model.pz0.fast
        mz = model.pz0.m(s);
        Pz = model.pz0.P(s);
    else
        J = size(s, 2);
        dz = size(model.pz0.m(s(:, 1)), 1);
        mz = zeros(dz, J);
        Pz = zeros([dz, dz, J]);
        for j = 1:J
            mz(:, j) = model.pz0.m(s(:, j));
            Pz(:, :, j) = model.pz0.P(s(:, j));
        end
%     m0 = model.px0.m;
%     P0 = model.px0.P;
%     s = m0(in)*ones(1, N) + chol(P0(in, in)).'*randn(ds, N);
%     mz = m0(il)*ones(1, N) + P0(il, in)/P0(in, in)*(s - m0(in)*ones(1, N));
%     Pz = P0(il, il) - P0(il, in)/P0(in, in)*P0(in, il);
    end
end
