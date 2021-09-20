function [shat, zhat, sys] = rbpf(model, y, theta, J, par)
% # Rao-Blackwellized particle filter for CLGSS models
% ## Usage
% * `xhat = rbpf(model, y)`
% * `[xhat, sys] = rbpf(model, y, theta, J, par)`
% 
% ## Description
% Generic Rao-Blackwellized particle filter (RBPF) for conditionally linear
% Gaussian state-space models. The method is suitable for both mixing
% models of the form [1]
%
%     s[n] = f(s[n-1]) + F(s[n-1])*z[n-1] + qs[n],
%     z[n] = g(s[n-1]) + G(s[n-1])*z[n-1] + qz[n],
%     y[n] = h(s[n]) + H(s[n])*z[n] + r[n],
%
% with q[n] ~ N(0, Q) and r[n] ~ N(0, Q) as well as hierarchical models [2]
%
%     s[n] ~ p(s[n] | s[n-1]),
%     z[n] = g(s[n]) + G(s[n])*z[n-1] + qz[n],
%     y[n] = h(s[n]) + H(s[n])*z[n] + r[n],
%
% with qz[n] ~ N(0, Qz), r[n] ~ N(0, R) and where s[n] is the nonlinear 
% state and z[n] is the linear state.
% 
% ## Input
% * `model`: State-space model structure. In addition to the standard
%   fields, the following additional fields are required by RBPFs:
%
% * `y`: dy-times-N measurement matrix.
% * `theta`: Additional model parameters.
% * `J`: Number of particles (default: `100`)
% * `par`: Struct of additional algorithm parameters:
%     - `XXXXX`
% 
%
% ## Output
% * `xhat`: 
% * `sys`: 
%
% ## References
% 1. T. B. Schon, F. Gustafsson, P.-J. Nordlund, "Marginalized Particle
%    Filters for Mixed Linear/Nonlinear State-Space Models", IEEE
%    Transactions on Signal Processing", vol. 53, no. 7, July 2005
% 2. 
% 
% ## Authors
% 2017-present -- Roland Hostettler

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
% * Document the convention used here: x corresponds to s, we have the
%   additonal fields m and P
% * Handle different cases, e.g., if H doesn't depend on s/theta, fast 
%   evaluation, etc. (h might be empty, H might be empty, etc.) (basically
%   in the KF parts...).
%   => We might want to do this through the kf_update-functions.
% * Only hierarchical model is revised and implemented just now.
% * Document the additional model structs: ps0, pz0, ps, pz, pym
    
    %% Defaults
    narginchk(2, 5);
    if nargin < 3 || isempty(theta)
        theta = NaN;
    end
    if nargin < 4 || isempty(J)
        J = 100;
    end
    if nargin < 5 || isempty(par)
        par = struct();
    end
    def = struct( ...
        'resample', @resample_ess, ...
        'sample', @sample_bootstrap, ...
        'calculate_incremental_weights', @calculate_incremental_weights_bootstrap, ...
        'predict_kf', @predict_kf_hierarchical, ...
        'update_kf', @update_kf ...
    );
    par = parchk(par, def);
    
    %% Initialize
    % Sample initial particles and intialize linear states
    s = model.ps0.rand(J);
    lw = log(1/J)*ones(1, J);
    [mz, Pz] = initialize_kf(model, s);
    
    % Expand data dimensions, prepend NaN for zero measurement
    [dy, N] = size(y);
    y = [NaN*ones(dy, 1), y];
    
    % Expand 'theta'
    [dtheta, Ntheta] = size(theta);
    if Ntheta == 1
        theta = theta*ones(1, N);
    end
    theta = [NaN*ones(dtheta, 1), theta];

    %% Preallocate
    ds = size(s, 1);
    dz = size(mz, 1);
    N = N+1;
    return_sys = (nargout >= 3);
    if return_sys
        sys = initialize_sys(N, ds, J);
        sys(1).x = s;
        sys(1).w = exp(lw);
        sys(1).mzp = [];
        sys(1).Pzp = [];
        sys(1).mz = mz;
        sys(1).Pz = Pz;
        sys(1).alpha = 1:J;
        sys(1).rstate = struct('r', false, 'ess', J);
        sys(1).qstate = [];
    end
    shat = zeros(ds, N-1);
    zhat = zeros(dz, N-1);

    %% Process data
    for n = 2:N
        %% Update
        % Resample and draw particles
        [alpha, lw, rstate] = par.resample(lw);
        s = s(:, alpha);
        mz = mz(:, alpha);
        Pz = Pz(:, :, alpha);
        [sp, lqs, qstate] = par.sample(model, y(:, n), s, mz, Pz, theta(:, n));
        
        % Prediction for linear states
        [mzp, Pzp] = par.predict_kf(model, sp, s, mz, Pz, theta(:, n));
        
        % Calculate particle weights
        lv = par.calculate_incremental_weights(model, y(:, n), sp, s, mzp, Pzp, theta(:, n), lqs);
        lw = lw + lv;
        w = exp(lw-max(lw));
        w = w/sum(w);
        lw = log(w);
        s = sp;
        if any(~isfinite(w))
            warning('libsmc:warning', 'NaN/Inf in particle weights.');
        end
        
        % Measurement update for linear states
        [mz, Pz] = par.update_kf(model, y(:, n), s, mzp, Pzp, theta(:, n));
        
        %% Point estimates
        % MMSE
        shat(:, n-1) = s*w';
        zhat(:, n-1) = mz*w';

        %% Store
        if return_sys
            sys(n).x = s;       % N.B.: This is a slight abuse of the struct but avoids breaking everything else.
            sys(n).mzp = mzp;   % TODO: When calculating the lineages, these fields are ignored.
            sys(n).Pzp = Pzp;
            sys(n).mz = mz;
            sys(n).Pz = Pz;
            sys(n).w = w;
            sys(n).alpha = alpha;
            sys(n).rstate = rstate;
            sys(n).qstate = qstate;
        end
    end
    
    %% Calculate full state trajectories
    % TODO: Currently ignores fields other than 'x' and thus needs to be
    % updated somehow (or replaced by a RB-version).
    if return_sys
        sys = calculate_particle_lineages(sys);
    end
end

%% Bootstrap sampling function
% Overwrites global bootstrap sampling function; samples from the marginal 
% dynamics p(s[n] | s[n-1]).
function [sp, lqs, qstate] = sample_bootstrap(model, ~, s, mz, Pz, theta)
    qstate = [];
    if model.ps.fast
        sp = model.ps.rand(s, mz, Pz, theta);
        lqs = model.ps.logpdf(sp, s, mz, Pz, theta);
    else
        [ds, J] = size(s, 2);
        sp = zeros(ds, J);
        lqs = zeros(1, J);
        for j = 1:J
            sp(:, j) = model.ps.rand(s(:, j), mz(:, j), Pz(:, :, j), theta);
            lqs(:, j) = model.ps.logpdf(sp(:, j), s(:, j), mz(:, j), Pz(:, :, j), theta);
        end
    end
end

%% Bootstrap weighing function
% Overwrites global bootstrap weighing function
function lv = calculate_incremental_weights_bootstrap(model, y, sp, ~, mzp, Pzp, theta, ~)
    if model.pym.fast
        lv = model.pym.logpdf(y, sp, mzp, Pzp);
    else
        J = size(sp, 2);
        lv = zeros(1, J);
        for j = 1:J
            lv(:, j) = model.pym.logpdf(y, sp(:, j), mzp(:, j), Pzp(:, :, j), theta);
        end
    end
end

%% 
% function lv = calculate_incremental_weights_generic(model, y, sp, s, mzp, Pzp, theta, lq)
% % todo: this is a mess, most likely doesn't work for other than the bw
% % compensation! should really be derived fully!
%     J = size(sp, 2);
%     lv = zeros(1, J);
%     for j = 1:J
%         lv(:, j) = model.pym.logpdf(y, sp(:, j), mzp(:, j), Pzp(:, :, j), theta) + model.ps.logpdf(sp(:, j), s(:, j), [], [], theta) - lq(j);
%     end
% end

%% Kalman filter prediction for mixing models
function [mzp, Pzp] = predict_kf_mixing(model, sp, s, mz, Pz, theta)
    J = size(s, 2);
    mzp = zeros(size(mz));
    Pzp = zeros(size(Pz));
    
    for j = 1:J
        sj = s(:, j);
        f = model.f(sj, theta);
        F = model.F(sj, theta);
        Q = model.Q(sj, theta);

        g = f(in);
        B = F(in, :);
%         Gs = eye(Nn);
        Qs = Q(in, in);

        f = f(il);
        A = F(il, :);
%         Gz = eye(Nl);
        Qz = Q(il, il);

        Qsz = Q(in, il);

        % KF prediction
        mzj = mz(:, j);
        Pzj = Pz(:, :, j);
        spj = sp(:, j);
        
        Mn = B*Pzj*B' + Qs;
        Ln = (A*Pzj*B' + Qsz')/Mn;
        mzp(:, j) = f + A*mzj + L*(spj - g - B*mzj);
        Pzp(:, :, j) = A*Pzj*A' + Qz - Ln*Mn*Ln';
        
if 0
        Flbar = A - Gz*Qsz'/(Gs*Qs)*B;
        Qlbar = Qz - Qsz'/Qs*Qsz;

        Nt = B*Pz(:, :, j)*B' + Gs*Qs*Gs';
        Lt = Flbar*Pz(:, :, j)*B'/Nt;
        mzp(:, j) = ( ...
            Flbar*mz(:, j) + Gz*Qsz'/(Gs*Qs)*(sp(:, j) - g) ...
            + f + Lt*(sp(:, j) - g - B*mz(:, j)) ...
        );
        Pzp(:, :, j) = ( ...
            Flbar*Pz(:, :, j)*Flbar' + Gz*Qlbar*Gz' - Lt*Nt*Lt' ...
        );
end
    end
end
