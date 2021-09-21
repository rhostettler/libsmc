function [shat, zhat, sys] = smooth_rbffbsi(model, y, theta, Js, sys, par)
% # Rao-Blackwellized FFBSi particle smoothing
% ## Usage
% * `shat = smooth_rbffbsi(model, y, theta, Js, sys)`
% * `[shat, zhat, sys] = smooth_rbffbsi(model, y, theta, Js, sys, par)`
%
% ## Description
% Rao-Blackwellized forward-filtering backward-simulation (FFBSi) particle 
% smoother as described in [1]. This smoothing algorithm can be used with
% the standard `ps()` smoothing frontend.
%
% IMPORTANT! Currently, smoothing is only implemented for the hierarchical
% model, see TODOs throughout the code.
%
% ## Input
% * `model`: State-space model struct.
% * `y`: dy-times-N matrix of measurements.
% * `theta`: Additional parameters.
% * `Js`: No. of smoothing particles.
% * `sys`: Particle system array of structs from a Rao-Blackwellized
%   particle filter.
% * `par`: Algorithm parameter struct, may contain the following fields:
%     - `[lambda, Omega] = predict_bf(model, ss, lambda_hat, Omega_hat, theta)`:
%       Backward filter prediction function (default:
%       `@predict_bf_hierarchical`).
%     - `[beta, state] = sample_backward_simulation(model, model, ss, s, lw, mz, Pz, lambda, Omega, theta)`:
%       Function to sample from the backward smoothing kernel (default:
%       `@sample_backward_simulation`).
%
% ## Output
% * `shat`: ds-times-N matrix of smoothed state estimates (MMSE) for the
%   nonlinear states.
% * `zhat`: dz-times-N matrix of smoothed state estimates (MMSE) for the
%   conditionally linear states.
% * `sys`: Particle system array of structs for the smoothed particle
%   system. The following fields are added by `smooth_ffbsi`:
%     - `xs`: Smoothed particles (nonlinear states `s`).
%     - `ws`: Smoothed particle weights (`1/Js` for FFBSi).
%     - `state`: State of the backward simulation sampler.
%     - `lambda`, `Omega`: Backward filter statistics.
% 
% ## References
% 1. F. Lindsten, P. Bunch, S. Särkkä, T. B. Schön, and S. J. Godsill, 
%    “Rao–Blackwellized particle smoothers for conditionally linear 
%    Gaussian models,” IEEE Journal of Selected Topics in Signal 
%    Processing, vol. 10, no. 2, pp. 353–365, March 2016.
%
% ## Authors
% 2021-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

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
% * Implement mixing model
% * Implement testcase

    %% Defaults
    narginchk(5, 6);
    if nargin < 6 || isempty(par)
        par = struct();
    end
    def = struct(...
        'predict_bf', @predict_bf_hierarchical, ...
    	'sample_backward_simulation', @sample_backward_simulation ...
    );
    par = parchk(par, def);
    
    %% Preallocate
    % Get dimensions
    [dy, N] = size(y);
    [ds, Jf] = size(sys(N).x);
    
    % Preallocate
    shat = zeros(ds, N);
    
    % Prepend NaN for zero measurement and parameter
    % N.B.: theta is expanded in 'ps' to the same length as y (but
    % excluding the non-existing measurement n = 0).
    y = [NaN*ones(dy, 1), y];
    theta = [NaN*size(theta, 1), theta];
    N = N+1;

    %% Initialize
    % Draw particles
    ir = resample_stratified(sys(N).w);
    beta = ir(randperm(Jf, Js));
    ss = sys(N).x(:, beta);
    shat(:, N) = mean(ss, 2);
    
    % Initialize backward filter
    [lambda_hat, Omega_hat] = initialize_bf(model, y(:, N), ss, sys(N), theta(:, N));

    % Store
    sys(N).xs = ss;
    sys(N).ws = 1/Js*ones(1, Js);
    sys(N).state = [];
    sys(N).lambda = [];
    sys(N).Omega = [];
    
    %% Backward recursion
    for n = N-1:-1:1
        %% Sample
        % Backward filter prediction
        % TODO: For the mixing model, we have to do a backward prediction
        % for each of the filter particles. This will require some changes
        % in the interface here and how we handle lambda and Omega.
        [lambda, Omega] = par.predict_bf(model, ss, lambda_hat, Omega_hat, theta(:, n));
        
        % Sampling
        [beta, state] = par.sample_backward_simulation(model, ss, sys(n).x, log(sys(n).w), sys(n).mz, sys(n).Pz, lambda, Omega, theta(:, n));
        ss = sys(n).x(:, beta);
        shat(:, n) = mean(ss, 2);
        
        % TODO: For mixing models, we have to update lambda and Omega here.
        
        % Backward filter update
        [lambda_hat, Omega_hat] = update_bf(model, y(:, n), ss, lambda, Omega, theta(:, n));
        
        %% Store
        sys(n).xs = ss;
        sys(n).ws = 1/Js*ones(1, Js);
        sys(n).state = state;
        sys(n).lambda = lambda;
        sys(n).Omega = Omega;
    end
    
    %% Recalculate the linear states
    % Only do so if 
    if nargout >= 2
        [zhat, sys] = smooth_rts(model, y, theta, sys);
    end
    
    %% Post-processing
    shat = shat(:, 2:N);
    zhat = zhat(:, 2:N);
end

%% Backward filter initialization
function [lambda_hat, Omega_hat] = initialize_bf(model, y, ss, sys, theta)
    narginchk(5, 5);
    Js = size(ss, 2);
    dz = size(sys.mz, 1);
    lambda_hat = zeros(dz, Js);
    Omega_hat = zeros(dz, dz, Js);
    
    for j = 1:Js
        hj = model.py.h(ss(:, j), theta);
        Hj = model.py.H(ss(:, j), theta);
        Rj = model.py.R(ss(:, j), theta);
        Lj = Hj'/Rj;
        lambda_hat(:, j) = Lj*(y - hj);
        Omega_tmp = Lj*Hj;
        Omega_hat(:, :, j) = (Omega_tmp + Omega_tmp')/2;
    end
end

%% Backward filter prediction
% Notation:
% Paper -> libsmc
% f     -> pz.g
% A     -> pz.G
% F     -> chol(pz.Q).'
function [lambda, Omega] = predict_bf_hierarchical(model, ss, lambda_hat, Omega_hat, theta)
    narginchk(5, 5);
    [dz, Js] = size(lambda_hat);
    lambda = zeros(dz, Js);
    Omega = zeros(dz, dz, Js);
    
    for j = 1:Js
        gj = model.pz.g(ss(:, j), theta);
        Gj = model.pz.G(ss(:, j), theta);
        Fj = chol(model.pz.Q(ss(:, j), theta)).';
        mj = lambda_hat(:, j) - Omega_hat(:, :, j)*gj;
        Mj = Fj'*Omega_hat(:, :, j)*Fj + eye(dz);
        Lj = Gj'*(eye(dz) - Omega_hat(:, :, j)*Fj/Mj*Fj');
        lambda(:, j) = Lj*mj;
        Omega_tmp = Lj*Omega_hat(:, :, j)*Gj;
        Omega(:, :, j) = (Omega_tmp + Omega_tmp')/2;
    end
end

%% Backward filter update
function [lambda_hat, Omega_hat] = update_bf(model, y, ss, lambda, Omega, theta)
    narginchk(6, 6);
    [dz, Js] = size(lambda);
    lambda_hat = zeros(dz, Js);
    Omega_hat = zeros(dz, dz, Js);

    for j = 1:Js
        hj = model.py.h(ss(:, j), theta);
        Hj = model.py.H(ss(:, j), theta);
        Rj = model.py.R(ss(:, j), theta);
        Lj = Hj'/Rj;
        lambda_hat(:, j) = lambda(:, j) + Lj*(y - hj);
        Omega_tmp = Omega(:, :, j) + Lj*Hj;
        Omega_hat(:, :, j) = (Omega_tmp + Omega_tmp')/2;
    end
end

%% Backward simulation sampling
function [beta, state] = sample_backward_simulation(model, ss, s, lw, mz, Pz, lambda, Omega, theta)
    narginchk(9, 9);
    state = [];
    Js = size(ss, 2);
    [dz, Jf] = size(mz);
    lv = zeros(1, Jf);
    beta = zeros(1, Js);
        
    for j = 1:Js
        %% Calculate backward weights
        % According to (21)
        for i = 1:Jf
            % log(Z[n])
            lZ = model.ps.logpdf(ss(:, j), s(:, i), theta);

            % Lambda[n], (21a)
            % TODO: Numerically stable implementation
            mzi = mz(:, i);
            Gammai = chol(Pz(:, :, i)).';
            Lambda = Gammai'*Omega(:, :, j)*Gammai + eye(dz);
            Lambda = (Lambda + Lambda')/2;
            
            % eta[n], (21b)
            % TODO: Numerically stable implementation
            eta = mzi'*Omega(:, :, j)*mzi - 2*lambda(:, j)'*mzi ...
                - (Gammai'*(lambda(:, j) - Omega(:, :, j)*mzi))'/Lambda*(Gammai'*(lambda(:, j) - Omega(:, :, j)*mzi));
    
            % Weight
            lv(:, i) = lw(i) + lZ - 1/2*log(det(Lambda)) - 1/2*eta;
        end

        %% Sample
        vs = exp(lv-max(lv));
        vs = vs/sum(vs);
        ir = resample_stratified(vs);
        beta(j) = ir(randi(Jf, 1));
    end
end

%% Recalculate the linear states
function [zhat, sys] = smooth_rts(model, y, theta, sys)
    %% Initialize
    N = size(y, 2);
    s = sys(1).xs;
    Js = size(s, 2);
    dz = size(sys(1).mz, 1);
    Iz = eye(dz);
    [mz, Pz] = initialize_kf(model, s);
    zhat = zeros(dz, N);
    zhat(:, 1) = mean(mz, 2);
    sys(1).mz = mz;
    sys(1).Pz = Pz;

    %% Filter
    % TODO: This is specific for the hierarchical model as is and should be
    % revised so that we can use it for both models (possibly only requires
    % replacing predict_kf_hierarchical (update_kf should be fine).
    for n = 2:N
        % Prediction
        sp = sys(n).xs;
        [mzp, Pzp] = predict_kf_hierarchical(model, sp, 1:Js, s, mz, Pz, theta(:, n));
        
        % Measurement update
        s = sp;
        [mz, Pz] = update_kf(model, y(:, n), s, mzp, Pzp, theta(:, n));
        
        % Store
        sys(n).mz = mz;
        sys(n).Pz = Pz;
    end
    
    %% Smoother
    % Note: This is generic and the same code is used for both the
    % hierarchical and the mixing model.
    mzs = mz;
    Pzs = Pz;
    zhat(:, N) = mean(mzs, 2);
    sys(N).mzs = mzs;
    sys(N).Pzs = Pzs;
    for n = N-1:-1:1
        for j = 1:Js
            Pzs(:, :, j) = (sys(n).Pz(:, :, j)\Iz + sys(n).Omega(:, :, j))\Iz;
            mzs(:, j) = Pzs(:, :, j)*(sys(n).Pz(:, :, j)\sys(n).mz(:, j) + sys(n).lambda(:, j));
        end        
        zhat(:, n) = mean(mzs, 2);
        sys(n).mzs = mzs;
        sys(n).Pzs = Pzs;
    end
end
