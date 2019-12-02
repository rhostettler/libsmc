function [shat, zhat, sys] = rbps(model, y, theta, Jf, Js, par, sys)
% # Rao--Blackwellized FFBSi particle smoother
% ## Usage
% * `shat = rbps(model, y, theta, Jf, Js, par, sys)`
% 
%
% ## Description
% 
% 
% 


% TODO:
% * Only hierarchical model implemented right now
% * No testcase implemented
% * make interface such that we can run a filter within the smoother and
%   other defualts
% * It would seem like we can embedd this into the generic ps function; the
%   only significant differences are the output, everything else can be
%   controlled with par. Using varargout, we can do this in ps, I think.
%   (=> move to smooth_rbffbsi; this would also allow for parallelized
%   variants, etc.). Maybe keep rbps as a frontend
% * Put the acutal smoothing function into smooth_mixing /
%   smooth_hierarchical functions
% * Enable rejection sampling-based backward simulation
% * Documentation

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

    %% Defaults
    % TODO: Implement proper defaults
    narginchk(7, 7);
    if nargin < 3 || isempty(theta)
        theta = NaN;
    end

    % Expand data dimensions, prepend NaN for zero measurement
    [dy, N] = size(y);
    y = [NaN*ones(dy, 1), y];
    
    % Expand 'theta'
    if size(theta, 2) == 1
        N = size(y, 2);
        theta = theta*ones(1, N);
    end

    %% Preallocate
    N = length(sys);
    [ds, Jf] = size(sys(N).x);
    dz = size(sys(N).mz, 1);
    
    shat = zeros(ds, N);
    lZ = zeros(1, Jf);
    lv = zeros(1, Jf);
    lambda_hat = zeros(dz, Js);
    Omega_hat = zeros(dz, dz, Js);
    
    zhat = zeros(dz, N);
    Iz = eye(dz);

    theta = [NaN*size(theta, 1), theta];

    %% Initialize
    % Draw particles
    ir = resample_stratified(sys(N).w);
    beta = ir(randperm(Jf, Js));
    ss = sys(N).x(:, beta);
    shat(:, N) = mean(ss, 2);
    
    % Initialize sufficient statistics of the backward information filter
    for j = 1:Js
        hj = model.py.h(ss(:, j), theta(:, N));
        Hj = model.py.H(ss(:, j), theta(:, N));
        Rj = model.py.R(ss(:, j), theta(:, N));
        lambda_hat(:, j) = Hj'/Rj*(y(:, N) - hj);
        Omega_hat(:, :, j) = Hj'/Rj*Hj;
    end

    % Store particle system
    sys(N).xs = ss;
    sys(N).ws = 1/Js*ones(1, Js);
    sys(N).state = [];
    sys(N).lambda = [];
    sys(N).Omega = [];
    
    %% Backward recursion
    for n = N-1:-1:1
        % Get filter samples
        s = sys(n).x;
        
        for j = 1:Js
            %% Calculate ancestor weights
            % (Hierarchical model only for now)
            % TODO: Check for model.ps.fast
            for i = 1:Jf
                lZ(:, i) = model.ps.logpdf(ss(:, j), s(:, i), theta(:, n));
            end
            
            % paper -> libsmc
            % f -> pz.g
            % A -> pz.G
            % F -> chol(pz.Q).'
            gj = model.pz.g(ss(:, j), theta(:, n));
            Gj = model.pz.G(ss(:, j), theta(:, n));
            Fj = chol(model.pz.Q(ss(:, j), theta(:, n))).';
%             Fj = sqrtm(model.pz.Q(ss(:, j), theta(:, n)));
            mj = lambda_hat(:, j) - Omega_hat(:, :, j)*gj;
            Mj = Fj'*Omega_hat(:, :, j)*Fj + Iz;
            Lj = Gj'*(Iz - Omega_hat(:, :, j)*Fj/Mj*Fj');
            lambda = Lj*mj;
            Omega = Lj*Omega_hat(:, :, j)*Gj;
            
            % Lambda^i, eta^i according to (21)
            for i = 1:Jf
                mzi = sys(n).mz(:, i);
                Pzi = sys(n).Pz(:, :, i);
                Gammai = chol(Pzi).';
                Lambda = Gammai'*Omega*Gammai + Iz;
                eta = wnorm(mzi, Omega) - 2*lambda'*mzi ...
                    - wnorm(Gammai'*(lambda - Omega*mzi), Lambda\Iz);
                lv(:, i) = -1/2*log(det(Lambda)) - 1/2*eta;
            end
                        
            %% Sample
            lws = log(sys(n).w) + lZ + lv;
            ws = exp(lws-max(lws));
            ws = ws/sum(ws);
            ir = resample_stratified(ws);
            beta = ir(randi(Jf, 1));
            ss(:, j) = s(:, beta);
            
            %% Information filter measurement update
            hj = model.py.h(ss(:, j), theta(:, n));
            Hj = model.py.H(ss(:, j), theta(:, n));
            Rj = model.py.R(ss(:, j), theta(:, n));
            lambda_hat(:, j) = lambda + Hj'/Rj*(y(:, n) - hj);
            Omega_hat(:, :, j) = Omega + Hj'/Rj*Hj;
            
            %% Store
            sys(n).lambda(:, j) = lambda;
            sys(n).Omega(:, :, j) = Omega;
        end
        
        % Point estimate
        shat(:, n) = mean(ss, 2);
        
        %% Store
        sys(n).xs = ss;
        sys(n).ws = 1/Js*ones(1, Js);
        sys(n).state = [];
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
        [mzp, Pzp] = predict_kf_hierarchical(model, sp, s, mz, Pz, theta(:, n));
        
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
