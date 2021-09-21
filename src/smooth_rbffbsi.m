function [shat, zhat, sys] = smooth_rbffbsi(model, y, theta, Js, sys, par)

% TODO:
% * Documentation
% * Copyright notice
% * Only hierarchical model implemented right now
% * No testcase implemented
% * Smoothing of linear states can be done at once, i think

    %% Defaults
    narginchk(5, 6);
    if nargin < 6 || isempty(par)
        par = struct();
    end
    def = struct(...
    	'sample_backward_simulation', @sample_backward_simulation ...
    );
    par = parchk(par, def);
    
    %% Preallocate
    [dy, N] = size(y);
    [ds, Jf] = size(sys(N).x);
    dz = size(sys(N).mz, 1);
    
    shat = zeros(ds, N);
    lambda_hat = zeros(dz, Js);
    Omega_hat = zeros(dz, dz, Js);    
    zhat = zeros(dz, N);
    
    Idz = eye(dz);
    
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
%         %% Sample
%         % Backward information filter prediction
%         [lambda, Omega] = par.predict_bif(model, ss, lambda_hat, Omega_hat);
%         
        
        for j = 1:Js
            %% Sample backward trajectory
            % Calculate sufficient statistics
            % Notation:
            % Paper -> libsmc
            % f     -> pz.g
            % A     -> pz.G
            % F     -> chol(pz.Q).'
            gj = model.pz.g(ss(:, j), theta(:, n));
            Gj = model.pz.G(ss(:, j), theta(:, n));
            Fj = chol(model.pz.Q(ss(:, j), theta(:, n))).';
            mj = lambda_hat(:, j) - Omega_hat(:, :, j)*gj;
            Mj = Fj'*Omega_hat(:, :, j)*Fj + Idz;
            Lj = Gj'*(Idz - Omega_hat(:, :, j)*Fj/Mj*Fj');
            lambda = Lj*mj;
            Omega = Lj*Omega_hat(:, :, j)*Gj;
            Omega = (Omega + Omega')/2;
            
            % Sample
            [beta, state] = par.sample_backward_simulation(model, ss(:, j), sys(n).x, log(sys(n).w), sys(n).mz, sys(n).Pz, lambda, Omega, theta(n));
            ss(:, j) = sys(n).x(:, beta);
            
            %% Information filter measurement update
            hj = model.py.h(ss(:, j), theta(:, n));
            Hj = model.py.H(ss(:, j), theta(:, n));
            Rj = model.py.R(ss(:, j), theta(:, n));
            lambda_hat(:, j) = lambda + Hj'/Rj*(y(:, n) - hj);
            Omega_hat_tmp = Omega + Hj'/Rj*Hj;
            Omega_hat(:, :, j) = (Omega_hat_tmp + Omega_hat_tmp')/2;
            
            %% Store
            sys(n).lambda(:, j) = lambda;
            sys(n).Omega(:, :, j) = Omega;
        end
        
        % Point estimate
        shat(:, n) = mean(ss, 2);
        
        %% Store
        sys(n).xs = ss;
        sys(n).ws = 1/Js*ones(1, Js);
        sys(n).state = state;
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

%% Sampling function
function [beta, state] = sample_backward_simulation(model, ss, s, lw, mz, Pz, lambda, Omega, theta)
    [dz, Jf] = size(mz);
    lZ = zeros(1, Jf);
    lv = zeros(1, Jf);
    Iz = eye(dz);
    state = [];
    
    %% Calculate ancestor weights
    % Lambda^i, eta^i according to (21)
    for i = 1:Jf
        % TODO: These statistics are calculated in three places; should put them into one function.
        % Calculate normalizing constant
        lZ(:, i) = model.ps.logpdf(ss, s(:, i), theta);
        
        % Other statistics
        mzi = mz(:, i);
%           Pzi = sys(n).Pz(:, :, i);
%           Gammai = chol(Pzi).';
        Gammai = chol(Pz(:, :, i));
        Lambda = Gammai'*Omega*Gammai + Iz;
        eta = mzi'*Omega*mzi - 2*lambda'*mzi ...
            - (Gammai'*(lambda - Omega*mzi))'/Lambda*(Gammai'*(lambda - Omega*mzi));
%           eta = wnorm(mzi, Omega) - 2*lambda'*mzi ...
%              - wnorm(Gammai'*(lambda - Omega*mzi), Lambda\Iz);

        % Ancestor weight
        lv(:, i) = -1/2*log(det(Lambda)) - 1/2*eta;
    end
                        
    %% Sample
    lws = lw + lZ + lv;
    ws = exp(lws-max(lws));
    ws = ws/sum(ws);
    ir = resample_stratified(ws);
    beta = ir(randi(Jf, 1));
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
