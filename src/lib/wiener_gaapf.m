function [xhat, Phat, sys] = wiener_gaapf(y, t, model, M, par)
% Gaussian approximation auxiliary particle filter for Wiener systems
% 
% SYNOPSIS
%   [xhat, Phat] = wiener_gaapf(y, t, model)
%   [xhat, Phat, sys] = wiener_gaapf(y, t, model)
%
% DESCRIPTION
%   
%
% PROPERTIES
% 
%
% SEE ALSO
%
%
% VERSION
%   2017-03-28
% 
% AUTHORS
%   Roland Hostettler <roland.hostettler@aalto.fi>   

% TODO:
%   * See if we can merge that in a more generic 'approximate' APF;
%     shouldn't be too difficult
    
    %% Parameter Checks & Defaults

    %% Initialization
    [~, py, px0] = modelchk(model);
    x = px0.rand(M);
    lw = log(1/M)*ones(1, M);
    
    if nargout == 3
        sys.x0 = x;
    end
    
    %% Preallocation
    Nx = size(x, 1);
    [Ny, N] = size(y);
    xhat = zeros(Nx, N);
    Phat = zeros(Nx, Nx, N);
    if nargout == 3
        sys.x = zeros(Nx, M, N);
        sys.w = zeros(1, M, N);
        sys.lw = zeros(1, M, N);
    end

    % TODO: Check which ones are needed and which ones not.
    xn = zeros(Nx, M);
    xhat_p = zeros(Nx, M);
    C = zeros(Nx, Nx, M);
    yhat_p = zeros(Ny, M);
    B = zeros(Nx, Ny, M);
    S = zeros(Ny, Ny, M);
    lv = zeros(1, M);
    
    %% Filtering
    for n = 1:N
        %% Preparations & quick access
        % TODO: Not sure if we handle the model like this. To be seen.
        F = model.F(t(n));
        Q = model.Q(t(n));

        %% Calculate the importance distribution's moments
        % TODO: What to do about this?
        for m = 1:M
            xp = F*x(:, m);
            [yhat_p(:, m), S(:, :, m), B(:, :, m)] = calculate_moments(xp, t(n), model);
            K = B(:, :, m)/S(:, :, m);
            xhat_p(:, m) = xp + K*(y(:, n) - yhat_p(:, m));
            C(:, :, m) = Q - K*S(:, :, m)*K';
            lv(m) = lw(m) + logmvnpdf(y(:, n).', yhat_p(:, m).', S(:, :, m)).';
        end
        v = exp(lv-max(lv));
        v = v/sum(v);

        %% Draw new particles
        alpha = sysresample(v);
        for m = 1:M
            xn(:, m) = xhat_p(:, alpha(m)) ...
                + chol(C(:, :, alpha(m))).'*randn(Nx, 1);
            K = B(:, :, alpha(m))'/Q;
            mu = yhat_p(:, alpha(m)) + K*(xn(:, m) - F*x(:, alpha(m)));
            Sigma = S(:, :, alpha(m)) - K*Q*K';
            lw(:, m) = py.logpdf(y(:, n), xn(:, m), t(n)) - logmvnpdf(y(:, n), mu, Sigma);
        end
        x = xn;
        w = exp(lw-max(lw));
        w = w/sum(w);
        lw = log(w);

        %% Estimate & store results
        xhat(:, n) = xn*w';
        % TODO: Estimate covariance, etc.
        
        %% Store
        if nargout == 3
        	sys.x(:, :, n) = x;
            sys.w(:, :, n) = w;
            sys.lw(:, :, n) = lw;
        end
    end
end

%% Systematic Resampling Algorithm
% TODO: Merge with 'global' resampling function
% function ri = resample(self, w)
%     M = length(w);
%     ri = zeros(1, M);
%     i = 0;
%     u = 1/M*rand();
%     for j = 1:M
%         Ns = floor(M*(w(j)-u)) + 1;
%         if Ns > 0
%             ri(i+1:i+Ns) = j;
%             i = i + Ns;
%         end
%         u = u + Ns/M - w(j);
%     end
%     ri = ri(randperm(M));
% end
