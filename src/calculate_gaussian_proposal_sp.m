function [mp, Pp, my, Py, Pxy] = calculate_gaussian_proposal_sp(y, x, t, Ex, Vx, Ey_x, Vy_x)
% TODO:
%   * Rename to something appropriate (calculate_proposal_sp() or similiar)
%   * Make so that we can supply unit-sigma points (Xi, wm, wc as
%   parameters)
%   * No of iterations should be supplied as parameter
%   * document

    L = 5;
    
    Nx = size(x, 1);
    mx = Ex(x, t);
    Px = Vx(x, t);
    mux = mx;
    Sigmax = Px;
    Ny = size(y, 1);
    
    alpha = 1;
    beta = 0;
    kappa = 0;
    
    [wm, wc, c] = ut_weights(Nx, alpha, beta, kappa);
    I = length(wm);
    
    for l = 1:L
        % Generate sigma-points
        X = ut_sigmas(mux, Sigmax, c);
        
        % Transform sigma-points
        Ey = zeros(Ny, 1);
        Ey2 = zeros(Ny, Ny);
        EVy_x = zeros(Ny, Ny);
        C = zeros(Ny, Nx);
        for i = 1:I
            Y(:, i) = Ey_x(X(:, i), t);
            Ey = Ey + wm(i)*Y(:, i);
            
            Ey2 = Ey2 + wc(i)*(Y(:, i)*Y(:, i)');            
            EVy_x = EVy_x + wc(i)*Vy_x(X(:, i), t);
            C = C + wc(i)*(Y(:, i)*X(:, i)');
        end
        
if 0
        Vy = zeros(Ny, Ny);
        Vxy = zeros(Nx, Ny);
        for i = 1:I
            Vy = Vy + wc(i)*((Y(:, i) - Ey)*(Y(:, i) - Ey)');
            Vxy = Vxy + wc(i)*((X(:, i) - mux)*(Y(:, i) - Ey)');
        end
end
        
        % Calculate (co)variances moments
        Vy = Ey2 - (Ey*Ey') + EVy_x;
        Vy = (Vy + Vy')/2;
        C = C - (Ey*mux');
        

        % Calculate linearization
        Phi = C/Sigmax;
        Gamma = Ey - Phi*mux;
        Sigma = Vy - Phi*Sigmax*Phi';

        % Moments of the joint approximation - OK
        my = Phi*mx + Gamma;
        Py = Phi*Px*Phi' + Sigma;
        Pxy = Px*Phi';
        
        % Posterior of x given y - OK
        K = Pxy/Py;
        mux = mx + K*(y - my);
        Sigmax = Px - K*Py*K';
        
        if Sigmax <= 0
            def = 1;
        end
    end
    
    mp = mux;
    Pp = Sigmax;
end
