function [xs, ys] = simulate_model(model, theta, N)

    narginchk(1, 3);
    if nargin < 2 || isempty(theta)
        theta = NaN;
    end
    if nargin < 3 || isempty(N)
        N = 100;
    end
    
    % TODO: expand theta
    
%     theta = NaN*ones(1, N);

    %% Generate data
    ys = zeros(1, N);
    x = model.px0.rand(1);
    
    dx = size(x, 1);
    xs = zeros(dx, N);
    
    for n = 1:N
        x = model.px.rand(x, theta(n));
        y = model.py.rand(x, theta(n));
        xs(:, n) = x;
        ys(:, n) = y;
    end
end
