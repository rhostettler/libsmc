%% Moment Matching
% TODO: Ideally, I want to repalce this with a function supplied by 'par',
%       similar as for 'resample' (which, by the way, has to be made more
%       generic itself)
function [yhat, S, B] = calculate_moments(x, t, model)
    Q = model.Q(t);
    R = model.R(t);
    r = zeros(size(R, 1));
    [yhat, Gx, Gr] = model.g(x, r, t);
    S = Gx*Q*Gx' + Gr*R*Gr';
    B = Q*Gx';
end
