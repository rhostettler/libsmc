function [xhat, sys] = ffbsi_ps(y, t, model, Mf, Ms, par, sys)
% Forward filtering backward simulation particle smoother
% 
% SYNOPSIS
%   xhat, = FFBSI_PS(y, t, model)
%   [xhat, sys] = FFBSI_PS(y, t, model, Mf, Ms, par, sys)
%
% DESCRIPTION
%   Forward filtering backward simulation particle smoother as described in
%   [1].
%
% PARAMETERS
%   y       Ny times N matrix of measurements.
%   t       1 times N vector of timestamps.
%   model   State space model structure.
%   Mf      Number of particles for the filter (if no sys is provided, see
%           below; optional, default: 250).
%   Ms      Number of particles for the smoother (optional, default: 100).
%   par     Structure of additional (optional) parameters. May contain any
%           parameter accepted by bootstrap_pf (if no sys is provided, see
%           below) plus the following FFBSi-specific parameters:
%
%               TODO: Write these out once finalized (in particular wrt
%               rejection-sampling based version)
%
%   sys     Particle system as obtained from a forward filter. If no system
%           is provided, a bootstrap particle filter is run to generate it.
%           sys must contain the following fields:
%
%               xf  Matrix of particles for the marginal filtering density.
%               wf  Vector of particle weights for the marginal filtering
%                   density.
%
% RETURNS
%   xhat    Minimum mean squared error state estimate (calculated using the
%           smoothing density).
%   sys     Particle system (array of structs) with all the fields returned
%           by the bootstrap particle filter (or the ones in the particle
%           system provided as an input) plus the following fields:
%           
%               xs  Nx times M matrix of particles for the joint smoothing
%                   density.
%               ws  1 times M matrix of particle weights for the joint
%                   smoothing density.
%               rs  Indicator whether the corresponding particle was
%                   sampled using rejection sampling or not.
%
% SEE ALSO
%   bootstrap_pf, sisr_pf
%
% REFERENCES
%   [1] W. Fong, S. J. Godsill, A. Doucet, and M. West, "Monte Carlo 
%       smoothing with application to audio signal enhancement," IEEE 
%       Transactions on Signal Processing, vol. 50, pp. 438? 449, February 
%       2002.
%
% AUTHORS
%   2017-03-28 -- Roland Hostettler <roland.hostettler@aalto.fi>   

% TODO
%   * Implement rejection sampling w/ adaptive stopping => should go into
%     an outside function
%   * Check how I can merge that with other backward simulation smoothers,
%     e.g. ksd_ps (they use exactly the same logic in the beginning, only
%     the smooth()-function is different
%   * Clean up code for the individual backward functions.

    %% Defaults
    narginchk(3, 7);
    if nargin < 4 || isempty(Mf)
        Mf = 250;
    end
    if nargin < 5 || isempty(Ms)
        Ms = 100;
    end
    if nargin < 6
        par = [];
    end
    def = struct(...
        'rs', false ...     % Rejection sampling-based sampling
    );
    par = parchk(par, def);
    
    %% Filter
    % If no filtered system is provided, run a bootstrap PF
    if nargin < 7 || isempty(sys)
        [~, sys] = bootstrap_pf(y, t, model, Mf, par);
    end
    
    %% Backward simulation
    if nargout < 2
        xhat = smooth(y, t, model, Ms, par, sys);
    else
        [xhat, sys] = smooth(y, t, model, Ms, par, sys);
    end
end

%% Backward Recursion
function [xhat, sys] = smooth(y, t, model, Ms, par, sys)
    px = model.px;
    N = length(sys);
    [Nx, Mf] = size(sys(N).x);
    xhat = zeros(Nx, N);
    t = [0, t];

    %% Initialize
    ir = sysresample(sys(N).w);
    b = ir(randperm(Mf, Ms));
    xs = sys(N).x(:, b);
    xhat(:, N) = mean(xs, 2);
    
    return_sys = (nargout == 2);
    if return_sys
        sys(N).xs = xs;
        sys(N).ws = 1/Ms*ones(1, Ms);
        sys(N).rs = zeros(1, Ms);
    end
    
    %% Backward recursion
    for n = N-1:-1:1
        %% Sample trajectory backwards
        % TODO: We should be able to set this also through a parameter. I
        %       believe that is implemented in cpfas or cpfas_ps. Check
        %       there (i.e. rather than par.rs, have a
        %       par.sample_backward_particles()).
        if ~par.rs
            xs = sample_backward_particle(xs, sys(n).x, t(n+1), log(sys(n).w), px);
            rs = zeros(1, Ms);
        else
            [xs, rs] = sample_backward_particle_rs(xs, sys(n).x, t(n+1), log(sys(n).w), px);
        end

        %% Estimate & Store
        xhat(:, n) = mean(xs, 2);
        if return_sys
            sys(n).xs = xs;
            sys(n).ws = 1/Ms*ones(1, Ms);
            sys(n).rs = rs;
        end
    end
    
    % Strip x[0] as we don't want it in the MMSE estiamte; if needed, it
    % can be obtained from sys.
    xhat = xhat(:, 2:N);
end

%% 
function xs = sample_backward_particle(xs, x, t, lw, px)
% Compute the backward smoothing weights
% j -> trajectory to expand
% i -> candidate particles
        
    Ms = size(xs, 2);
    Mf = size(x, 2);
    for j = 1:Ms
        lv = calculate_transition_weights(xs(:, j), x, t, px);
        lwb = lw + lv;
        wb = exp(lwb-max(lwb));
        wb = wb/sum(wb);

        % Draw a new particle from the categorical distribution and
        % extend the trajectory
        ir = sysresample(wb);
        alpha = ir(randi(Mf, 1));
        xs(:, j) = x(:, alpha);
    end
end

%%
function [xs, rs] = sample_backward_particle_rs(xs, x, t, lw, px)
% TODO: It appears like there's a bug somewhere here.

    Mf = size(x, 2);
    Ms = size(xs, 2);
    L = 10;
    rs = zeros(1, Ms);
    
    % TODO: Scales O(Ms*log(Ms)) because we don't sample for all j at once,
    % but that will do for now
    ir = sysresample(exp(lw));
    for j = 1:Ms
        l = 0;
        done = 0;
        lv = zeros(1, Mf);
        iv = zeros(1, Mf);
        while ~done
            % Sample from prior
            alpha = ir(randi(Mf, 1));
            
            % Calculate non-normalized weight
            lv(alpha) = calculate_transition_weights(xs(:, j), x(:, alpha), t, px);
            iv(alpha) = 1;

            % Calculate upper bound on normalizing constant
            u = rand(1);
            paccept = (exp(lv(alpha))/px.rho);
            if paccept > 1
                warning('Acceptance probability larger than one, check your bounding constant.');
            end
            accepted = (u < paccept);

            l = l+1;
            done = accepted || (l >= L);
        end
        if ~accepted
            % Exhaustive search for the non-calculated ones
            %lv(~iv) = calculate_transition_weights(xs(:, j), x(:, ~iv), t, px);
            lv = calculate_transition_weights(xs(:, j), x, t, px);
            lv = lw + lv;
            v = exp(lv-max(lv));
            v = v/sum(v);
            tmp = sysresample(v);
            alpha = tmp(randi(Mf, 1));
        end
        xs(:, j) = x(:, alpha);
        rs(:, j) = accepted;
    end
end

%%
function lv = calculate_transition_weights(xp, x, t, px)
    M = size(x, 2);
    if px.fast
        lv = px.logpdf(xp*ones(1, M), x, t);
    else
        lv = zeros(1, M);
        for i = 1:M
           lv(i) = px.logpdf(xp, x(:, i), t);
        end
    end
end
