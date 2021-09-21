function [shat, zhat, sys] = rbps(model, y, theta, Jf, Js, par, sys)
% # Rao-Blackwellized FFBSi particle smoother
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
    narginchk(2, 7);
    if nargin < 3 || isempty(theta)
        theta = NaN;
    end
    if nargin < 4 || isempty(Jf)
        % N.B.: If sys is provided, this is overwritten later on  by the
        % no. of particles used in the external filter.
        Jf = 250;
    end
    if nargin < 5 || isempty(Js)
        Js = 100;
    end
    if nargin < 6 || isempty(par)
        par = struct();
    end
    def = struct(...
    	'sample_backward_simulation', @sample_backward_simulation ...
    );
    par = parchk(par, def);
    
    %% Smoothing
    % If no filtered system is provided, run a bootstrap RBPF first
    if nargin < 7 || isempty(sys)
        [~, sys] = rbpf(model, y, theta, Jf);
    end
    [shat, zhat, sys] = smooth_rbffbsi(model, y, theta, Js, sys, par);
end
