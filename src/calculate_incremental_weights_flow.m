function lv = calculate_incremental_weights_flow(model, y, xp, x, theta, q)
% # 
% ## Usage
% * 
%
% ## Description
% 
%
% ## Input
% * 
%
% ## Outut
% * 
%
% ## Authors
% 2019-present -- Roland Hostettler

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
% * Consider renaming it to a dummy such that it can be used with others
%   (e.g., sample_gaussian() could use this as well for improved
%   efficiency)
% * In this case, we could actually integrate it into _generic and make
%   that one more 

    narginchk(6, 6);
    lv = q;
end
