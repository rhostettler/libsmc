function alpha = resample_stratified(w)
% # Stratified resampling
% ## Usage
% * `alpha = resample_stratified(w)`
%
% ## Description
% Stratified resampling, returns randomized indices `alpha` such that
% Pr(alpha) = w(alpha).
%
% ## Input
% * `w`: Probabilities.
%
% ## Output
% * `alpha`: The resampled indices.
%
% ## Authors
% 2017-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

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
% * Get rid of the outer loop

    narginchk(1, 1);
    [I, J] = size(w);
    w = w./(sum(w, 2)*ones(1, J));
    alpha = zeros(I, J);
    for i = 1:I
        k = 0;
        u = 1/J*rand();
        for j = 1:J
            % Get the no. of times we need to replicate this sample; if N = 0,
            % the following lines will just ignore it.
            N = floor(J*(w(i, j)-u)) + 1;
    %        if N > 0
                alpha(i, k+1:k+N) = j*ones(1, N);
                k = k + N;
    %        end
            u = u + N/J - w(i, j);
        end
        alpha(i, :) = alpha(i, randperm(J));
    end
end
