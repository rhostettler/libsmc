function par = parchk(par, def)
% Check function parameters and set defaults
%
% SYNOPSIS
%   par = parchk(par, def)
%
% DESCRIPTION
%   
%
% PARAMETERS
%   par
%
%   def
%
% VERSION
%   2017-03-23
%
% AUTHORS
%   Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * Include the possibility of specifying other attributes such as range
%     of the parameters.

    narginchk(2, 2);
    if isempty(par)
        % If no options are set, use the default ones...
        par = def;
    else
        % ...otherwise, copy the missing ones
        fields = fieldnames(def);
        for i = 1:length(fields)
            if ~isfield(par, fields{i})
                par.(fields{i}) = def.(fields{i});
            end
        end
    end
end
