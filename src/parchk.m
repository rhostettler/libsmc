function out = parchk(in, defaults)
% Validate function parameters
%
% USAGE
%   outpar = PARCHK(inpar, defaults)
%
% DESCRIPTION
%   Validates a set of function parameters, that is, checks for missing
%   parameters, sets defaults, and complains about missing parameters.
%   Parameters are name-value pairs.
%
% PARAMETERS
%   in          Parameters to validate.
%   defaults    Default parameters.
%
% RETURNS
%   out         Validated parameters.
%
% AUTHORS
%   2017-2018 -- Roland Hostettler <roland.hostettler@aalto.fi>

    narginchk(2, 2);
    out = defaults;
    fields = fieldnames(in);
    for i = 1:length(fields)
        if isfield(defaults, fields{i})
            out.(fields{i}) = in.(fields{i});
        else
            warning('Discarding unknown parameter ''%s''.', fields{i});
        end
    end
end
