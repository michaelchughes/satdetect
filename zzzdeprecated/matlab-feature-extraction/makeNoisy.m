function [Im] = makeNoisy(Im, sigma)

if ~exist('sigma', 'var')
    sigma = 0.005;
end
Im = Im + 0.05 * randn(size(Im));
Im = Im - min(min(Im));
Im = Im / max(max(Im));
