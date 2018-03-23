function w = gaussian_kernel(r, h)
% GAUSSIAN_KERNEL
%
% Copyright 2010, Kenny Erleben, DIKU.

H = repmat( h, 1, size(r,2));
K = 1./(pi.*H.^2);
w = K .* exp( - r.^2 ./ H.^2 );

end
