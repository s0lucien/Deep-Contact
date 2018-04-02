function w = nabla_gaussian_kernel(r, h)
% NABLA_GAUSSIAN_KERNEL 
%
% Copyright 2010, Kenny Erleben, DIKU.

G = gaussian_kernel(r, h);
H = repmat( h, 1, size(r,2));
w = G .* (- 2*r ./ H.^2);

end
