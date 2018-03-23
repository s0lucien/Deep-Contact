function w = poly6_kernel(r, h)
% Poly 6 kernel 
%
% Copyright 2010, Kenny Erleben, DIKU.

H = repmat( h, 1, size(r,2));
w = zeros(size(r));

w(abs(r)<=H) = 315./(64.*pi.*H(abs(r)<=H).^9).* ((H(abs(r)<=H).^2 - r(abs(r)<=H).^2).^3);

end
