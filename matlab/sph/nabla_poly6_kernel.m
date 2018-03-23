function w = nabla_poly6_kernel(r, h)
% First order derivative of Poly 6 kernel 
%
% Copyright 2010, Kenny Erleben, DIKU.

H = repmat( h, 1, size(r,2));
w = zeros(size(r));

w(abs(r)<=H) = -945./(32.*pi.*H(abs(r)<=H).^9).* r(abs(r)<=H).*((H(abs(r)<=H).^2 - r(abs(r)<=H).^2).^2);

end
