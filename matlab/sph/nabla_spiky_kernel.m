function w = nabla_spiky_kernel(r, h)
% First order derivative of spiky kernel 
%
% Copyright 2010, Kenny Erleben, DIKU.

H = repmat( h, 1, size(r,2));
w = zeros(size(r));

% TODO add your own implementation (hint see Müller ea. 03 and Lee and Han 10')
%
w(abs(r)<=H) = 45./(pi.*H(abs(r)<=H).^6).* ((H(abs(r)<=H) - abs(r(abs(r)<=H))).^2);

w(r>0) = -w(r>0)


end
