function w = spiky_kernel(r, h)
% SPIKY KERNEL 
%
% Copyright 2010, Kenny Erleben, DIKU.

H = repmat( h, 1, size(r,2));
w = zeros(size(r));

% TODO add your own implementation (hint see Müller ea. 03 and Lee and Han 10')
%
w(abs(r)<=H) = 15./(pi.*H(abs(r)<=H).^6).* ((H(abs(r)<=H) - abs(r(abs(r)<=H))).^3);


end
