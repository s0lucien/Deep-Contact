function w = m4_kernel(r, h)
% M4 Spline kernel 
%
% Copyright 2010, Kenny Erleben, DIKU.

H = repmat( h, 1, size(r,2));
q = abs(r) ./ H;
w = zeros(size(q));

% TODO add your own implementation (hint see Monaghan 2005)
%
%  w( q <= 1)        = ????
%  w( q>1 & q <= 2 ) = ???


end
