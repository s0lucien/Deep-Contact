function W = kernel_wrapper( X, Y, kernel, h)
% Converts X and Y arguments into r = sqrt(x^2 + y^2) arguments.
%
% Copyright 2010, Kenny Erleben, DIKU.

[M N] = size(X);
W     = zeros(M,N);
for i =1:M,
  for j =1:N,
    W(i,j) = kernel( sqrt(X(i,j)^2 + Y(i,j)^2), h);
  end
end
end
