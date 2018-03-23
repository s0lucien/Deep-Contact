function Out = compute_density( In, Kernel, kNN )
%
% Copyright, 2010. Kenny Erleben, DIKU.
%
Out     = In;
Out.rho = sum( ( In.m( kNN.indices(:,:) ) .* Kernel( kNN.distances(:,:), In.h )) , 2);
end
