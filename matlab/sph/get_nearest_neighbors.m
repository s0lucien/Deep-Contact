function [kNN] = get_nearest_neighbors( In )
% GET NEAREST NEIGHBORS
%
% Copyright, 2010. Kenny Erleben, DIKU.

K       = min(In.K,In.N);
P       = [In.X In.Y];
[IDX D] = knnsearch(P, P, 'K', K);

%--- Technically corret we should increase/dscrease K in this routine to
%--- make sure it is consistent with the support radii. However, it is 
%--- computationally expensive to do so, so we hope for the best! 

% Assemble kNN info
kNN = struct( 'indices', IDX, 'distances', D );

end
