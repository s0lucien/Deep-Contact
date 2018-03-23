function Out = add_pressure_forces( In, nabla_kernel, kNN )
%
% Copyright, 2010. Kenny Erleben, DIKU.
Out     = In;

% We extract and form information into temporaries this is not efficient
% but it makes the code more readable

r     = kNN.distances(:,:);
h     = In.h;

mask = r>0;  % Used to mask out all divisions by zero and forces going to infinity

X_a   = repmat( In.X,   1, In.K );
Y_a   = repmat( In.Y,   1, In.K );
p_a   = repmat( In.p,   1, In.K );
rho_a = repmat( In.rho, 1, In.K );
m_a   = repmat( In.m,   1, In.K );

X_b   = In.X(   kNN.indices(:,:) );
Y_b   = In.Y(   kNN.indices(:,:) );
p_b   = In.p(   kNN.indices(:,:) );
rho_b = In.rho( kNN.indices(:,:) );
m_b   = In.m(   kNN.indices(:,:) );

%
% Note: according to the chain rule
%
% If K(r,h) then we have dKdxa = dKdr drdxa now let x = xa - xb and
% r = | x | then drdxa =  x / r
%
%
nabla_K_r       = nabla_kernel( r, h );
nabla_rx        = zeros(size(X_a) );
nabla_ry        = zeros(size(Y_a) );
nabla_rx(mask)  = ( X_a(mask) -  X_b(mask) ) ./ ( r(mask) ); 
nabla_ry(mask)  = ( Y_a(mask) -  Y_b(mask) ) ./ ( r(mask) ); 

% This is the pressure force as given by Monaghan 2005 page 1716 eq. 2.45
nabla_p_r       = (rho_a .* m_b).*( (p_a ./ rho_a.^2)  + (p_b ./ rho_b.^2)  ) .* nabla_K_r;
Out.Fx          = In.Fx  - sum( (nabla_p_r .* nabla_rx) , 2);
Out.Fy          = In.Fy  - sum( (nabla_p_r .* nabla_ry) , 2);
end
