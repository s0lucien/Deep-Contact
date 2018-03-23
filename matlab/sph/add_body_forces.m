function Out = add_body_forces( In )
%
% Copyright, 2010. Kenny Erleben, DIKU.

Out     = In;

% --- Body foces corresponding to Monaghan 2005 eq. 2.45 -- a Toy star potential, see page 1716
Out.Fx  = In.Fx -  ( (In.X - (In.width/2))  + 0.5*In.rho.*In.Vx );
Out.Fy  = In.Fy -  ( (In.Y - (In.height/2)) + 0.5*In.rho.*In.Vy );

end
