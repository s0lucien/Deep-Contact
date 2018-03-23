function [Out dt] = semi_implicit_euler(In, dt)
%
% Copyright, 2010. Kenny Erleben, DIKU.
%
Out = In;

% --- Compute semi safe time-step, inspired by CFL conditions
min_distance = min( In.h/2 );
max_speed    = max( sqrt( In.Vx.^2 + In.Vy.^2 ) + eps );
max_force    = max( sqrt( In.Fx.^2 + In.Fy.^2 ) + eps );
% Mueller et. al. 2003 used constant time step of 0.01 second
dt           = min( min_distance/max_force, min( dt, min_distance  /  max_speed ));

% --- First we do the velocity update
Out.Vx = In.Vx + dt* (In.Fx ./ In.rho); 
Out.Vy = In.Vy + dt* (In.Fy ./ In.rho);

% --- Second we do the position update
Out.X = In.X + dt*Out.Vx;
Out.Y = In.Y + dt*Out.Vy;

end
