function Out = box_projection(In)
%
% Copyright, 2010. Kenny Erleben, DIKU.
%
  Out = In;

  bounciness = 1.0; % zero corresponds to unelastic collisions and one fully elastic collisions

  % --- Handle collisions with box walls  
  Out.Vy(In.Y < In.h)           = -bounciness * In.Vy(In.Y < In.h);
  Out.Vy(In.Y > In.height-In.h) = -bounciness * In.Vy(In.Y > In.height-In.h);
  Out.Vx(In.X < In.h)           = -bounciness * In.Vx(In.X < In.h);
  Out.Vx(In.X > In.width-In.h)  = -bounciness * In.Vx(In.X > In.width-In.h);
  
  % --- Back-project onto box walls if we moved beyong box wall
  Out.X = max(In.h,In.X);
  Out.X = min(In.width-In.h,Out.X);
  Out.Y = max(In.h,In.Y);
  Out.Y = min(In.height-In.h,Out.Y);

end
