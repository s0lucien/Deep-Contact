function draw( In, draw_velocities, draw_forces, draw_kernels, draw_circles)
% DRAW
%
% Copyright, 2010, Kenny Erlben, DIKU.

[meshX, meshY, meshZ] = sphere;

V     = In.m ./ In.rho;         % Compute current volume of particle
r     = (V/pi).^(1/2);          % Assuming spherical shape compute radius

for i=1:In.N
  center_x = In.X(i);
  center_y = In.Y(i);
  center_z = 0;
  radius   = r(i);  
  color    = In.rho(i);
  
  if(draw_circles)
    draw_circle( [ center_x center_y], radius, 100, 'r-' );
  else
    surf( meshX*radius + center_x, meshY*radius + center_y, meshZ*radius*0.01 + center_z, ones(size(meshZ))*color ); % Cool 3D drawing  
  end
  
  if(draw_kernels)  
    draw_circle( [ center_x center_y], In.h(i), 100, 'b.' );
  end
  
end

%Draw bounding box.
plot( [0 In.width In.width 0 0],[0 0 In.height In.height 0], 'r-', 'LineWidth', 3 );

scale = min(In.h(:)) / max( [ sqrt(In.Vx.^2 + In.Vy.^2)'   sqrt(In.Fx.^2 + In.Fy.^2)'    ] );

if( draw_velocities )
  plot( [In.X (scale*In.Vx)+In.X ]', [In.Y  (scale*In.Vy)+In.Y]', 'b-', 'LineWidth', 2  )
end
  
if( draw_forces )
  plot( [In.X (scale*In.Fx)+In.X ]', [In.Y  (scale*In.Fy)+In.Y]', 'g-', 'LineWidth', 2  )
end

axis([0 In.width 0 In.height])

if( ~draw_circles)
  colormap cool;
  shading interp;
  %view(45,45);
  camlight headlight; 
  camproj('orthographic');
  lighting phong
  colorbar
end
end

