%
% Copyright 2010, Kenny Erleben, DIKU.
%

close all;
clear all;

% --- Setup parameter values that control the simulation ----------------

box_width    = 3.0;           % The world box width

box_height   = 3.0;           % The world box height

N            = 25;            % Number of particles

rho          = 1000;          % Material density

r_init       = 0.05;          % Kernel support radius best guess 
                              % Observe in Müller et al they used fixed
                              % support radius of 0.045 m.Becker et al 2007
                              % used the fixed value 0.1m  
                              
gas_constant = 20;            % In experiment of Becker et. al 2007 the
                              % gas constant is simply set to a value 
                              % between 200-300. In Müller et al. 2005 they
                              % used the value 20.
                              % Kelager 06 used values in the range of
                              % 4-10.
                              
K_init       = 11;            % Initial number of nearest neighbors.
                              % Kelager used 20-40 particles in 3D.

layout_type  = 'grid';        % Initial spatial layout of particles, can be
                              % grid, random and spiral
                              
init_type    = 'consistent';  % Method to determine particle support radia,
                              % can be fixed, volume, neighbors, consistent
                              
config       = create( N, rho, gas_constant, layout_type, init_type, K_init, r_init, box_width, box_height, @poly6_kernel );


%--- Now do simulation -------------------------------------------------

T         = 0.5;     % The total number of seconds that should be simulated simulate
fps       = 30;      % The number of frames of per second that should be displayed
nb_frames = T*fps-1; % Compute the total number of frames, we do not visualize first frame
frame     = 1;       % Initialize frame counter

mov(1:nb_frames) = struct('cdata', [],'colormap', []); % Preallocate movie structure.

while T > 0

  dt_wanted = min(T,1/fps);  % How much time do we want to simulate before showing next frame?
  dt_done   = 0;             % How much time has passed 
  
  while( dt_done < dt_wanted ) 

    config       = clear_forces( config );  
    kNN          = get_nearest_neighbors( config );
    config       = compute_density( config, @poly6_kernel, kNN );
    config       = add_body_forces( config ); 
    config       = compute_pressure( config );
    config       = add_pressure_forces( config, @nabla_spiky_kernel, kNN );
    dt           = dt_wanted - dt_done;
    [config dt]  = semi_implicit_euler( config, dt );
    dt_done      = dt_done + dt;
    config       = box_projection( config );
      
  end

  T = T - dt_wanted;

  figure(1);
  clf;
  hold on;  
  draw_velocities = false;
  draw_forces     = false;
  draw_kernels    = false;
  draw_circles    = true;
  draw( config, draw_velocities, draw_forces, draw_kernels, draw_circles );  
  hold off;
  axis square;
  
  if(frame<=nb_frames)
    mov(frame) = getframe(gcf);  % Record a frame for a movie
    frame = frame + 1;
  end

end

% Create AVI file
% movie2avi(mov, 'run_sph_demo.avi', 'compression', 'None');

