function [ Out ] = create( N, rho_0, gas_constant, layout_type, init_type, K_init, h_init, box_width, box_height, Kernel )
% Create a configuration data structure with all needed simulation information.
%
% Copyright 2010, Kenny Erleben, DIKU.

Vx  = zeros( N, 1); % Make initial zero velocities
Vy  = zeros( N, 1);
Fx  = zeros( N, 1); % Make initial zero forces
Fy  = zeros( N, 1);

dV  = ones( N, 1)*(4*(h_init).^2);     % Initial particle volumes, Monaghan 92 -- recommends setting initial volume equal to cell size
V   = sum(dV(:));                      % Total volume of material
m   = dV*rho_0;                        % Compute some initial mass of each particle
M   = sum(m(:));                       % Total mass
rho = rho_0 * ones( N, 1);             % Make some initial density value
p   = zeros( N, 1);                    % Make some initial pressure

% --- Compute position layout of particles
switch lower(layout_type)
  
  case 'grid'
    
    Q  = ceil(sqrt(N));         % Monaghan 92 -- recommends grid initialization
    dh = h_init*2;
    h0 = (box_width - dh*(Q-1))/2;
    h1 = h0 + (Q-1)*dh;
    
    [x, y] = meshgrid( h0:dh:h1, h0:dh:h1 );
    X   = x(1:N)';
    Y   = y(1:N)';
    
  case 'random'
    
    R = (V/pi)^(1/2);  % Determine radius in a ball of same volume as total volume of all particles.
    
    radius = rand(N,1)*R;
    theta  = rand(N,1)*(2*pi);
    
    X      = radius.*cos(theta) + box_width/2;
    Y      = radius.*sin(theta) + box_height/2;
    
  case 'spiral'
    
    R = (V/pi)^(1/2);  % Determine radius in a ball of same volume as total volume of all particles.
    
    radius = (0 : R/((N-1)) : R)';
    theta  = rand(N,1)*(2*pi);
    
    X      = radius.*cos(theta) + box_width/2;
    Y      = radius.*sin(theta) + box_height/2;
    
  otherwise
    disp('Unknown layout method')
end


%--- Now let us find a kernel support radius to work with ------------

K   = min(N,K_init);  % Setup initial number of nearest neighbors
P   = [X Y];          % Points needed for kNN searches

switch lower(init_type)
  
  case 'volume'  % Compute support radius from particle volumes
    
    %h =  ((3/(4*pi))*(m ./ rho)).^(1/3); % Sphere in 3D
    %h =  ( (1/pi)*(m ./ rho)).^(1/2);    % Circle in 2D
    h  =  1.3 * (m ./ rho).^(1/2);        % Monaghan 2005, eq. (4.2)
    K  = increaseK(h,P,K,N);
    
  case 'fixed'  % Keep support radius fixed at constant value
    
    h  = h_init * ones( N, 1);  % Set kernel support radius to initial guess
    K  = increaseK(h,P,K,N);
    
  case 'neighbors'
    
    % Keep a constant number of neighbors, that is increase h for
    % each particle to cover the given number of neighbors
    
    [~, D] = knnsearch(P,P, 'K', K);
    h      = max(D, [], 2);
    
  case 'consistent'  % Modify support radious to be self consistent with density estimate
    
    h  = h_init * ones( N, 1);  % Set kernel support radius to initial guess
    
    % Setup parameter values to control stopping criteria
    max_iter  = 10;
    tol       = 0.1;
    
    % Initialize loop variables
    update_h = true;
    n        = 1;
    
    while update_h && n < max_iter
      
      % First make sure that number of nearest neighbors are large enough
      % to cover the indivisual support radius'
      [K, IDX, D] = increaseK(h,P,K,N);
      
      % Next we try to estimate the density with the current support radius
      % value
      rho = sum( ( m( IDX(:,:) ) .* Kernel( D(:,:), h )) , 2); % Monaghan 2005, eq. (4.3)
      
      % Then we try to estimate the support radius value from the estimated
      % density
      h_est = 1.3 * (m ./ rho).^(1/2);   % Monaghan 2005, eq. (4.2)
      
      % Finally we test if the estimtaed support radius was close enough to
      % the starting value if so then we say the support radius is
      % consistent
      max_rel_h = max( abs( (h - h_est)./ h ) );
      update_h  = (max_rel_h > tol);
      if(update_h)
        h = h_est;
      end
      
      n = n + 1;
    end
    
  otherwise
    
    error('Unknown kernel initialization method')
    
end

Out = struct( 'X', X, 'Y', Y, 'Vx', Vx, 'Vy', Vy, 'Fx', Fx, 'Fy', Fy, 'm', m, 'rho', rho, 'p', p, 'width', box_width, 'height', box_height, 'h', h, 'K', K, 'rho_0', rho_0, 'V', V, 'M', M, 'N', N, 'gas_constant', gas_constant );

end


function [K IDX D] = increaseK(h,P,K,N)
% Auxiliary function used to find a kNN neighborhood that is big enough to
% cover the wanted support radius.
%
% Copyright 2010, Kenny Erleben, DIKU.
K = min(K,N);
halt = false;
while ~halt
  [IDX, D] = knnsearch(P, P, 'K', K);
  d = max(D, [], 2);  % First we determine the maximum distance in neighborhood
  if ( max(h) > ( min( d(:) ) ) && K<N ) % Next we test if the support radius is larger than the smallest largest distance
    K = K + 1;
  else
    halt = true;
  end
end
end