%
% Copyright 2010, Kenny Erleben, DIKU.
%

close all;
clear all;

% --- Setup parameter values that control the simulation ----------------

box_width    = 5.0;
box_height   = 5.0;
rho_0        = 1000;
gas_constant = 20;
K            = 11;
layout_type  = 'grid';
init_type    = 'volume';

for N=25:25:100
  for r=0.05:0.01:0.1
        
    config       = create( N, rho_0, gas_constant, layout_type, init_type, K, r, box_width, box_height, @poly6_kernel );
    
    tic;
    kNN          = get_nearest_neighbors( config );
    toc
    
    tic;
    config       = compute_density( config, @poly6_kernel, kNN );
    toc
       
    figure(1);
    clf;
    hold on;
    draw_velocities = false;
    draw_forces     = false;
    draw_kernels    = true;
    draw_circles    = true;
    draw( config, draw_velocities, draw_forces, draw_kernels, draw_circles );
    title(['Grid layout for N = ' num2str(N)  ' and h = '  num2str( min(config.h(:)) ) ] );
    hold off;
    axis square;
    
    X = (0:box_width/100:box_width)';
    Y = ones(size(X))*(box_height/2.0);
    Q = [X Y];
    h = ones(size(X))*min(config.h(:));
    
    % Compute density at all points in Q using kernel support radius from
    % config. (Hint: call knnsearch to find distances)
    
    % >> insert your code here!!! <<  Hint use knnsearch and poly6_kernel
    %
    % rho = ??
    %
    [ids, d] = knnsearch([config.X, config.Y], [X, Y], 'k', K);
    
    rho = zeros(size(ids, 1))
    for i=1:size(ids, 1)
      rho(i) = sum(config.rho(ids(i, :)) .* poly6_kernel(d(i,:), min(config.h(:)))');
    end
    
    figure(2);
    clf;
    plot(X,rho,'b-');
    hold on;
    title(['Density profile for N = ' num2str(N)  ' and h = '  num2str( min(config.h(:)) ) ]);
    xlabel('x coordinate');
    ylabel('density value');
    hold off;
    
  end
end





