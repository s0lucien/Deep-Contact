%
% Copyright 2017, Kenny Erleben, DIKU.
%
close all;
clear all;
clc;

addpath('./solvers');

%solvers = {'pgs', 'psor', 'prox_adaptive_r', 'prox_b_continuation'};
solvers = {'prox_adaptive_r'};

for i=1:length(solvers)
    
    lcp_solver_method = solvers{i};
            
    %--- Create a configuration to simulate
    config = setup_config( 150, 100, 100, 5, lcp_solver_method );
    
    %--- Create data structures for making a small movie of the simulation
    
    T         = 5.0;    % The total number of seconds that should be simulated
    fps       = 30;     % The number of frames per second that should be displayed
    
    movie_motion = VideoWriter(['movies/' lcp_solver_method '_motion.avi']);
    movie_error_plot = VideoWriter(['movies/' lcp_solver_method '_error.avi']);
    
    open(movie_motion);
    open(movie_error_plot);
    
    %--- Create a simulation loop
    while T > 0
        
        dt_wanted = min(T,1/fps);  % How much time do we want to simulate before showing next frame?
        dt_done   = 0;             % How much time has passed
        
        while( dt_done < dt_wanted )
            
            info         = collision_detection(config);
            dt           = dt_wanted - dt_done;
            [config, dt, theta]  = simulate(config,info,dt);
            dt_done      = dt_done + dt;
            
            figure(2);
            clf;
            semilogy(theta, '-r', 'LineWidth', 2 );
            grid
            xlabel('Iterations')
            ylabel('Merit value')
            title('Convergence Rate')
            cur = getframe(gcf);
            writeVideo(movie_error_plot,cur);
        end
        
        T = T - dt_wanted;
        
        figure(1);
        clf;
        hold on;
        draw_config( config );
        draw_info( config, info );
        hold off;
        axis square;
        cur = getframe(gcf);
        writeVideo(movie_motion,cur);
        
    end
    
    close(movie_motion);
    close(movie_error_plot);
    
end