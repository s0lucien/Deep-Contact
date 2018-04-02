function Out = clear_forces( In )
% Clear forces on all particles
%
% Copyright, 2010. Kenny Erleben, DIKU.

Out     = In;
Out.Fx  = zeros(size(Out.Fx));
Out.Fy  = zeros(size(Out.Fy));
end
