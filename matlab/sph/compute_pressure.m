function Out = compute_pressure( In )
%
% Copyright, 2010. Kenny Erleben, DIKU.
Out     = In;

% Implement an equation of state giving the pressure value
%
% This is inspired by Müller et al. 2003. It is a spring like model that
% penalizes deviations from the ideal gas law. p V = n R T or equivalent
% form  p = rho (R/M) T

Out.p = In.gas_constant * (In.rho - In.rho_0);

% According to Desbrun et al 96 the gas constant acts as a spring
% coefficient and controls the stiffness behaviour of the material.
% Observe that if truely stiff materials is wanted this implies
% high values of the gas constant resulting in smaller time-steps to
% ensure stability when doing numerical integration. 

end
