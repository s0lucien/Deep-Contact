function [lambda, theta] = solve_lcp(A, b, lambda, lcp_solver_method)

if nargin<4 
    lcp_solver_method = 'psor';
end

if strcmp(lcp_solver_method,'pgs')
    [lambda, theta] = pgs(A,b,lambda);
elseif strcmp(lcp_solver_method,'psor')
    [lambda, theta] = psor(A,b,lambda);
elseif strcmp(lcp_solver_method,'prox_adaptive_r')
    [lambda, theta] = prox_adaptive_r(A,b,lambda);
elseif strcmp(lcp_solver_method,'prox_b_continuation')
    [lambda, theta] = prox_b_continuation(A,b,lambda);
else
    error(['solve_lcp: unknown method = ' lcp_solver_method])
end

end
