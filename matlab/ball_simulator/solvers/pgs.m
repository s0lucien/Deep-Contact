function [lambda, theta] = pgs(A, b, lambda)

N = length(b);
K = 100;
theta = zeros(K,1);
gamma = 1.0;
r = gamma ./ diag(A);    % Initial r-factors

k = 1;
while k <= K
    
    lambda_old = lambda;
        
    for i=1:N         
        res = A(i,:)*lambda + b(i);
        lambda(i) = max(0, lambda(i) - r(i) * res );
    end
    
    delta = lambda - lambda_old;
    theta(k) = max(abs(delta(i)));
    
    k = k +1; 
end

end
