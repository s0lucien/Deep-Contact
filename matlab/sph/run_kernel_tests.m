%
% Copyright 2010, Kenny Erleben, DIKU.
%

close all;
clear all;

h = 1;   %--- Here we simply pick a kernel support radius, its a dummy value any positive value will do.

%--- Test by numerical integration if all kernels are normalized 
fun = @(X,Y) kernel_wrapper( X, Y, @gaussian_kernel, h);
Q = quad2d(fun, -3*h, 3*h, -3*h, 3*h)
fun = @(X,Y) kernel_wrapper( X, Y, @m4_kernel, h);
Q = quad2d(fun, -h, h, -h, h)
fun = @(X,Y) kernel_wrapper( X, Y, @poly6_kernel, h);
Q = quad2d(fun, -h, h, -h, h)
fun = @(X,Y) kernel_wrapper( X, Y, @spiky_kernel, h);
Q = quad2d(fun, -h, h, -h, h)

%--- Next we wish to plot the kernels so we can inspect their shapes and
%--- verify whether they are symmetrical and have finite support radius of h
dh = h/100;
r = -3*h : dh: 3*h;

G     = gaussian_kernel(r,h);
M4    = m4_kernel(r,h);
P6    = poly6_kernel(r,h);
spiky = spiky_kernel(r,h);

figure(2);
clf;
hold on;
plot(r,G,'r-');
% plot(r,M4,'g-');
plot(r,P6,'b-');
plot(r,spiky,'m-');
title('Values');
xlabel('r');
ylabel('W(r)');
legend('Gaussian', 'Poly6', 'Spiky');
axis tight;
hold off;

%--- Next we wish to have a look at the kernel derivatives and visually
%--- inspect if their shape match the slopes of the corresponding kernel functions.
nabla_G     = nabla_gaussian_kernel(r,h);
% nabla_m4    = nabla_m4_kernel(r,h);
nabla_p6    = nabla_poly6_kernel(r,h);
nabla_spiky = nabla_spiky_kernel(r,h);

figure(3);
clf;
hold on;
plot(r,nabla_G,'r-');
% plot(r,nabla_m4,'g-');
plot(r,nabla_p6,'b-');
plot(r,nabla_spiky,'m-');
title('Derivatives');
xlabel('r');
ylabel('\nabla_r W(r)');
legend('Gaussian', 'Poly6', 'Spiky');
axis tight;
hold off;
