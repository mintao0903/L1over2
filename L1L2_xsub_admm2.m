function [x,output] = L1L2_xsub_admm2(A,b,pm,rho)
% Solves
%           min_x .5||Ax-b||^2 + lambda(|x|_1/|x|_2)
[M,N]       = size(A); 

std = pm.std;
if isfield(pm,'lambda') 
    lambda = pm.lambda; 
else
    lambda = 1e-5;  % default value
end

% maximum number of iterations
if isfield(pm,'maxit') 
    maxit = pm.maxit; 
else 
    maxit = 5*N; % default value
end
% initial guess
if isfield(pm,'x0') 
    x0 = pm.x0; 
else 
    x0 = ones(N,1);
end

if isfield(pm,'xg')
    xg = pm.xg; 
else 
    xg = x0;
end
if isfield(pm,'inneritr') 
    inneritr = pm.inneritr; 
else 
    inneritr = 5; 
end
if isfield(pm,'reltol') 
    reltol = pm.reltol; 
else 
    reltol  = 0.1*std; 
end
AAt     = A*A';
L       = chol(speye(M) + 1/rho*AAt, 'lower');
L       = sparse(L);
U       = sparse(L');   

x = x0; Atb = A'*b;
v = x0;
obj = @(x) .5*norm(A*x-b)^2 + lambda*(norm(x,1)/norm(x));
for it = 1:maxit
    start_time  = tic;
    %%y update
    rhs = Atb + rho*(x - v);
    y = rhs/rho - (A'*(U\(L\(A*rhs))))/rho^2;
    %%x update
    xold = x;
    out = x_sub_admm2(x, y + v, rho/lambda, inneritr);
    x = out.sol;
%     xfinal= max(abs(x),1e-4).*sign(x); 
    nnzx = nnz(x);
    % v updates
    v = v - (x - y);
    
    relerr      = norm(xold - x)/max([norm(xold), norm(x), eps]);
    residual    = norm(A*x - b);
%     
    output.relerr(it)   = relerr;
    output.obj(it)      = obj(x);
    output.time(it)     = toc(start_time);
    output.res(it)      = residual;
    output.err(it)      = norm(x - xg);
    output.nnz(it)     = nnzx;
    if relerr < reltol && it > 2
        break;
    end
end
output.ite_num = it;
end