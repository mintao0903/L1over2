function [x, output] = CS_L1_uncon_ADMM(A, b, pm)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%         min_x .5||Ax-b||^2 + lambda |x|_1             %%%  
%%%                                                                     %%%
%%% Input: dictionary A, data b, parameters set pm                      %%%
%%%        pm.lambda: regularization paramter                           %%%
%%%        pm.delta: penalty parameter for ADMM                         %%%
%%%        pm.maxit: max iterations                                     %%%
%%%        pm.reltol: rel tolerance for ADMM: default value: 1e-6       %%%
%%%        pm.alpha: alpha in the regularization                        %%%
%%% Output: computed coefficients x                                     %%%
%%%        written by Min Tao
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[M,N]       = size(A); 
start_time  = tic; 


%% parameters
if isfield(pm,'lambda'); 
    lambda = pm.lambda; 
else
    lambda = 1e-5;  % default value
end
% parameter for ADMM
if isfield(pm,'delta'); 
    delta = pm.delta; 
else
    delta = 100 * lambda;
end
% maximum number of iterations
if isfield(pm,'maxit'); 
    maxit = pm.maxit; 
else 
    maxit = 5*N; % default value
end
% initial guess
if isfield(pm,'x0'); 
    x0 = pm.x0; 
else 
    x0 = zeros(N,1); % initial guess
end
if isfield(pm,'xg'); 
    xg = pm.xg; 
else 
    xg = x0;
end
if isfield(pm,'reltol'); 
    reltol = pm.reltol; 
else 
    reltol  = 1e-6; 
end



%% pre-computing/initialize
AAt     = A*A';
L       = chol(speye(M) + 1/delta*AAt, 'lower');
L       = sparse(L);
U       = sparse(L');   

x       = zeros(N,1);
Atb     = A'*b;
y       = x0; 
u       = x;

obj         = @(x) .5*norm(A*x-b)^2 + lambda*norm(x,1);
output.pm   = pm;

for it = 1:maxit                
    %update x
    x       = shrink(y - u, lambda/delta);

    %update y
    yold    = y;
    rhs     = Atb  + delta * (x + u);
    y       = rhs/delta - (A'*(U\(L\(A*rhs))))/delta^2;
    
    %update u
    u       = u + x - y;
    
    % stop conditions & outputs
    relerr      = norm(yold - y)/max([norm(yold), norm(y), eps]);
    residual    = norm(A*x - b)/norm(b);
    
    output.relerr(it)   = relerr;
    output.obj(it)      = obj(x);
    output.time(it)     = toc(start_time);
    output.res(it)      = residual;
    output.err(it)      = norm(x - xg)/norm(xg);
    
    if relerr < reltol && it > 2
        break;
    end
    
end
fx = obj(x); fxg = obj(xg);
% Evaluation 
output.error = norm(x - xg)/norm(xg);
if output.error < 1e-3 % success
     output.rate = 1;
elseif fx + eps <= fxg % model failure
        output.rate = -1;
else    % algorithm failure
        output.rate = -2;
end
end