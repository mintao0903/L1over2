%% construct one specific example to show that the stationary point of the unconstrained model consistent
%% Written by Min Tao
clc; close all;
pm = [];
%% parameter settings
M = 2*64; N = 2*1024;   K = 10;             % sparsity
matrixtype = 3; 
F = 5;  r = 0.7;
%% construct sensing matrix
switch matrixtype 
    case 1
%         A   = randn(M,N); % Gaussian matrix
%         A   = A / norm(A);    
       sigma = r*ones(N);
       sigma = sigma + (1-r)*eye(N);
       mu = zeros(1,N);
       A = mvnrnd(mu,sigma,M);
%         A   = A / norm(A); 
    case 2    
       w = rand(M,1); W_col = w*ones(1,N); k_row = ones(M,1)*(1:N);
       A = cos(2*pi*W_col.*k_row/F)/sqrt(M);
%          A   = A / norm(A);   
    case 3
        A   = dctmtx(N); % dct matrix
        idx = randperm(N-1);
        A   = A([1 idx(1:M-1)+1],:); % randomly select m rows but always include row 1
%         A   = A / norm(A);

end
%% construct sparse ground-truth 
x_ref = zeros(N,1); % true vector
xs = randn(K,1); L=1;
x_ref(randsample_separated(N, K, L)) = xs;

%% Given lambda, construct b, so that x is a stationary point
lambda = 1e-2;
x = x_ref; suppt= find(x~=0);
[b,y,w,output] =  construct_L1overL2(A,x,lambda);
%% check optimality
err1 = norm(lambda * (w/norm(x) - x *norm(x,1)/ norm(x)^3) + A' * (A * x - b));
err2 = max([max(w)-1,min(w) + 1,norm(w(x > 0) - 1),norm(w(x < 0) + 1)]);
%% construct example and set initial value
xg = x;   sigma = 0;
pm.delta = normest(A*A',1e-2)*sqrt(2); pm.xg = xg; 
pmL1 = pm;      
pm.maxit = 5*N; pmL1.std = sigma;
pmL1.lambda = lambda;
obj         = @(x) .5*norm(A*x-b)^2 + lambda*(norm(x,1)/norm(x));

%% initialization with inaccurate L1 solution
[xL1, out_L1]     =  CS_L1_uncon_ADMM(A,b,pmL1); 
error_L1          =  norm(xg - xL1);
pm = pmL1;
pm.x0             = xL1;       
%% 
pm.reltol = 1e-8; 
%         pm.lambda = 0.2;%% balance parameters for L1/L2: min_x .5||Ax-b||^2 + lambda(|x|_1/|x|_2)
rho = 100*pm.lambda;
t2 = cputime;
[x_2, out_2] = L1L2_xsub_admm3(A,b,pm,rho);%%% proximal
t2 = cputime - t2; supptadmmp = find(x_2~=0);
% intersect(supptadmmp,suppt)==suppt;
error_admmp = norm(xg - x_2);
pm.inneritr = 10;
t3 = cputime;
[x_3,out_3] = L1L2_xsub_admm2(A,b,pm,rho);
t3 = cputime - t3;  supptadmmi = find(x_3~=0); 
% intersect(  supptadmmi , suppt)==suppt;
error_admmi = norm(xg - x_3);


figure(1)
semilogy(1:length(out_2.obj),out_2.obj,'-r','LineWidth',1.5);
hold on;
semilogy(1:length(out_3.obj),out_3.obj,':b','LineWidth',2.5);
LEG = legend('ADMM$_p$','ADMM$_i$','Location','best');
set(LEG,'Fontsize',12,'Interpreter','latex');
axis tight;


figure(2)
semilogy(1:length(out_2.err),out_2.err,'-r','LineWidth',1.5);
hold on;
semilogy(1:length(out_3.err),out_3.err,':b','LineWidth',2.5);
LEG = legend('ADMM$_p$','ADMM$_i$','Location','best');
set(LEG,'Fontsize',12,'Interpreter','latex');
axis tight;
figure(3)
semilogy(1:length(out_2.nnz),out_2.nnz,'r*','LineWidth',1.5);
hold on;
semilogy(1:length(out_3.nnz),out_3.nnz,'bs','LineWidth',2.5);
LEG = legend('ADMM$_p$','ADMM$_i$','Location','best');
set(LEG,'Fontsize',12,'Interpreter','latex');
axis tight;


fprintf('err1:%6.2e,err2:%6.2e\n',err1,err2);
fprintf('Err:ADMMp:%6.2e, ADMMi:%6.2e\n',error_admmp,error_admmi);
fprintf('T:ADMMp:%6.2e, ADMMi:%6.2e\n',t2,t3);
fprintf('nnz:ADMMp:%6.2e, ADMMi:%6.2e\n',out_2.nnz(end),out_3.nnz(end));

      