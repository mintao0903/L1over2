function out = x_subR_iterative_new_2(qq, rho)
%% the function is coded  by min tao  to solve the proximity of L1/L2 
%% the basic idea to start from the optimality of this subproblem.
%% use iterative method to solve equations, and push the solution on one of the piece for the qudratic function a
%% correct some typos on Oct 31
%% test by many examples, measured by the residual of KKT system
signQ = sign(qq); qq = abs(qq);  [q,index] = sort(qq,'descend');
N = length(q);    t(1) = 1; t(2) = N; iter = 500;
t0 = floor((t(2) + t(1))/2); out.flag = 0; flag3 = 0;
if rho > (1/(q(1)^2));
    flag3 = 1;
% elseif rho < 1/(q(1)^2) %zero solution
%     out.sol = zeros(N,1);  out.a = 0; out.r = 0; out.nnz = 0;
else %one sparse solution
    aa = zeros(N,1); aa(index(1)) = q(1);
    onesp =  aa.*signQ; out.sol = onesp;
    out.a = q(1); out.r = q(1); out.nnz = 1;
end;
while (t(2) - t(1) > 1) && flag3
   qt = q(1:t0);
   nq = norm(qt,1);
   out1 = getR(rho,  qt,  nq,  t0, iter);
   r = abs(out1.r);
   if out1.flag == 1
       if (q(t0+1)-1/(rho*r)>0)
           t(1) = t0;
       else
           t(2) = t0;
       end
   else
       r = out1.r;
       a =  r^3*( rho - sqrt(rho^2 - 4*(rho*nq - t0/r)/(r^3)))/2;
       flag2 = ((q(t0) - 1/(rho*r))*(1 - a/(rho*r^3))>0);
       if flag2
          if ((q(t0+1) - 1/(rho*r))*(1 - a/(rho*r^3)) <= 0)      
           numb = 1 - a/(rho*r^3); xx = (qt - 1/(rho*r))/numb;
           aa = zeros(N,1); aa(1:t0) = xx; X(index) = aa;
           out.sol = X'.*signQ;  out.flag = 1; out.nnz = t0; out.a = a; out.r = r;
           return;
           else 
           t(1) = t0 + 1;
           end
       else
           t(2) = t0;
       end
   end
   t0 = floor((t(2) + t(1))/2);
end

if flag3
    if t0 == 1;
      aa = zeros(N,1); aa(index(1)) = q(1); onesp =  aa.*signQ;
      out.sol = onesp; out.nnz = t0; out.a = max(abs(qq)); out.r = out.a;
    else
     qt = q(1:t0); q2n = sum(qt.^2);
     nq = norm(qt,1);out1 = getR(rho, qt, nq,  t0, iter);
     r = out1.r; a =  nq - rho*(q2n*r-r^3);
     numb = 1 - a/(rho*r^3); xx = (qt - 1/(rho*r))/numb;
     aa = zeros(N,1); aa(1:t0) = xx; X(index) = aa;
     out.sol = X'.*signQ;  out.nnz = t0; out.a = a; out.r  = r;
    end
end;
end

function out = getR(rho, qt, nq,  t0, iter)
   phat = -norm(qt)^2/3; 
   b = sqrt(abs(phat)); 
   r  = nthroot(4*nq/rho,3);  out.flag = 0;
   for ii = 1: iter
       rt = r; delta = rho*nq - t0/r;
       Del = rho^2 - 4*(delta)/r^3;
      if Del<0
           out.r = r; out.flag = 1;%for this t0, we can not find the pair (a,r) such that kkt sty
           return;
      else
         a = r^3*( rho - sqrt(Del))/2;
         qhat = (nq - a)/(2*rho); 
         phi = acos(qhat/b^3);
         r = 2*b*cos(pi/3-phi/3);
%          qhat = (nq - a)/rho;
%          del = (qhat/2)^2 + (phat/3)^3; pqhat = (del)^(1/2);
%          r = (-qhat/2 + pqhat)^(1/3) + (-qhat/2 - pqhat)^(1/3);
%          r = abs(r);
       end
       if abs(r - rt)/r < 1e-3
           out.r = r;
           break;
       end;
        
   end
   out.r = r;
end