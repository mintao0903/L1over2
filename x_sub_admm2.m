function out = x_sub_admm2(x,q,rho,itr)
%%
%sovle  min norm(x,1)/norm(y,2) + rho*norm(x - q)/2
%       s.t. x = y
%       --x: initial point     

y = x;
v = zeros(length(y),1);
beta = 100;
obj = [];
for ii = 1:itr
    ny = norm(y);
    xx = mShrink((rho*q + beta*y - v)/(rho + beta),1/((rho + beta)*ny));
    d = xx + v/beta;
    tt = mfindroot(norm(xx,1),norm(d),beta);
    y = tt*d;
    v = v + beta*(xx - y); 
%     obj = [obj, norm(xx,1)/norm(xx) + (rho*norm(xx - q)^2)/2];
end
out.sol = xx;
out.obj = obj;
end
% plot(obj)