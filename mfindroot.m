function [tau] = mfindroot(c,eta,rho)

if c == 0 || eta == 0 
    tau = 1;
else
    a = (27*c/(rho*(eta^3))) + 2;
    C = nthroot((a + (a^2 - 4)^0.5)/2,3);
    tau = (1 + C + 1/C)/3;
end

end