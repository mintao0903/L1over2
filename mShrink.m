function v = mShrink(s,lambda)

v = sign(s).*(max(abs(s) - lambda , 0));

end