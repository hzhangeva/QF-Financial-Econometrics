% 1. Function for simulating GARCH(p,q) time series
function [y,sigma] = simuGARCH(p,p_val,q,q_val,serieslen,const)
%discard 500 initial points
maxpq = max(p,q);
y = zeros(serieslen+500,1);
sigma = zeros(serieslen+500,1);
for i = 1:maxpq
y(i) = normrnd(0,1);
sigma(i) = normrnd(0,1);
end
for i = (maxpq+1):(serieslen+500)
sigma(i) = sqrt(const + sum(sigma(i-p:i-1).*sigma(i-p:i-
1).*p_val)+sum(y(i-q:i-1).*y(i-q:i-1).*q_val));
y(i) = sigma(i)*normrnd(0,1);
end
y = y(501:end);
sigma = sigma(501:end);
end
% 2. Function perform power/size analysis
function [p] =
sizetest(n,p,q,p_val,q_val,serieslen,const,theta0,sig_level,hypothesis)
p_value_colletion = zeros(p+q+1,n);
for i = 1:n
%simulate data
[y,~] = simuGARCH(p,p_val,q,q_val,serieslen,const);
%MLE estimation
[~, ~, ~, ~, p_value] = MLE(y, theta0, sig_level, hypothesis, p, q);
p_value_colletion(:,i) = p_value;
end
bool=p_value_colletion<sig_level;
p = sum(bool,2)/n;
end