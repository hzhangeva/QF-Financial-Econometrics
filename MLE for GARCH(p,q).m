%% 1. define the density function for GARCH(p,q) model
function res = densityfun(theta, y, p ,q)
%there are p+q+1 parameters in theta
%y is the whole time series
maxpq = max(p,q);
archpara = theta(p+2:end);
garchpara = theta(2:p+1);
%yt = y(maxpq+1:end);
sigmat = zeros(length(y),1);
sigmat(1:maxpq) = std(y);
for i=(maxpq+1):length(y)
sigmat(i) = sqrt(theta(1) + sum(sigmat(i-p:i-1).*sigmat(i-p:i-
1).*garchpara)+sum(y(i-q:i-1).*y(i-q:i-1).*archpara));
end
res = 1/sqrt(2*pi)./sigmat.*exp(-y.^2./(2*(sigmat.^2)));
end
%% 2. define the scaled log likelihood function
function res = likelihood(func_handle, theta, y, p, q)
res = -sum(log(func_handle(theta, y, p ,q)))/(length(y));
end
%% 3. define the MLE function
function [theta_hat, llik, lower_CI, upper_CI, p_value] = MLE(y, theta0,
sig_level, hypothesis, p, q)
%K is length(theta_hat)
[theta_hat,llik,~,~,~,hessian] = fminunc(@(theta) likelihood(@densityfun,
theta, y, p, q),theta0);
K = length(theta_hat);
%disp(K)
%pay attention here
n = length(y);
variance_mat = inv(hessian);
sigma_hat = zeros(1,K);
for i=1:K
sigma_hat(i) = sqrt(variance_mat(i,i))/sqrt(n);
end
%disp(sigma_hat)
z_statistic = (theta_hat-hypothesis)./sigma_hat;
p_value = 2*normcdf(-abs(z_statistic),0,1);
%calculate confidence interval
threshold = abs(icdf('Normal',sig_level/2,0,1));
lower_CI = hypothesis - threshold * sigma_hat;
upper_CI = hypothesis + threshold * sigma_hat;
end